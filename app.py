import heapq
import json
import os
import random
import re
import statistics

import networkx as nx
import openai
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

openai.api_key = os.getenv("OPENAI_API_KEY")


class TrafficMap:
    """
    Represents the road network as a weighted undirected graph.
    Nodes represent intersections and edges represent road segments.
    """

    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, weight):
        """Add an undirected edge between nodes u and v with a given weight."""
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))


class TravelerRequest:
    """
    Contains the details for a traveler's journey.
    """

    def __init__(self, vehicle_type, start, destination):
        self.vehicle_type = vehicle_type
        self.start = start
        self.destination = destination


class Agent:
    """
    Represents an agent (traveler/vehicle). The agent plans its route based on its traveler
    request by either using the baseline shortest path or via an LLM.
    """

    def __init__(self, name: str, request: TravelerRequest):
        self.name = name
        self.request = request
        self.route = None
        self.schedule = []
        self.planned_time = 0

        self.current_index_on_route = 0
        self.distance_on_edge = 0
        self.location_track = []

        self.actual_time = 0
        self.finished = False

    def plan_route(self, traffic_map: TrafficMap, method="shortest_path"):
        """
        Plan a route based on the chosen method.
          - "shortest_path": use Dijkstra's algorithm.
          - "llm": use the OpenAI LLM (via API call) for route planning.
        """
        if method == "llm" or method == "gpt-4o-mini":
            route = self.plan_route_llm(traffic_map)
        elif method == "gpt-3.5-turbo":
            route = self.plan_route_llm(traffic_map, model="gpt-3.5-turbo")
        else:
            route = self.plan_route_shortest_path(
                traffic_map, self.request.start, self.request.destination
            )

        self.route = route
        if route is None:
            return None
        # Build a schedule: assume constant speed so that each edge takes its weight in time.
        t = 0
        self.schedule = []
        for i in range(len(route) - 1):
            start_node = route[i]
            end_node = route[i + 1]
            weight = next(
                (w for (n, w) in traffic_map.graph[start_node] if n == end_node), 1
            )
            segment = {
                "start_node": start_node,
                "end_node": end_node,
                "start_time": t,
                "end_time": t + weight,
            }
            self.schedule.append(segment)
            t += weight
        self.planned_time = t
        return route

    def plan_route_shortest_path(self, traffic_map: TrafficMap, start, goal):
        """
        Computes the shortest path from start to goal using Dijkstra's algorithm.
        """
        if start == goal:
            return [start]

        queue = [(0, start, [start])]
        visited = set()

        while queue:
            (dist, node, path) = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node == goal:
                return path
            for neighbor, weight in traffic_map.graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (dist + weight, neighbor, path + [neighbor]))
        return None

    def plan_route_llm(self, traffic_map: TrafficMap, model="gpt-4o-mini"):
        """
        Uses the OpenAI LLM to plan a route.
        Constructs a prompt from the map and traveler request, calls the LLM, and parses the result.
        Falls back to the shortest path if parsing fails.
        """
        prompt = create_map_prompt(traffic_map, self.request)
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in traffic simulation and route planning.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            json_array = re.search(r"\[.*\]", answer, re.DOTALL)
            if json_array:
                route = json.loads(json_array.group())
                if isinstance(route, list) and all(
                    isinstance(elem, str) for elem in route
                ):
                    return route
            # Fallback: attempt a direct evaluation.
            route = eval(answer)
            if isinstance(route, list):
                return route
        except Exception as e:
            print("Error calling the LLM or parsing its output:", e)

        print("LLM route planning failed, falling back to shortest path")
        return self.plan_route_shortest_path(
            traffic_map, self.request.start, self.request.destination
        )

    def position_at_time(self, t, pos_dict):
        """
        Given a simulation time 't' and a mapping of node labels to (x,y) positions,
        returns the current (x,y) coordinate of the agent.
          - Before departure, the agent is at the starting node.
          - While traversing an edge, it linearly interpolates between the nodes.
          - After finishing, it remains at the destination.
        """
        if not self.schedule or t < self.schedule[0]["start_time"]:
            return pos_dict[self.route[0]]

        for seg in self.schedule:
            if seg["start_time"] <= t < seg["end_time"]:
                alpha = (t - seg["start_time"]) / (seg["end_time"] - seg["start_time"])
                x0, y0 = pos_dict[seg["start_node"]]
                x1, y1 = pos_dict[seg["end_node"]]
                return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0))
        return pos_dict[self.route[-1]]


class TrafficSimulation:
    """
    Orchestrates the overall simulation. Holds the map and agents and provides a
    way to compute congestion over time.
    """

    def __init__(self, traffic_map: TrafficMap):
        self.traffic_map = traffic_map
        self.agents = []

    def add_agent(
        self,
        agent_name: str,
        traveler_request: TravelerRequest,
        plan_method="shortest_path",
    ):
        """
        Create a new agent from the traveler request using the specified planning method
        (either "shortest_path" for baseline or "llm" to use the OpenAI LLM) and add it to the simulation.
        """
        agent = Agent(agent_name, traveler_request)
        agent.plan_route(self.traffic_map, method=plan_method)
        self.agents.append(agent)

    def compute_congestion_at_time(self, t):
        """
        Computes a simple congestion metric at time t. For agents in transit on an edge,
        count the usage per edge and sum the extra counts (usage - 1) for each edge.
        """
        edge_usage = {}
        for agent in self.agents:
            if not agent.schedule:
                continue
            for seg in agent.schedule:
                if seg["start_time"] <= t < seg["end_time"]:
                    edge = tuple(sorted((seg["start_node"], seg["end_node"])))
                    edge_usage[edge] = edge_usage.get(edge, 0) + 1
                    break
        congestion_metric = sum(max(0, count - 1) for count in edge_usage.values())
        return congestion_metric, edge_usage

    def get_agent_by_name(self, agent_name: str):
        """
        Finds the agent in the TrafficSimulation's list of agents based on name.
        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        return None


def create_map_prompt(traffic_map: TrafficMap, req: TravelerRequest) -> str:
    """
    Formats the map layout and traveler request as a prompt for the LLM.
    """
    # Unique nodes.
    nodes = sorted(set(traffic_map.graph.keys()))
    # Unique edges - avoid duplicates (since the graph is undirected).
    visited = set()
    edges_list = []
    for u in traffic_map.graph:
        for v, weight in traffic_map.graph[u]:
            edge = tuple(sorted((u, v)))
            if edge not in visited:
                visited.add(edge)
                edges_list.append(f"{edge[0]} - {edge[1]} (weight {weight})")
    prompt = (
        "Here is a map of road intersections and the connecting road segments:\n"
        f"Nodes: {', '.join(nodes)}\n"
        f"Edges: {', '.join(edges_list)}\n\n"
        "Traveler details:\n"
        f"  Vehicle type: {req.vehicle_type}\n"
        f"  Start: {req.start}\n"
        f"  Destination: {req.destination}\n\n"
        "Based on the above map and traveler details, please provide the best route as a JSON array of node names."
        "Pay special attention to speed, where higher weights take longer to traverse, and also congestion predictions."
        'Example output: ["A", "B", "C"]. Provide only the JSON array and no additional commentary.'
    )
    return prompt


def generate_layout(traffic_map):
    G = nx.Graph()
    for node, neighbors in traffic_map.graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)

    # Choose layout algorithm
    # pos = nx.spring_layout(G, seed=42)
    pos = nx.kamada_kawai_layout(G)
    return pos


def interpolate_position(agent, frame_index, node_positions, traffic_map):
    if frame_index >= len(agent.location_track):
        return node_positions[agent.route[-1]]

    current_node, distance_on_edge = agent.location_track[frame_index]
    try:
        current_index = agent.route.index(current_node)
    except ValueError:
        current_index = 0

    if current_index == len(agent.route) - 1:
        return node_positions[current_node]

    next_node = agent.route[current_index + 1]

    edge_weight = None
    for neighbor, weight in traffic_map.graph[current_node]:
        if neighbor == next_node:
            edge_weight = weight
            break
    if edge_weight is None or edge_weight == 0:
        return node_positions[current_node]

    fraction = distance_on_edge / edge_weight
    x0, y0 = node_positions[current_node]
    x1, y1 = node_positions[next_node]
    x_pos = x0 + fraction * (x1 - x0)
    y_pos = y0 + fraction * (y1 - y0)
    return (x_pos, y_pos)


def generate_graphs(simulation, congestion_track, filename: str, title: str):
    
    node_positions = generate_layout(simulation.traffic_map)

    edge_traces = []
    for node, neighbors in simulation.traffic_map.graph.items():
        for neighbor, weight in neighbors:
            if node < neighbor:
                x0, y0 = node_positions[node]
                x1, y1 = node_positions[neighbor]
                width = weight * 2
                et = go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=width, color="#888"),
                    hoverinfo="none",
                    showlegend=False,
                )
                edge_traces.append(et)

    node_x = []
    node_y = []
    node_text = []
    for node, (x, y) in node_positions.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(color="blue", size=10),
        hoverinfo="text",
        showlegend=False,
    )

    colors = px.colors.qualitative.Plotly
    vehicle_x = []
    vehicle_y = []
    vehicle_text = []
    vehicle_colors = []
    for i, agent in enumerate(simulation.agents):
        pos = interpolate_position(agent, 0, node_positions, simulation.traffic_map)
        vehicle_x.append(pos[0])
        vehicle_y.append(pos[1])
        vehicle_text.append(agent.name)
        vehicle_colors.append(colors[i % len(colors)])

    vehicle_trace = go.Scatter(
        x=vehicle_x,
        y=vehicle_y,
        mode="markers+text",
        text=vehicle_text,
        textposition="top center",
        marker=dict(size=12, color=vehicle_colors),
        name="Vehicles",
    )

    congestion_x, congestion_y = list(range(len(congestion_track))), congestion_track

    congestion_trace = go.Scatter(
        x=congestion_x,
        y=congestion_y,
        mode="lines+markers",
        line=dict(color="red"),
        name="Congestion",
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=["Traffic Simulator", "Congestion Over Time"],
    )

    for et in edge_traces:
        fig.add_trace(et, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(vehicle_trace, row=1, col=1)
    fig.add_trace(congestion_trace, row=1, col=2)

    num_edge_traces = len(edge_traces)
    vehicle_index = num_edge_traces + 1
    congestion_index = num_edge_traces + 2

    max_frames = max(len(agent.location_track) for agent in simulation.agents)

    frames = []
    for frame_idx in range(max_frames):
        # Update vehicle positions.
        frame_vehicle_x = []
        frame_vehicle_y = []
        for agent in simulation.agents:
            pos = interpolate_position(
                agent, frame_idx, node_positions, simulation.traffic_map
            )
            frame_vehicle_x.append(pos[0])
            frame_vehicle_y.append(pos[1])

        # Update congestion chart data.
        frame_congestion_x = list(range(frame_idx + 1))
        frame_congestion_y = congestion_track[: frame_idx + 1]

        # The frame updates only the vehicle and congestion traces.
        frame_data = [
            go.Scatter(
                x=frame_vehicle_x,
                y=frame_vehicle_y,
                mode="markers+text",
                text=vehicle_text,
                textposition="top center",
                marker=dict(size=12, color=vehicle_colors),
            ),
            go.Scatter(
                x=frame_congestion_x,
                y=frame_congestion_y,
                mode="lines+markers",
                line=dict(color="red"),
            ),
        ]
        frames.append(
            go.Frame(
                data=frame_data,
                traces=[vehicle_index, congestion_index],
                name=str(frame_idx),
            )
        )

    fig.frames = frames

    # play/pause controls plus an animation slider
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 85},
                "showactive": False,
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(k)],
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k in range(max_frames)
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "y": -0.05,
                "currentvalue": {
                    "visible": True,
                    "prefix": "Frame: ",
                    "xanchor": "right",
                    "font": {"size": 12},
                },
                "len": 0.8,
            }
        ],
        title=title,
        showlegend=False,
    )

    fig.update_xaxes(title_text="", row=1, col=1, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="", row=1, col=1, showgrid=False, zeroline=False)
    fig.update_xaxes(title_text="Frame", row=1, col=2)
    fig.update_yaxes(title_text="Congestion", row=1, col=2)
    pio.write_html(fig, file=filename, auto_open=False)


def run_simulation(simulation: TrafficSimulation, filename: str, title: str):
    all_finished = False
    occupied_edges = {}
    sim_step_count = 0
    congestion_track = []
    while not all_finished:
        sim_step_count += 1
        for agent in simulation.agents:
            agent.location_track.append(
                (agent.route[agent.current_index_on_route], agent.distance_on_edge)
            )
            if agent.finished:
                continue
            # print(f'processing agent: {agent.name}')
            step_size = 1 if agent.request.vehicle_type == "car" else 0.5
            if agent.current_index_on_route < len(agent.route) - 1:
                edge_start = agent.route[agent.current_index_on_route]
                target_edge_destination = agent.route[agent.current_index_on_route + 1]
                edges = simulation.traffic_map.graph[edge_start]

                target_edges = [x for x in edges if x[0] == target_edge_destination]

                if len(target_edges) == 0:
                    print(f"invalid route. Terminating for agent {agent.name}")
                    agent.finished = True
                    continue
                target_edge = target_edges[0]
                edge_destination = target_edge[0]
                edge_weight = target_edge[1]

                if edge_start not in occupied_edges:
                    occupied_edges[edge_start] = {edge_destination: {agent.name}}
                elif edge_destination not in occupied_edges[edge_start]:
                    occupied_edges[edge_start][edge_destination] = {agent.name}
                else:
                    occupied_edges[edge_start][edge_destination].add(agent.name)

                agent.distance_on_edge += step_size / len(
                    occupied_edges[edge_start][edge_destination]
                )

                if agent.distance_on_edge >= edge_weight:
                    agent.distance_on_edge = agent.distance_on_edge - edge_weight
                    agent.current_index_on_route += 1
                    occupied_edges[edge_start][edge_destination].remove(agent.name)

                    if agent.current_index_on_route < len(agent.route) - 1:

                        edge_start = agent.route[agent.current_index_on_route]
                        edge_destination = agent.route[agent.current_index_on_route + 1]

                        if edge_start not in occupied_edges:
                            occupied_edges[edge_start] = {
                                edge_destination: {agent.name}
                            }
                        elif edge_destination not in occupied_edges[edge_start]:
                            occupied_edges[edge_start][edge_destination] = {agent.name}
                        else:
                            occupied_edges[edge_start][edge_destination].add(agent.name)

            elif agent.current_index_on_route == len(agent.route) - 1:
                agent.finished = True
                # print(f"agent {agent.name} finished at {sim_step_count - 1}")
                agent.actual_time = sim_step_count - 1

        congs = []
        for occupied_edge_start in occupied_edges:
            congs.append(
                max(
                    [
                        len(occupied_edges[occupied_edge_start][occ_edge_dest])
                        for occ_edge_dest in occupied_edges[occupied_edge_start]
                    ]
                )
            )
        congestion_track.append(max(congs))
        all_finished = (
            len([agent for agent in simulation.agents if agent.finished == False]) == 0
        )

    generate_graphs(simulation, congestion_track, filename, title)


def random_letter(start, end, excluding=[]):
    c = chr(random.randint(ord(start), ord(end)))
    while c in excluding:
        c = chr(random.randint(ord(start), ord(end)))
    return c

def random_vehicle():
    return 'car' if random.randint(0,1) == 1 else 'truck'


def generate_traveler_request(node_start_letter, node_end_letter):
    start = random_letter(node_start_letter, node_end_letter)
    end = random_letter(node_start_letter, node_end_letter, excluding=[start])
    return TravelerRequest(vehicle_type=random_vehicle(), start=start, destination=end)

def prepare_sim_1(agent_count=50):
    # Map layout creation
    traffic_map = TrafficMap()
    traffic_map.add_edge("A", "B", 1)
    traffic_map.add_edge("A", "D", 1)
    traffic_map.add_edge("A", "M", 3)
    traffic_map.add_edge("B", "E", 1)
    traffic_map.add_edge("B", "C", 1)
    traffic_map.add_edge("C", "N", 0.5)
    traffic_map.add_edge("C", "O", 0.5)
    traffic_map.add_edge("C", "F", 1)
    traffic_map.add_edge("N", "F", 0.5)
    traffic_map.add_edge("N", "O", 0.5)
    traffic_map.add_edge("F", "O", 0.5)
    traffic_map.add_edge("D", "E", 1)
    traffic_map.add_edge("D", "G", 1)
    traffic_map.add_edge("E", "F", 1)
    traffic_map.add_edge("E", "H", 1)
    traffic_map.add_edge("G", "H", 1)
    traffic_map.add_edge("G", "J", 1)
    traffic_map.add_edge("M", "J", 1)
    traffic_map.add_edge("K", "J", 1)
    traffic_map.add_edge("K", "H", 1)
    traffic_map.add_edge("K", "L", 1)
    traffic_map.add_edge("M", "L", 2)
    traffic_map.add_edge("M", "I", 4)
    traffic_map.add_edge("L", "I", 1)
    traffic_map.add_edge("H", "I", 1)
    traffic_map.add_edge("F", "I", 1)
    traffic_map.add_edge("F", "P", 0.5)
    traffic_map.add_edge("F", "P", 0.5)
    traffic_map.add_edge("H", "P", 0.5)
    traffic_map.add_edge("H", "Q", 0.5)
    traffic_map.add_edge("I", "Q", 0.5)

    # Define traveler requests.
    traveler_requests = []
    for i in range(agent_count):
        traveler_requests.append(generate_traveler_request('A', 'Q'))
    return traffic_map, traveler_requests

def prepare_sim_2(agent_count=10):
    # Map layout creation
    traffic_map = TrafficMap()
    traffic_map.add_edge("A", "B", 1)
    traffic_map.add_edge("B", "C", 1)
    traffic_map.add_edge("C", "D", 1)
    traffic_map.add_edge("D", "E", 1)
    traffic_map.add_edge("E", "A", 1)

    # Define traveler requests.
    traveler_requests = []
    for i in range(agent_count):
        traveler_requests.append(generate_traveler_request('A', 'E'))
    return traffic_map, traveler_requests

def prepare_sim_3(agent_count=20):
    # Map layout creation
    traffic_map = TrafficMap()
    traffic_map.add_edge("A", "B", 1)
    traffic_map.add_edge("B", "C", 2)
    traffic_map.add_edge("B", "E", 4)
    traffic_map.add_edge("C", "D", 1)
    traffic_map.add_edge("D", "B", 2)
    traffic_map.add_edge("E", "F", 1)
    traffic_map.add_edge("F", "G", 2)
    traffic_map.add_edge("G", "E", 1)

    # Define traveler requests.
    traveler_requests = []
    for i in range(agent_count):
        traveler_requests.append(generate_traveler_request('A', 'G'))
    return traffic_map, traveler_requests


def run_sim(traffic_map, traveler_requests, tag):
    titles = [
        "Shortest Path",
        "GPT 4o Mini",
        "GPT 3.5 Turbo"
    ]

    filenames = [
        f"out/traffic-sim-shortest-path-{tag}.html",
        f"out/traffic-sim-gpt-4o-mini-{tag}.html",
        f"out/traffic-sim-gpt-3-5-turbo-{tag}.html",
    ]

    print("=== SHORTEST PATH ===")
    simulation_baseline = TrafficSimulation(traffic_map)
    for i, tr in enumerate(traveler_requests):
        simulation_baseline.add_agent(f"traveler{i}", tr, plan_method="shortest_path")
    run_simulation(simulation_baseline, filenames[0], titles[0])

    congestion_impacts = []
    for agent in simulation_baseline.agents:
        congestion_impact = agent.actual_time - agent.planned_time
        congestion_impacts.append(congestion_impact)

    shortest_mean = statistics.mean(congestion_impacts)
    shortest_median = statistics.median(congestion_impacts)
    print(f'Average Congestion Impact: {shortest_mean}')
    print(f'Median Congestion Impact: {shortest_median}')

    print("=== GPT 4o mini ===")
    simulation_gpt_4o_mini = TrafficSimulation(traffic_map)
    for i, tr in enumerate(traveler_requests):
        simulation_gpt_4o_mini.add_agent(f"traveler{i}", tr, plan_method="llm")
    run_simulation(simulation_gpt_4o_mini, filenames[1], titles[1])

    congestion_impacts = []
    for agent in simulation_gpt_4o_mini.agents:
        congestion_impact = agent.actual_time - agent.planned_time
        congestion_impacts.append(congestion_impact)

    gpt4o_mean = statistics.mean(congestion_impacts)
    gpt4o_median = statistics.median(congestion_impacts)
    print(f'Average Congestion Impact: {gpt4o_mean}')
    print(f'Median Congestion Impact: {gpt4o_median}')

    print("=== GPT 3.5 turbo ===")
    simulation_gpt_35_turbo = TrafficSimulation(traffic_map)
    for i, tr in enumerate(traveler_requests):
        simulation_gpt_35_turbo.add_agent(f"traveler{i}", tr, plan_method="gpt-3.5-turbo")
    run_simulation(simulation_gpt_35_turbo, filenames[2], titles[2])

    congestion_impacts = []
    for agent in simulation_gpt_35_turbo.agents:
        congestion_impact = agent.actual_time - agent.planned_time
        congestion_impacts.append(congestion_impact)

    gpt35_mean = statistics.mean(congestion_impacts)
    gpt35_median = statistics.median(congestion_impacts)

    print(f'Average Congestion Impact: {gpt35_mean}')
    print(f'Median Congestion Impact: {gpt35_median}')

    return {
        "shortest_mean": shortest_mean,
        "shortest_median": shortest_median,
        "gpt4o_mean": gpt4o_mean,
        "gpt4o_median": gpt4o_median,
        "gpt35_mean": gpt35_mean,
        "gpt35_median": gpt35_median,
    }

if __name__ == "__main__":

    traffic_map, traveler_requests = prepare_sim_1(60)
    for i in range(30):
        results = run_sim(traffic_map, traveler_requests, f'sim1-{i}')
        results['iteration'] = i
        with open("out/sim1-results.txt", "a") as file:
            file.write(str(results))

    traffic_map, traveler_requests = prepare_sim_2(10)
    for i in range(30):
        results = run_sim(traffic_map, traveler_requests, f'sim2-{i}')
        results['iteration'] = i
        with open("out/sim2-results.txt", "a") as file:
            file.write(str(results))

    traffic_map, traveler_requests = prepare_sim_3(20)
    for i in range(30):
        results = run_sim(traffic_map, traveler_requests, f'sim3-{i}')
        results['iteration'] = i
        with open("out/sim3-results.txt", "a") as file:
            file.write(str(results))

