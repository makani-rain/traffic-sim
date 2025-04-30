
import heapq
import json
import math
import re
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import openai
import os

# Set your OpenAI API Key here.
openai.api_key = os.getenv("OPENAI_API_KEY")


# -------------------------------
# Map and Route Planning Classes
# -------------------------------
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
    
    def shortest_path(self, start, goal):
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
            for neighbor, weight in self.graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (dist + weight, neighbor, path + [neighbor]))
        return None


# -------------------------------
# Traveler Request and Agent Classes
# -------------------------------
class TravelerRequest:
    """
    Contains the details for a traveler's journey.
    """
    def __init__(self, vehicle_type, start, destination):
        self.vehicle_type = vehicle_type
        self.start = start
        self.destination = destination


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
        "Example output: [\"A\", \"B\", \"C\"]. Provide only the JSON array and no additional commentary."
    )
    return prompt


class Agent:
    """
    Represents an agent (traveler/vehicle). The agent plans its route based on its traveler
    request by either using the baseline shortest path or via an LLM.
    """
    def __init__(self, name: str, request: TravelerRequest):
        self.name = name
        self.request = request
        self.route = None        # List of nodes that form the route.
        self.schedule = []       # List of segments with time intervals.
        self.total_time = 0      # Total travel time computed from the schedule.

    def plan_route(self, traffic_map: TrafficMap, method="shortest_path"):
        """
        Plan a route based on the chosen method.
          - "shortest_path": use Dijkstra's algorithm.
          - "llm": use the OpenAI LLM (via API call) for route planning.
        """
        if method == "llm":
            route = self.plan_route_llm(traffic_map)
        else:
            route = traffic_map.shortest_path(self.request.start, self.request.destination)

        self.route = route
        if route is None:
            return None
        # Build a schedule: assume constant speed so that each edge takes its weight in time.
        t = 0
        self.schedule = []
        for i in range(len(route) - 1):
            start_node = route[i]
            end_node = route[i+1]
            weight = next((w for (n, w) in traffic_map.graph[start_node] if n == end_node), 1)
            segment = {
                "start_node": start_node,
                "end_node": end_node,
                "start_time": t,
                "end_time": t + weight
            }
            self.schedule.append(segment)
            t += weight
        self.total_time = t
        return route

    def plan_route_llm(self, traffic_map: TrafficMap):
        """
        Uses the OpenAI LLM to plan a route.
        Constructs a prompt from the map and traveler request, calls the LLM, and parses the result.
        Falls back to the shortest path if parsing fails.
        """
        prompt = create_map_prompt(traffic_map, self.request)
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in traffic simulation and route planning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            json_array = re.search(r'\[.*\]', answer, re.DOTALL)
            if json_array:
                route = json.loads(json_array.group())
                if isinstance(route, list) and all(isinstance(elem, str) for elem in route):
                    return route
            # Fallback: attempt a direct evaluation.
            route = eval(answer)
            if isinstance(route, list):
                return route
        except Exception as e:
            print("Error calling the LLM or parsing its output:", e)
        
        print("LLM route planning failed, falling back to shortest path")
        return traffic_map.shortest_path(self.request.start, self.request.destination)

    def position_at_time(self, t, pos_dict):
        """
        Given a simulation time 't' and a mapping of node labels to (x,y) positions,
        returns the current (x,y) coordinate of the agent.
          - Before departure, the agent is at the starting node.
          - While traversing an edge, it linearly interpolates between the nodes.
          - After finishing, it remains at the destination.
        """
        if not self.schedule or t < self.schedule[0]['start_time']:
            return pos_dict[self.route[0]]
        
        for seg in self.schedule:
            if seg['start_time'] <= t < seg['end_time']:
                alpha = (t - seg['start_time']) / (seg['end_time'] - seg['start_time'])
                x0, y0 = pos_dict[seg['start_node']]
                x1, y1 = pos_dict[seg['end_node']]
                return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0))
        return pos_dict[self.route[-1]]


# -------------------------------
# Simulation and Evaluation Classes
# -------------------------------
class TrafficSimulation:
    """
    Orchestrates the overall simulation. Holds the map and agents and provides a 
    way to compute congestion over time.
    """
    def __init__(self, traffic_map: TrafficMap):
        self.traffic_map = traffic_map
        self.agents = []

    def add_agent(self, agent_name: str, traveler_request: TravelerRequest, plan_method="shortest_path"):
        """
        Create a new agent from the traveler request using the specified planning method
        (either "shortest_path" for baseline or "llm" to use the OpenAI LLM) and add it to the simulation.
        """
        agent = Agent(agent_name, traveler_request)
        agent.plan_route(self.traffic_map, method=plan_method)
        self.add_planned_agent(agent)

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
                if seg['start_time'] <= t < seg['end_time']:
                    edge = tuple(sorted((seg['start_node'], seg['end_node'])))
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
        


def save_animation(simulation: TrafficSimulation, filename='simulation.gif'):
    """
    Creates an animated visualization of the simulation using FuncAnimation and saves
    the output as a GIF using the Pillow writer.
    """
    G = nx.Graph()
    for node, neighbors in simulation.traffic_map.graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)
    
    # Choose layout algorithm
    # pos = nx.spring_layout(G, seed=42)
    pos = nx.kamada_kawai_layout(G)

    # Determine total simulation time and number of frames.
    t_end = max((agent.total_time for agent in simulation.agents), default=0)
    dt = 0.1
    total_frames = int(t_end / dt) + 1

    # Precompute congestion values for the background congestion curve.
    time_steps = np.arange(0, t_end + dt, dt)
    cong_values = [simulation.compute_congestion_at_time(t)[0] for t in time_steps]

    # Set up figure and subplots.
    fig, (ax_map, ax_cong) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.4)

    # --- Left Axes: Traffic Map ---
    ax_map.set_title("Traffic Map with Agent Positions")
    
    # Group edges by weight for better visualization
    edges = G.edges(data=True)
    short_edges = [(u, v) for u, v, d in edges if d['weight'] <= 1]
    medium_edges = [(u, v) for u, v, d in edges if 1 < d['weight'] <= 2]
    long_edges = [(u, v) for u, v, d in edges if d['weight'] > 2]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax_map, node_size=500, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax_map, font_size=8)
    
    # Draw edges with color coding
    nx.draw_networkx_edges(G, pos, ax=ax_map, edgelist=short_edges, width=2, edge_color='green', label='Weight <= 1')
    nx.draw_networkx_edges(G, pos, ax=ax_map, edgelist=medium_edges, width=2, edge_color='orange', label='1 < Weight <= 2')
    nx.draw_networkx_edges(G, pos, ax=ax_map, edgelist=long_edges, width=2, edge_color='red', label='Weight > 2')
    
    # Add edge labels for weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, ax=ax_map, edge_labels=edge_labels, font_size=6)

    # Initialize agent markers with unique colors.
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    agent_markers = []
    for idx, agent in enumerate(simulation.agents):
        init_pos = agent.position_at_time(0, pos)
        marker, = ax_map.plot([init_pos[0]], [init_pos[1]], 'o', ms=10,
                              color=colors[idx % len(colors)], label=f"Agent {idx + 1}")
        agent_markers.append(marker)

    # Add a legend to the map
    # ax_map.legend(loc='best', fontsize=8)
    ax_map.axis('off')

    # --- Right Axes: Congestion Plot ---
    ax_cong.set_title("Congestion Over Time")
    ax_cong.set_xlabel("Simulation Time")
    ax_cong.set_ylabel("Congestion Metric")
    line_cong, = ax_cong.plot(time_steps, cong_values, 'b-', lw=2, label="Congestion")
    time_indicator = ax_cong.axvline(x=0, color='r', linestyle='--', lw=2)
    ax_cong.legend()

    # --- Update Function for Animation ---
    def update(frame):
        current_time = frame * dt
        # Determine current positions for all agents.
        positions = [agent.position_at_time(current_time, pos) for agent in simulation.agents]
        groups = {}
        for idx, (x, y) in enumerate(positions):
            key = (round(x, 3), round(y, 3))
            groups.setdefault(key, []).append(idx)
        for group in groups.values():
            if len(group) == 1:
                i = group[0]
                x, y = positions[i]
                agent_markers[i].set_data([x], [y])
            else:
                num = len(group)
                offset_radius = 0.03
                for j, i in enumerate(group):
                    angle = 2 * math.pi * j / num
                    dx = offset_radius * math.cos(angle)
                    dy = offset_radius * math.sin(angle)
                    x, y = positions[i]
                    agent_markers[i].set_data([x + dx], [y + dy])
        time_indicator.set_xdata([current_time])
        return agent_markers + [time_indicator]

    ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)
    ani.save(filename, writer='pillow', fps=10)
    print(f"Animation saved as {filename}")
    plt.close(fig)

def compare_route_planning(traffic_map: TrafficMap, traveler_request: TravelerRequest):
    """
    Compares the total travel time for the shortest-path algorithm and the OpenAI LLM approach.
    """
    # Create and plan agents using both methods
    shortest_path_agent = Agent("ShortestPathAgent", traveler_request)
    shortest_path_agent.plan_route(traffic_map, method="shortest_path")

    llm_agent = Agent("LLMAgent", traveler_request)
    llm_agent.plan_route(traffic_map, method="llm")

    # Print the results
    print("Comparison of Travel Time:")
    print(f"Shortest-path travel time: {shortest_path_agent.total_time}")
    print(f"LLM-based travel time: {llm_agent.total_time}")

    # Optional: Check which method was faster
    if shortest_path_agent.total_time < llm_agent.total_time:
        print("Shortest-path algorithm was faster.")
    elif shortest_path_agent.total_time > llm_agent.total_time:
        print("LLM-based planning was faster.")
    else:
        print("Both methods resulted in the same travel time.")
    
# -------------------------------
# Main: Creating the Simulation and Saving GIFs
# -------------------------------
if __name__ == "__main__":
    # Map layout creation
    traffic_map = TrafficMap()
    traffic_map.add_edge('A', 'B', 1)
    traffic_map.add_edge('A', 'D', 1)
    traffic_map.add_edge('A', 'M', 3)
    traffic_map.add_edge('B', 'E', 1)
    traffic_map.add_edge('B', 'C', 1)
    traffic_map.add_edge('C', 'N', 0.5)
    traffic_map.add_edge('C', 'O', 0.5)
    traffic_map.add_edge('C', 'F', 1)
    traffic_map.add_edge('N', 'F', 0.5)
    traffic_map.add_edge('N', 'O', 0.5)
    traffic_map.add_edge('F', 'O', 0.5)
    traffic_map.add_edge('D', 'E', 1)
    traffic_map.add_edge('D', 'G', 1)
    traffic_map.add_edge('E', 'F', 1)
    traffic_map.add_edge('E', 'H', 1)
    traffic_map.add_edge('G', 'H', 1)
    traffic_map.add_edge('G', 'J', 1)
    traffic_map.add_edge('M', 'J', 1)
    traffic_map.add_edge('K', 'J', 1)
    traffic_map.add_edge('K', 'H', 1)
    traffic_map.add_edge('K', 'L', 1)
    traffic_map.add_edge('M', 'L', 2)
    traffic_map.add_edge('M', 'I', 4)
    traffic_map.add_edge('L', 'I', 1)
    traffic_map.add_edge('H', 'I', 1)
    traffic_map.add_edge('F', 'I', 1)
    traffic_map.add_edge('F', 'P', 0.5)
    traffic_map.add_edge('F', 'P', 0.5)
    traffic_map.add_edge('H', 'P', 0.5)
    traffic_map.add_edge('H', 'Q', 0.5)
    traffic_map.add_edge('I', 'Q', 0.5)
    
    # Define traveler requests.
    traveler1 = TravelerRequest(vehicle_type="car", start="A", destination="D")
    traveler2 = TravelerRequest(vehicle_type="car", start="A", destination="D")
    traveler3 = TravelerRequest(vehicle_type="truck", start="B", destination="C")
    traveler4 = TravelerRequest(vehicle_type="car", start="D", destination="C")
    traveler5 = TravelerRequest(vehicle_type="car", start="B", destination="D")
    traveler6 = TravelerRequest(vehicle_type="car", start="E", destination="D")
    traveler7 = TravelerRequest(vehicle_type="car", start="D", destination="F")
    traveler8 = TravelerRequest(vehicle_type="car", start="D", destination="A")
    traveler9 = TravelerRequest(vehicle_type="car", start="G", destination="C")
    traveler10 = TravelerRequest(vehicle_type="car", start="A", destination="G")

    traveler11 = TravelerRequest(vehicle_type="car", start="A", destination="B")
    traveler12 = TravelerRequest(vehicle_type="car", start="A", destination="C")
    traveler13 = TravelerRequest(vehicle_type="car", start="A", destination="D")
    traveler14 = TravelerRequest(vehicle_type="car", start="A", destination="E")
    traveler15 = TravelerRequest(vehicle_type="car", start="A", destination="F")
    traveler16 = TravelerRequest(vehicle_type="car", start="A", destination="G")
    traveler17 = TravelerRequest(vehicle_type="car", start="A", destination="H")
    traveler18 = TravelerRequest(vehicle_type="car", start="A", destination="I")
    traveler19 = TravelerRequest(vehicle_type="car", start="A", destination="J")
    traveler20 = TravelerRequest(vehicle_type="car", start="A", destination="K")
    traveler21 = TravelerRequest(vehicle_type="car", start="A", destination="L")
    traveler22 = TravelerRequest(vehicle_type="car", start="A", destination="M")
    traveler23 = TravelerRequest(vehicle_type="car", start="A", destination="N")
    traveler24 = TravelerRequest(vehicle_type="car", start="A", destination="O")
    traveler25 = TravelerRequest(vehicle_type="car", start="A", destination="P")
    traveler26 = TravelerRequest(vehicle_type="car", start="A", destination="Q")

    traveler27 = TravelerRequest(vehicle_type="car", start="K", destination="A")
    traveler28 = TravelerRequest(vehicle_type="car", start="K", destination="B")
    traveler29 = TravelerRequest(vehicle_type="car", start="K", destination="C")
    traveler30 = TravelerRequest(vehicle_type="car", start="K", destination="D")
    traveler31 = TravelerRequest(vehicle_type="car", start="K", destination="E")
    traveler32 = TravelerRequest(vehicle_type="car", start="K", destination="F")
    traveler33 = TravelerRequest(vehicle_type="car", start="K", destination="G")
    traveler34 = TravelerRequest(vehicle_type="car", start="K", destination="H")
    traveler35 = TravelerRequest(vehicle_type="car", start="K", destination="I")
    traveler36 = TravelerRequest(vehicle_type="car", start="K", destination="J")
    traveler37 = TravelerRequest(vehicle_type="car", start="K", destination="L")
    traveler38 = TravelerRequest(vehicle_type="car", start="K", destination="M")
    traveler39 = TravelerRequest(vehicle_type="car", start="K", destination="N")
    traveler40 = TravelerRequest(vehicle_type="car", start="K", destination="O")
    traveler41 = TravelerRequest(vehicle_type="car", start="K", destination="P")
    traveler42 = TravelerRequest(vehicle_type="car", start="K", destination="Q")

    traveler43 = TravelerRequest(vehicle_type="car", start="O", destination="A")
    traveler44 = TravelerRequest(vehicle_type="car", start="O", destination="B")
    traveler45 = TravelerRequest(vehicle_type="car", start="O", destination="C")
    traveler46 = TravelerRequest(vehicle_type="car", start="O", destination="D")
    traveler47 = TravelerRequest(vehicle_type="car", start="O", destination="E")
    traveler48 = TravelerRequest(vehicle_type="car", start="O", destination="F")
    traveler49 = TravelerRequest(vehicle_type="car", start="O", destination="G")
    traveler50 = TravelerRequest(vehicle_type="car", start="O", destination="H")
    traveler51 = TravelerRequest(vehicle_type="car", start="O", destination="I")
    traveler52 = TravelerRequest(vehicle_type="car", start="O", destination="J")
    traveler53 = TravelerRequest(vehicle_type="car", start="O", destination="K")
    traveler54 = TravelerRequest(vehicle_type="car", start="O", destination="L")
    traveler55 = TravelerRequest(vehicle_type="car", start="O", destination="M")
    traveler56 = TravelerRequest(vehicle_type="car", start="O", destination="N")
    traveler57 = TravelerRequest(vehicle_type="car", start="O", destination="P")
    traveler58 = TravelerRequest(vehicle_type="car", start="O", destination="Q")

    # Baseline using Shortest Path
    simulation_baseline = TrafficSimulation(traffic_map)
    simulation_baseline.add_agent("traveler1", traveler1, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler2", traveler2, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler3", traveler3, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler4", traveler4, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler5", traveler5, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler6", traveler6, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler7", traveler7, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler8", traveler8, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler9", traveler9, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler10", traveler10, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler11", traveler11, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler12", traveler12, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler13", traveler13, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler14", traveler14, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler15", traveler15, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler16", traveler16, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler17", traveler17, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler18", traveler18, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler19", traveler19, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler20", traveler20, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler21", traveler21, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler22", traveler22, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler23", traveler23, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler24", traveler24, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler25", traveler25, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler26", traveler26, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler27", traveler27, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler28", traveler28, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler29", traveler29, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler30", traveler30, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler31", traveler31, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler32", traveler32, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler33", traveler33, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler34", traveler34, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler35", traveler35, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler36", traveler36, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler37", traveler37, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler38", traveler38, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler39", traveler39, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler40", traveler40, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler41", traveler41, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler42", traveler42, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler43", traveler43, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler44", traveler44, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler45", traveler45, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler46", traveler46, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler47", traveler47, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler48", traveler48, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler49", traveler49, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler50", traveler50, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler51", traveler51, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler52", traveler52, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler53", traveler53, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler54", traveler54, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler55", traveler55, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler56", traveler56, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler57", traveler57, plan_method="shortest_path")
    simulation_baseline.add_agent("traveler58", traveler58, plan_method="shortest_path")
    save_animation(simulation_baseline, filename="traffic_simulation_shortest-1.gif")

    # Use the OpenAI LLM for route planning
    simulation_llm = TrafficSimulation(traffic_map)
    simulation_llm.add_agent("traveler1", traveler1, plan_method="llm")
    simulation_llm.add_agent("traveler2", traveler2, plan_method="llm")
    simulation_llm.add_agent("traveler3", traveler3, plan_method="llm")
    simulation_llm.add_agent("traveler4", traveler4, plan_method="llm")
    simulation_llm.add_agent("traveler5", traveler5, plan_method="llm")
    simulation_llm.add_agent("traveler6", traveler6, plan_method="llm")
    simulation_llm.add_agent("traveler7", traveler7, plan_method="llm")
    simulation_llm.add_agent("traveler8", traveler8, plan_method="llm")
    simulation_llm.add_agent("traveler9", traveler9, plan_method="llm")
    simulation_llm.add_agent("traveler10", traveler10, plan_method="llm")
    simulation_llm.add_agent("traveler11", traveler11, plan_method="llm")
    simulation_llm.add_agent("traveler12", traveler12, plan_method="llm")
    simulation_llm.add_agent("traveler13", traveler13, plan_method="llm")
    simulation_llm.add_agent("traveler14", traveler14, plan_method="llm")
    simulation_llm.add_agent("traveler15", traveler15, plan_method="llm")
    simulation_llm.add_agent("traveler16", traveler16, plan_method="llm")
    simulation_llm.add_agent("traveler17", traveler17, plan_method="llm")
    simulation_llm.add_agent("traveler18", traveler18, plan_method="llm")
    simulation_llm.add_agent("traveler19", traveler19, plan_method="llm")
    simulation_llm.add_agent("traveler20", traveler20, plan_method="llm")
    simulation_llm.add_agent("traveler21", traveler21, plan_method="llm")
    simulation_llm.add_agent("traveler22", traveler22, plan_method="llm")
    simulation_llm.add_agent("traveler23", traveler23, plan_method="llm")
    simulation_llm.add_agent("traveler24", traveler24, plan_method="llm")
    simulation_llm.add_agent("traveler25", traveler25, plan_method="llm")
    simulation_llm.add_agent("traveler26", traveler26, plan_method="llm")
    simulation_llm.add_agent("traveler27", traveler27, plan_method="llm")
    simulation_llm.add_agent("traveler28", traveler28, plan_method="llm")
    simulation_llm.add_agent("traveler29", traveler29, plan_method="llm")
    simulation_llm.add_agent("traveler30", traveler30, plan_method="llm")
    simulation_llm.add_agent("traveler31", traveler31, plan_method="llm")
    simulation_llm.add_agent("traveler32", traveler32, plan_method="llm")
    simulation_llm.add_agent("traveler33", traveler33, plan_method="llm")
    simulation_llm.add_agent("traveler34", traveler34, plan_method="llm")
    simulation_llm.add_agent("traveler35", traveler35, plan_method="llm")
    simulation_llm.add_agent("traveler36", traveler36, plan_method="llm")
    simulation_llm.add_agent("traveler37", traveler37, plan_method="llm")
    simulation_llm.add_agent("traveler38", traveler38, plan_method="llm")
    simulation_llm.add_agent("traveler39", traveler39, plan_method="llm")
    simulation_llm.add_agent("traveler40", traveler40, plan_method="llm")
    simulation_llm.add_agent("traveler41", traveler41, plan_method="llm")
    simulation_llm.add_agent("traveler42", traveler42, plan_method="llm")
    simulation_llm.add_agent("traveler43", traveler43, plan_method="llm")
    simulation_llm.add_agent("traveler44", traveler44, plan_method="llm")
    simulation_llm.add_agent("traveler45", traveler45, plan_method="llm")
    simulation_llm.add_agent("traveler46", traveler46, plan_method="llm")
    simulation_llm.add_agent("traveler47", traveler47, plan_method="llm")
    simulation_llm.add_agent("traveler48", traveler48, plan_method="llm")
    simulation_llm.add_agent("traveler49", traveler49, plan_method="llm")
    simulation_llm.add_agent("traveler50", traveler50, plan_method="llm")
    simulation_llm.add_agent("traveler51", traveler51, plan_method="llm")
    simulation_llm.add_agent("traveler52", traveler52, plan_method="llm")
    simulation_llm.add_agent("traveler53", traveler53, plan_method="llm")
    simulation_llm.add_agent("traveler54", traveler54, plan_method="llm")
    simulation_llm.add_agent("traveler55", traveler55, plan_method="llm")
    simulation_llm.add_agent("traveler56", traveler56, plan_method="llm")
    simulation_llm.add_agent("traveler57", traveler57, plan_method="llm")
    simulation_llm.add_agent("traveler58", traveler58, plan_method="llm")
    save_animation(simulation_llm, filename="traffic_simulation_llm-1.gif")
    
    
    
