import matplotlib.pyplot as plt
import pandas as pd
import statistics
import json

def run(filename, sim_name):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Plot mean values
    plt.figure(figsize=(10, 5))
    mean_mean_shortest = statistics.mean(df["shortest_mean"])
    mean_mean_gpt4o = statistics.mean(df["gpt4o_mean"])
    mean_mean_gpt35 = statistics.mean(df["gpt35_mean"])
    mean_median_shortest = statistics.mean(df["shortest_median"])
    mean_median_gpt4o = statistics.mean(df["gpt4o_median"])
    mean_median_gpt35 = statistics.mean(df["gpt35_median"])
    plt.plot(df['iteration'], df['shortest_mean'], label=f'Baseline: Shortest Mean (Overall Mean={mean_mean_shortest:.2f})', marker='o')
    plt.plot(df['iteration'], df['gpt4o_mean'], label=f'GPT-4o Mean (Overall Mean={mean_mean_gpt4o:.2f}, Δ={mean_mean_gpt4o - mean_mean_shortest:.2f})', marker='s')
    plt.plot(df['iteration'], df['gpt35_mean'], label=f'GPT-3.5 Mean (Overall Mean={mean_mean_gpt35:.2f}, Δ={mean_mean_gpt35 - mean_mean_shortest:.2f})', marker='^')

    plt.xlabel('Iteration')
    plt.ylabel('Mean Value')
    plt.title(f'Mean Congestion Impact Values Across Iterations - {sim_name}')
    plt.legend()
    plt.grid()
    plt.savefig(f'means-{sim_name}')

    # Plot median values
    plt.figure(figsize=(10, 5))
    plt.plot(df['iteration'], df['shortest_median'], label=f'Baseline: Shortest Median (Mean={mean_median_shortest:.2f})', marker='o')
    plt.plot(df['iteration'], df['gpt4o_median'], label=f'GPT-4o Median (Mean={mean_median_gpt4o:.2f}, Δ={mean_median_gpt4o - mean_median_shortest:.2f})', marker='s')
    plt.plot(df['iteration'], df['gpt35_median'], label=f'GPT-3.5 Median (Mean={mean_median_gpt35:.2f}, Δ={mean_median_gpt35 - mean_median_shortest:.2f})', marker='^')

    plt.xlabel('Iteration')
    plt.ylabel('Median Value')
    plt.title(f'Median Congestion Impact Values Across Iterations - {sim_name}')
    plt.legend()
    plt.grid()
    plt.savefig(f'medians-{sim_name}')

if __name__=='__main__':
    run('out/sim1-results.txt', 'Large')
    run('out/sim2-results.txt', 'Circle')
    run('out/sim3-results.txt', 'Bottleneck')