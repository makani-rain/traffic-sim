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
    plt.plot(df['iteration'], df['shortest_mean'], label=f'Shortest Mean (Overall Mean={statistics.mean(df["shortest_mean"]):.2f})', marker='o')
    plt.plot(df['iteration'], df['gpt4o_mean'], label=f'GPT-4o Mean (Overall Mean={statistics.mean(df["gpt4o_mean"]):.2f})', marker='s')
    plt.plot(df['iteration'], df['gpt35_mean'], label=f'GPT-3.5 Mean (Overall Mean={statistics.mean(df["gpt35_mean"]):.2f})', marker='^')

    plt.xlabel('Iteration')
    plt.ylabel('Mean Value')
    plt.title(f'Mean Values Across Iterations - {sim_name}')
    plt.legend()
    plt.grid()
    plt.savefig(f'means-{sim_name}')

    # Plot median values
    plt.figure(figsize=(10, 5))
    plt.plot(df['iteration'], df['shortest_median'], label=f'Shortest Median (Mean={statistics.mean(df["shortest_median"]):.2f})', marker='o')
    plt.plot(df['iteration'], df['gpt4o_median'], label=f'GPT-4o Median (Mean={statistics.mean(df["gpt4o_median"]):.2f})', marker='s')
    plt.plot(df['iteration'], df['gpt35_median'], label=f'GPT-3.5 Median (Mean={statistics.mean(df["gpt35_median"]):.2f})', marker='^')

    plt.xlabel('Iteration')
    plt.ylabel('Median Value')
    plt.title(f'Median Values Across Iterations - {sim_name}')
    plt.legend()
    plt.grid()
    plt.savefig(f'medians-{sim_name}')

if __name__=='__main__':
    run('out/sim1-results.txt', 'sim1')
    run('out/sim2-results.txt', 'sim2')
    run('out/sim3-results.txt', 'sim3')