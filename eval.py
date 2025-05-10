import matplotlib.pyplot as plt
import pandas as pd
import json

with open('out/sim1-results.txt', 'r') as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(data)

# Plot mean values
plt.figure(figsize=(10, 5))
plt.plot(df['iteration'], df['shortest_mean'], label='Shortest Mean', marker='o')
plt.plot(df['iteration'], df['gpt4o_mean'], label='GPT-4o Mean', marker='s')
plt.plot(df['iteration'], df['gpt35_mean'], label='GPT-3.5 Mean', marker='^')

plt.xlabel('Iteration')
plt.ylabel('Mean Value')
plt.title('Mean Values Across Iterations')
plt.legend()
plt.grid()
plt.show()

# Plot median values
plt.figure(figsize=(10, 5))
plt.plot(df['iteration'], df['shortest_median'], label='Shortest Median', marker='o')
plt.plot(df['iteration'], df['gpt4o_median'], label='GPT-4o Median', marker='s')
plt.plot(df['iteration'], df['gpt35_median'], label='GPT-3.5 Median', marker='^')

plt.xlabel('Iteration')
plt.ylabel('Median Value')
plt.title('Median Values Across Iterations')
plt.legend()
plt.grid()
plt.show()