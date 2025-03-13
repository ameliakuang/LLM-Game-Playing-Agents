import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style
sns.set(style="whitegrid")

# Read the CSV files
space_invaders_data = pd.read_csv('space_invaders_best_performance.csv')
pong_data = pd.read_csv('pong_best_performance.csv')

# Instead of adding 1 to the data, we'll set the ticks explicitly
# Create separate plots for each game
# Plot Space Invaders data
plt.figure(figsize=(8, 6))
plt.plot(space_invaders_data['Optimization Step'], space_invaders_data['Mean Reward'], 
         marker='o', linestyle='-', linewidth=2, markersize=8, color='#FF5733')
plt.fill_between(space_invaders_data['Optimization Step'],
                 space_invaders_data['Mean Reward'] - space_invaders_data['Std Dev Reward'],
                 space_invaders_data['Mean Reward'] + space_invaders_data['Std Dev Reward'],
                 alpha=0.2, color='#FF5733')
plt.title('Space Invaders Performance', fontsize=16)
plt.xlabel('Optimization Step', fontsize=14)
plt.ylabel('Mean Reward', fontsize=14)
# Set x-ticks explicitly starting at 1
x_ticks = np.arange(len(space_invaders_data)) + 1
plt.xticks(space_invaders_data['Optimization Step'], x_ticks)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('space_invaders_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot Pong data
plt.figure(figsize=(8, 6))
plt.plot(pong_data['Optimization Step'], pong_data['Mean Reward'], 
         marker='o', linestyle='-', linewidth=2, markersize=8, color='#3498DB')
plt.fill_between(pong_data['Optimization Step'],
                pong_data['Mean Reward'] - pong_data['Std Dev Reward'],
                pong_data['Mean Reward'] + pong_data['Std Dev Reward'],
                alpha=0.2, color='#3498DB')
plt.title('Pong Performance', fontsize=16)
plt.xlabel('Optimization Step', fontsize=14)
plt.ylabel('Mean Reward', fontsize=14)
# Set x-ticks explicitly starting at 1
x_ticks = np.arange(len(pong_data)) + 1
plt.xticks(pong_data['Optimization Step'], x_ticks)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pong_performance.png', dpi=300, bbox_inches='tight')
plt.close() 