import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("training_log.csv")  # Replace with your actual file path

# Calculate cumulative reward for each epoch
epoch_rewards = df.groupby('Epoch')['Reward'].sum().reset_index()

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(epoch_rewards['Epoch'], epoch_rewards['Reward'], 
         marker='o', linestyle='-', linewidth=2, markersize=8,
         color='#2ca02c', alpha=0.8)

# Formatting
plt.title('Cummulative Reward Progression', fontsize=24, pad=20)
plt.xlabel('Epoch Number', fontsize=24)
plt.ylabel('Cumulative Reward', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)

# X-axis configuration
plt.xticks(range(1, 51))  # Show all 50 epochs
plt.xlim(0.5, 50.5)  # Add padding

plt.tight_layout()
plt.savefig('epoch_cummulative_reward.png', dpi=300, bbox_inches='tight')
plt.show()