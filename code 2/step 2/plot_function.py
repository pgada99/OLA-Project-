
import matplotlib.pyplot as plt
import numpy as np



def plot_function(data, optimal, label, title):
  color = ['r', 'b', 'g', 'y']
  # Create a figure and subplots
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))
  plt.suptitle(title)
  # Plot data on each subplot
  axs[0, 0].set_title("Instantaneous Rewards")
  # Instantaneous Rewards subplot
  for i in range(len(data[:,0])):
    mean = np.mean(data[i,:], axis=0)
    std = np.std(data[i,:], axis=0)
    axs[0, 0].plot(mean,  label=label[i],color=color[i])  # Plot mean of TS rewards in red
    axs[0, 0].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2,
                       color=color[i])  # Fill the region between mean - std and mean + std for UCB1

  axs[0, 0].plot(optimal, 'k--', label='Optimal')  # Plot the optimal reward (constant) in black dashed line
  axs[0, 0].legend()  # Add legend
  axs[0, 0].set_ylabel('Reward')  # Set y-axis label
  axs[0, 0].set_xlabel('t')  # Set x-axis label

  # Cumulative Rewards subplot
  axs[0, 1].set_title("Cumulative Rewards")
  for i in range(len(data[:,0])):
    mean = np.mean(data[i,:], axis=0)
    std = np.std(data[i,:], axis=0)
    axs[0, 1].plot(np.cumsum(mean),  label=label[i],color=color[i])  # Plot mean of TS rewards in red
    axs[0, 1].fill_between(range(len(mean)), np.cumsum(mean - std), np.cumsum(mean + std), alpha=0.2, color=color[i] ) # Fill the region between mean - std and mean + std for UCB1

  axs[0, 1].plot(np.cumsum(optimal), 'k--', label='Optimal')  # Plot cumulative optimal reward in black dashed line
  axs[0, 1].legend()  # Add legend
  axs[0, 1].set_ylabel('Reward')  # Set y-axis label
  axs[0, 1].set_xlabel('t')  # Set x-axis label

  # Instantaneous Regret subplot
  axs[1, 0].set_title("Instantaneous Regret")
  for i in range(len(data[:,0])):
    mean = np.mean(optimal-data[i,:], axis=0)
    std = np.std(optimal-data[i,:], axis=0)
    axs[1, 0].plot(mean,  label=label[i],color=color[i])  # Plot mean of TS rewards in red
    axs[1, 0].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2,
                       color=color[i])  # Fill the region between mean - std and mean + std for UCB1

  axs[1, 0].legend()  # Add legend
  axs[1, 0].set_ylabel('Regret')  # Set y-axis label
  axs[1, 0].set_xlabel('t')  # Set x-axis label


  # Cumulative Rewards subplot:
  axs[0, 1].set_title("Cumulative Rewards")
  for i in range(len(data[:,0])):
    mean = np.mean(optimal-data[i,:], axis=0)
    std = np.std(optimal-data[i,:], axis=0)
    axs[1, 1].plot(np.cumsum(mean),  label=label[i],color=color[i])  # Plot mean of TS rewards in red
    axs[1, 1].fill_between(range(len(mean)), np.cumsum(mean - std), np.cumsum(mean + std), alpha=0.2, color=color[i] ) # Fill the region between mean - std and mean + std for UCB1

  axs[1, 1].legend()  # Add legend
  axs[1, 1].set_ylabel('Regret')  # Set y-axis label
  axs[1, 1].set_xlabel('t')  # Set x-axis label
  # Set shared labels for y-axis
  fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical')
  # Adjust spacing between subplots
  plt.subplots_adjust(wspace=0.3, hspace=0.2)
  # Display the figure
  plt.show()
