

from UCB1_Learner import *
import numpy as np

class SWUCB1_Learner(UCB1_Learner):
  def __init__(self, n_arms, prices, cost, window_size):
    # Call the constructor of the parent class UCB1_Learner
    super().__init__(n_arms, prices, cost)
    self.window_size = window_size  # Size of the sliding window
    self.pulled_arms = []  # List to store the pulled arms in the sliding window
    self.arm_selections_windows = np.zeros(n_arms)  # Arm selections count for each arm in the sliding window

  def update(self, pulled_arm, conversion_rate):
    self.t += 1  # Increment the time step
    self.update_observations(pulled_arm, conversion_rate)  # Update the observations for the pulled arm
    self.arm_selections_windows[pulled_arm] += 1  # Increment the arm selections count for the pulled arm in the window
    self.pulled_arms.append(pulled_arm)
    if self.t > self.window_size:
      # Remove the oldest arm from the sliding window and update its counts and cumulative rewards
      index = int(self.pulled_arms[-self.window_size])
      self.arm_selections_windows[index] -= 1
      self.collected_rewards_arm[index] -= self.collected_rewards[-self.window_size]

  def pull_arm(self):
    # Find the first unexplored arm in the sliding window (arm_selections_windows = 0)
    index = find_first_zero(self.arm_selections_windows)
    if index == -1:  # If all arms in the sliding window are explored, use UCB1 strategy
      # Calculate the UCB values for each arm in the sliding window
      ucb_values = self.collected_rewards_arm*(self.prices-self.cost*np.ones(self.n_arms)) / self.arm_selections_windows + np.sqrt(2 * np.log(self.t) / self.arm_selections_windows)
      idx = np.argmax(ucb_values)  # Select the arm with the highest UCB value
    else:
      idx = index  # Select the first unexplored arm in the sliding window
    return idx