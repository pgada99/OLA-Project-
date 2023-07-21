import numpy as np
from Learner import *

def find_first_zero(array):
    for i, num in enumerate(array):
        if num == 0:
            return i
    return -1  # Return -1 if zero is not found in the array

# UCB1 learner
class UCB1_Learner(Learner):
  def __init__(self, n_arms,  prices, cost):
    super().__init__(n_arms, prices, cost)

  def pull_arm(self):
    index = find_first_zero(self.arm_selections)  # Find the first zero in the arm selections
    if index == -1:  # If no zeros found, use UCB1 strategy
      t = np.sum(self.arm_selections) + 1  # Total number of arm selections
      ucb_values = (self.collected_rewards_arm *(self.prices-self.cost*np.ones(self.n_arms))/ self.arm_selections + np.sqrt(2*np.log(t) / self.arm_selections))  # Calculate UCB values
      idx = np.argmax(ucb_values)  # Select arm with highest UCB value
    else:
      idx = index  # Select the first unexplored arm
    return idx

  def update(self, pulled_arm, conversion_rate):
    self.t += 1
    self.update_observations(pulled_arm, conversion_rate)