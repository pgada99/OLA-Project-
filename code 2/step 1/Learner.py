
import numpy as np

class Learner:
  def __init__(self, n_arms, prices, cost):
    self.n_arms = n_arms
    self.t = 0
    self.rewards_per_arm = [[] for i in range(n_arms)]
    self.collected_rewards = np.array([])
    self.arm_selections = np.zeros(n_arms)  # Number of times each arm has been pulled
    self.collected_rewards_arm = np.zeros(n_arms)
    self.cost = cost
    self.prices = prices
    #self.collected_results = np.array([])

  def update_observations(self, pulled_arm, reward):
    self.rewards_per_arm[pulled_arm].append(reward)
    self.collected_rewards = np.append(self.collected_rewards, reward)
    self.collected_rewards_arm[pulled_arm] += reward
    self.arm_selections[pulled_arm] += 1
    #self.collected_results_arm[pulled_arm] += result
    #self.collected_results = np.append(self.collected_results, result)