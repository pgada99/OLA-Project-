import math
import numpy as np
from Non_Stationary_Environment import*
#Defining the environment
class Non_Stationary_Environment_hf(Non_Stationary_Environment):

  def __init__(self, n_arms, probabilities, prices, bids, cost_click,  daily_click, cost, horizon, phases):
    # Call the constructor of the parent class Environment
    super().__init__(n_arms, probabilities, prices, bids, cost_click,  daily_click, cost, horizon)
    self.phases = phases
    self.phases_size = np.ceil(horizon/len(phases))
  def round(self, pulled_arm):
    # Determine the current phase based on the time step
    current_phase = self.phases[math.floor(self.t / self.phases_size)]
    
    optimal_bid_idx = self.optimal_bid( current_phase, pulled_arm)
    click_day = np.round(self.daily_click[optimal_bid_idx]+np.random.normal(0,1))
    result = np.random.binomial(1, self.probabilities[current_phase][pulled_arm],int(click_day))
    conversion_rate = result.sum()/click_day
    reward = np.max(conversion_rate*click_day*(self.prices[pulled_arm]-self.cost)-(self.cost_click[optimal_bid_idx]+np.random.normal(0,1)))
    self.t += 1 
    return reward, conversion_rate