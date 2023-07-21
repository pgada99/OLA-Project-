
from Environment import *
import numpy as np
import math

class Non_Stationary_Environment(Environment):

  def __init__(self, n_arms, probabilities, prices, bids, cost_click,  daily_click, cost, horizon):
    # Call the constructor of the parent class Environment
    super().__init__(n_arms, probabilities, prices, bids, cost_click,  daily_click, cost)
    self.t = 0  # Current time step
    n_phases = len(self.probabilities)  # Number of phases in the non-stationary environment
    self.phases_size = math.ceil(horizon/n_phases)  # Size of each phase in terms of time steps

  def optimal_bid(self, current_phase, price_indx):
    optimal_bid_idx = np.argmax(self.probabilities[current_phase][price_indx] * self.daily_click *
                                (self.prices[price_indx] - self.cost) - self.cost_click)
    return  optimal_bid_idx

  def round(self, pulled_arm):
    # Determine the current phase based on the time step
    current_phase = math.floor(self.t / self.phases_size)

    optimal_bid_idx = self.optimal_bid( current_phase, pulled_arm)
    click_day = np.round(self.daily_click[optimal_bid_idx]+np.random.normal(0,1))
    result = np.random.binomial(1, self.probabilities[current_phase][pulled_arm],int(click_day))
    conversion_rate = result.sum()/click_day
    reward = np.max(conversion_rate*click_day*(self.prices[pulled_arm]-self.cost)-(self.cost_click[optimal_bid_idx]+np.random.normal(0,1)))
    self.t += 1  # Increment the time step
    return reward, conversion_rate
  


