import numpy as np

from price_curve import *
from advertising_curve import *

#Base learner class
class GP_Learner:
  def __init__(self, n_bids, bids, n_arms, cost, prices):
    self.n_bids = n_bids
    self.n_arms = n_arms
    self.t = 0
    self.click_cost_curve = advertising_curve(n_bids, bids)
    self.click_daily_curve = advertising_curve(n_bids, bids)
    self.bids_selections = np.zeros(n_bids)  # Number of times each bid has been pulled
    self.collected_rewards_bids = np.zeros(n_bids)  # Collected rewards for each bid
    self.bids = bids
    self.price_curve = price_curve(n_arms)
    self.cost = cost
    self.prices = prices

  def update_observations(self, pulled_bid, click_cost, click_daily, pulled_arm, conversion_rate_obs):

    self.click_cost_curve.update_observations( pulled_bid, click_cost)
    self.click_daily_curve.update_observations( pulled_bid, click_daily)
    self.price_curve.update_observations(pulled_arm, conversion_rate_obs)
    self.update_model()

  def update_model(self):
    self.click_cost_curve.update_curve()
    self.click_daily_curve.update_curve()