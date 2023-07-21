
import numpy as np

# Defining the environment
class Environment():

  def __init__(self, n_arms, probabilities, prices, bids, cost_click,  daily_click, cost):
    self.n_arms = n_arms
    self.probabilities = probabilities
    self.cost_click = cost_click
    self.daily_click = daily_click
    self.prices = prices
    self.bids = bids
    self.cost = cost

  def optimal_bid(self, price_indx):
    optimal_bid_idx = np.argmax(self.probabilities[price_indx] * self.daily_click *
                                (self.prices[price_indx] - self.cost) - self.cost_click)
    return  optimal_bid_idx

  def round(self, pulled_arm):
    optimal_bid_idx = self.optimal_bid( pulled_arm)
    click_day = np.round(self.daily_click[optimal_bid_idx]+np.random.normal(0,1))

    result = np.random.binomial(1, self.probabilities[pulled_arm],int(click_day))

    conversion_rate = result.sum()/click_day
    reward = np.max(conversion_rate*click_day*(self.prices[pulled_arm]-self.cost)-(self.cost_click[optimal_bid_idx]+np.random.normal(0,1)))
    return reward, conversion_rate, click_day