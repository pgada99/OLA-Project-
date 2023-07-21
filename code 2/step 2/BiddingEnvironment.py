import numpy as np

def daily_click_fun(x):
  x = x/100
  return 100 * (1.0 - np.exp(-4 * x + 2 * x ** 3))

def cost_click_fun(x):
  x = x/100
  return 400 * (1.0 - np.exp(-5 * x + 2 * x ** 3))

class BiddingEnvironment():
    def __init__(self, bids, prices, probabilities, cost):
        # Constructor method. Initialize the environment with the given bids, prices, probabilities, and cost.
        self.bids = bids
        # Calculate mean cost per click based on bids using the cost_click_fun function.
        self.click_cost = cost_click_fun(bids)
        # Calculate mean daily clicks based on bids using the daily_click_fun function.
        self.daily_clicks = daily_click_fun(bids)
        self.prices = prices
        self.cost = cost
        self.probabilities = probabilities

    def round(self, pulled_bid):
        # Simulate a single round of the bidding environment with the given pulled_bid.
        # Generate a random observation for cost per click and daily clicks based on pulled_bid.
        cost_click_obs = self.click_cost[pulled_bid] + np.random.normal(0, 1)
        daily_click_obs = self.daily_clicks[pulled_bid] + np.random.normal(0, 1)
        # Calculate the reward for each bid based on the random observations.
        reward = np.max(
            self.probabilities * daily_click_obs * (self.prices - self.cost * np.ones(len(self.prices)))
            - cost_click_obs * np.ones(len(self.prices))
        )
        # Return the reward, cost per click observation, and daily click observation.
        return reward, cost_click_obs, daily_click_obs
