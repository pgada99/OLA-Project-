import numpy as np


def daily_click_fun(x):
  x = x/100
  return 100 * (1.0 - np.exp(-4 * x + 2 * x ** 3))

def cost_click_fun(x):
  x = x/100
  return 400 * (1.0 - np.exp(-5 * x + 2 * x ** 3))

prices= [350, 400, 450, 500, 550]

## Bidding Environment class
class Environment_Bids():
    def __init__(self, bids, n_arms, prices, probabilities, cost):
        # Constructor method. Initialize the bidding environment with the given parameters.
        self.bids = bids  # Possible bid values
        self.n_arms = n_arms  # Number of arms (bids)
        self.probabilities = probabilities  # Probability of a click for each bid
        self.cost_click = cost_click_fun(bids)  # Mean cost per click based on bids
        self.daily_click = daily_click_fun(bids)  # Mean daily clicks based on bids
        self.prices = prices  # Prices for each bid
        self.cost = cost  # Cost per click for the advertiser

    def optimal_bid(self, price_indx):
        # Calculate the optimal bid index for a given price index (price_indx).
        # Find the bid index that maximizes the expected reward (clicks * (price - cost) - cost_click) for the given price index.
        optimal_bid_idx = np.argmax(
            self.probabilities[price_indx] * self.daily_click * (self.prices[price_indx] - self.cost) - self.cost_click
        )
        return optimal_bid_idx

    def round(self, pulled_arm):
        # Simulate a single bidding round with the pulled_arm (the bid chosen by the agent).
        # Determine the optimal bid index based on the pulled arm (bid index) and its corresponding price index.
        optimal_bid_idx = self.optimal_bid(pulled_arm)
        # Generate random observations for click_day and cost_click based on the optimal bid index.
        click_day = np.round(self.daily_click[optimal_bid_idx] + np.random.normal(0, 1))
        cost_click_obs = self.cost_click[optimal_bid_idx] + np.random.normal(0, 1)
        # Simulate click events based on the pulled arm's probability and click_day.
        result = np.random.binomial(1, self.probabilities[pulled_arm], int(click_day))
        # Calculate the conversion rate (clicks / click_day) based on the simulated click events.
        conversion_rate = result.sum() / click_day
        # Calculate the reward based on the conversion rate, click_day, price, cost, and cost_click observation.
        reward = np.max(conversion_rate * click_day * (self.prices[pulled_arm] - self.cost) - cost_click_obs)
        # Return the optimal bid index, cost_click observation, click_day, conversion rate, and reward for the round.
        return optimal_bid_idx, cost_click_obs, click_day, conversion_rate, reward
