
import numpy as np
from advertising_curve import *

# Base learner class for the advertising scenario
class GP_Learner_advert:
    def __init__(self, n_bids, bids, prices, probabilities, cost):
        # Constructor method. Initialize the base learner with the given parameters.
        self.n_bids = n_bids  # Number of possible bids
        self.t = 0  # Current round number
        self.click_cost_curve = advertising_curve(n_bids, bids)  # Click cost curve for each bid
        self.click_daily_curve = advertising_curve(n_bids, bids)  # Click daily curve for each bid
        self.bids_selections = np.zeros(n_bids)  # Number of times each bid has been pulled
        self.collected_rewards_bids = np.zeros(n_bids)  # Collected rewards for each bid
        self.bids = bids  # Possible bid values
        self.pulled_bids = []  # List to store pulled bids for later analysis
        self.prices = prices  # Prices for each bid
        self.probabilities = probabilities  # Probability of a click for each bid
        self.cost = cost  # Cost per click for the advertiser

    def update_observations(self, pulled_bid, click_cost, click_daily):
        # Update the click cost and click daily observations for the pulled_bid.
        # Call the corresponding update_observations method for click cost and click daily curves.
        self.click_cost_curve.update_observations(pulled_bid, click_cost)
        self.click_daily_curve.update_observations(pulled_bid, click_daily)

    def update_model(self):
        # Update the click cost and click daily curves based on the collected observations.
        # Call the corresponding update_curve method for click cost and click daily curves.
        self.click_cost_curve.update_curve()
        self.click_daily_curve.update_curve()
