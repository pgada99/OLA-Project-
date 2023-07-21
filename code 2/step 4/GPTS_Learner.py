import numpy as np
from GP_Learner import *

n_arms = 5
prices= [350, 400, 450, 500, 550]


# GPTS Learner class, inherits from GP_Learner
class GPTS_Learner(GP_Learner):
    def __init__(self, n_bids, bids, n_arms, cost, prices):
        # Constructor method. Initialize the GPTS Learner by calling the parent class constructor and setting additional attributes.
        super().__init__(n_bids, bids, n_arms, cost, prices)
        self.beta_parameters = np.ones((n_arms, 2))  # Beta parameters for the Beta distribution

    def update_observations(self, bid_idx, cost_click, daily_click, pulled_arm, conversion_rate_obs):
        # Update the observations with the given bid index, cost per click, daily clicks, pulled arm, and conversion rate observation.
        # Call the parent class's update_observations method to update relevant parameters.
        super().update_observations(bid_idx, cost_click, daily_click, pulled_arm, conversion_rate_obs)

    def update(self, pulled_bid, cost_click, daily_click, pulled_arm, conversion_rate_obs):
        # Increment the current round number (t) for the GPTS Learner.
        self.t += 1
        # Update the observations based on the pulled_bid, cost per click, daily clicks, pulled arm, and conversion rate observation.
        self.update_observations(pulled_bid, cost_click, daily_click, pulled_arm, conversion_rate_obs)
        # Update the Beta parameters using the observed conversion rate and daily clicks for the pulled arm.
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + conversion_rate_obs * daily_click
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (1 - conversion_rate_obs) * daily_click

    def pull_arm(self):
        # Sample conversion rates from the Beta distribution with the updated Beta parameters.
        conversion_rate = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        # Sample values from normal distributions for click cost and click daily curves.
        sampled_values_cc = np.random.normal(self.click_cost_curve.means, self.click_cost_curve.sigmas)
        sampled_values_dc = np.random.normal(self.click_daily_curve.means, self.click_daily_curve.sigmas)
        # Calculate the expected reward for each arm using the sampled conversion rates and sampled values.
        reward_arm = np.array([
            np.max(
                sampled_values_dc * conversion_rate[i] * (self.prices[i] - self.cost) - sampled_values_cc
            )
            for i in range(n_arms)
        ])
        # Select the arm index with the highest expected reward (greedy action selection).
        idx = np.argmax(reward_arm)
        return idx
