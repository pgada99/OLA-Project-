import numpy as np
from GP_Learner_advert import *


class GPTS_Learner_advert(GP_Learner_advert):
    
    def __init__(self, n_bids, bids, prices, probabilities, cost):
        # Constructor method. Initialize the object by calling the constructor of the parent class.
        super().__init__(n_bids, bids, prices, probabilities, cost)

    def update_observations(self, bid_idx, cost_click, daily_click):
        # Update the observations with the given bid index, cost per click, and daily click count.
        # Call the parent class's update_observations method to update relevant parameters.
        super().update_observations(bid_idx, cost_click, daily_click)
        # Add the pulled bid to the list of pulled bids for later analysis.
        self.pulled_bids.append(self.bids[bid_idx])

    def update(self, pulled_arm, cost_click, daily_click):
        # Increment the current round number (t) for the agent.
        self.t += 1
        # Update the observations based on the pulled_arm, cost per click, and daily click count.
        self.update_observations(pulled_arm, cost_click, daily_click)
        # Update the underlying model to adapt to new information.
        self.update_model()

    def pull_arm(self):
        # Sample values from normal distributions based on click cost and click daily curves.
        sampled_values_cc = np.random.normal(self.click_cost_curve.means, self.click_cost_curve.sigmas)
        
        sampled_values_dc = np.random.normal(self.click_daily_curve.means, self.click_daily_curve.sigmas)
        i=0
        
        # Calculate the expected reward for each bid based on sampled values.
        reward_bid = np.array([
            np.max(
                sampled_values_dc[i] * self.probabilities * (self.prices - self.cost * np.ones(len(self.prices)))
                - sampled_values_cc[i] * np.ones(len(self.prices))
            )
            for i in range(len(self.bids))
        ])
        # Return the index of the bid with the highest expected reward (greedy action selection).
        return np.argmax(reward_bid)
