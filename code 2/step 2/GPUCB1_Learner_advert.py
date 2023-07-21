import numpy as np
from GP_Learner_advert import *


def find_first_zero(array):
    for i, num in enumerate(array):
        if num == 0:
            return i
    return -1  # Return -1 if zero is not found in the array


# GPUCB1 Learner class for the advertising scenario, inherits from GP_Learner_advert
class GPUCB1_Learner_advert(GP_Learner_advert):
    def __init__(self, n_bids, bids, prices, probabilities, cost):
        # Constructor method. Initialize the GPUCB1 Learner by calling the parent class constructor and setting additional attributes.
        super().__init__(n_bids, bids, prices, probabilities, cost)
        self.bid_called = np.zeros(n_bids)  # Keep track of how many times each bid has been called

    def update_observations(self, bid_idx, click_cost, click_daily):
        # Update the observations with the given bid index, click cost, and click daily count.
        # Call the parent class's update_observations method to update relevant parameters.
        super().update_observations(bid_idx, click_cost, click_daily)
        # Add the pulled bid to the list of pulled bids for later analysis.
        self.pulled_bids.append(self.bids[bid_idx])

    def update(self, pulled_arm, click_cost, click_daily):
        # Increment the current round number (t) for the GPUCB1 Learner.
        self.t += 1
        # Update the observations based on the pulled_arm, click cost, and click daily count.
        self.update_observations(pulled_arm, click_cost, click_daily)
        # Increment the counter for the bid that was pulled.
        self.bid_called[pulled_arm] += 1
        # Update the underlying model to adapt to new information.
        self.update_model()

    def pull_arm(self):
        # Compute the exploration parameter (delta) for UCB1.
        delta = np.sqrt(2 * np.log(self.t))
        # Calculate the upper confidence bound for daily clicks and click cost curves.
        upper_bound_dc = self.click_daily_curve.means + delta * self.click_daily_curve.sigmas
        upper_bound_cc = self.click_cost_curve.means + delta * self.click_cost_curve.sigmas
        # Find the index of the first bid that has not been called (exploration phase).
        index = find_first_zero(self.bid_called)
        if index == -1:
            # If all bids have been called at least once, calculate the expected reward for each bid using UCB1.
            reward_bid = np.array([
                np.max(
                    upper_bound_dc[i] * self.probabilities * (self.prices - self.cost * np.ones(len(self.prices)))
                    - upper_bound_cc[i] * np.ones(len(self.prices))
                )
                for i in range(len(self.bids))
            ])
            # Select the bid index with the highest expected reward (greedy action selection).
            idx = np.argmax(reward_bid)
        else:
            # If there are still bids that haven't been called, select the next one for exploration.
            idx = index

        return idx
