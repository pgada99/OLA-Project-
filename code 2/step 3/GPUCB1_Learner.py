import numpy as np

from GP_Learner import *

def find_first_zero(array):
    for i, num in enumerate(array):
        if num == 0:
            return i
    return -1  # Return -1 if zero is not found in the array


n_arms = 5
prices= [350, 400, 450, 500, 550]


# GPUCB1 Learner class, inherits from GP_Learner
class GPUCB1_Learner(GP_Learner):
    def __init__(self, n_bids, bids, n_arms, cost, prices):
        # Constructor method. Initialize the GPUCB1 Learner by calling the parent class constructor and setting additional attributes.
        super().__init__(n_bids, bids, n_arms, cost, prices)
        self.cumulative_rewards = np.zeros(n_arms)  # Cumulative rewards for each arm

    def update_observations(self, bid_idx, cost_click, daily_click, pulled_arm, conversion_rate_obs):
        # Update the observations with the given bid index, cost per click, daily clicks, pulled arm, and conversion rate observation.
        # Call the parent class's update_observations method to update relevant parameters.
        super().update_observations(bid_idx, cost_click, daily_click, pulled_arm, conversion_rate_obs)

    def update(self, pulled_bid, cost_click, daily_click, pulled_arm, conversion_rate_obs):
        # Increment the current round number (t) for the GPUCB1 Learner.
        self.t += 1
        # Accumulate the observed conversion rate for the pulled arm in the cumulative rewards.
        self.cumulative_rewards[pulled_arm] += conversion_rate_obs
        # Update the observations based on the pulled_bid, cost per click, daily clicks, pulled arm, and conversion rate observation.
        self.update_observations(pulled_bid, cost_click, daily_click, pulled_arm, conversion_rate_obs)
        # Update the underlying model to adapt to new information.
        self.update_model()

    def pull_arm(self):
        # Find the index of the first arm that has not been pulled yet (exploration phase).
        index = find_first_zero(self.price_curve.arm_selections)
        if index == -1:
            # If all arms have been pulled at least once, calculate the UCB1 values for each arm.
            t = np.sum(self.price_curve.arm_selections) + 1  # Total number of arm selections so far
            ucb_values = self.cumulative_rewards / self.price_curve.arm_selections + np.sqrt(2 * np.log(t) / self.price_curve.arm_selections)
            # Sample values from normal distributions for click cost and click daily curves.
            sampled_values_cc = np.random.normal(self.click_cost_curve.means, self.click_cost_curve.sigmas)
            sampled_values_dc = np.random.normal(self.click_daily_curve.means, self.click_daily_curve.sigmas)
            # Calculate the expected reward for each arm using UCB1 values and sampled values.
            reward_arm = np.array([
                np.max(
                    sampled_values_dc * ucb_values[i] * (self.prices[i] - self.cost) - sampled_values_cc
                )
                for i in range(n_arms)
            ])
            # Select the arm index with the highest expected reward (greedy action selection).
            idx = np.argmax(reward_arm)
        else:
            # If there are still arms that haven't been pulled, select the next one for exploration.
            idx = index

        return idx
