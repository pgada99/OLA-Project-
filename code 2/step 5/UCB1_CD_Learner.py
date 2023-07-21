
import numpy as np

from UCB1_Learner import *

from scipy.stats import norm
from scipy.stats import t as t_student
import pingouin as pg


class UCB1_CD_Learner(UCB1_Learner):
    def __init__(self, n_arms,prices, cost,  window_size, alpha):
        super().__init__(n_arms,prices, cost )
        self.alpha = alpha  # Confidence level for change detection
        self.window_size = window_size  # Size of the sliding window for change detection
        self.rewards_history = [[] for _ in range(n_arms)]  # List to store historical rewards for each arm
        self.pulled_arms = [[] for _ in range(n_arms)]  # List to store historical rewards for each arm
        self.change_detected = [False] * n_arms  # List to track if changes have been detected for each arm

    def detect_change(self, pulled_arm):
        if len(self.rewards_history[pulled_arm]) >= 2*self.window_size:
            window_rewards = self.rewards_history[pulled_arm][-self.window_size:]
            historical_rewards = self.rewards_history[pulled_arm][:-self.window_size]


            # Conducting two-sample ttest
            result = pg.ttest(window_rewards,
                              historical_rewards,
                              correction=True)
            result['p-val'][0]
            if result['p-val'][0] < self.alpha:
                self.change_detected[pulled_arm] = True
            else:
                self.change_detected[pulled_arm] = False

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.rewards_history[pulled_arm].append(reward)
        self.pulled_arms[pulled_arm].append(1)
        for arm in range(self.n_arms):
          self.detect_change(arm)

    def pull_arm(self):
        index = find_first_zero(self.arm_selections)
        if index == -1:
            for arm in range(self.n_arms):
                if self.change_detected[arm]:

                    # Once a change is detected, reset window rewards to historical rewards
                    self.rewards_history[arm] = self.rewards_history[arm][-self.window_size:]
                    # Clear the cumulative rewards and arm selections for the arm
                    self.collected_rewards_arm[arm] = np.sum(self.rewards_history[arm][-self.window_size:])
                    self.arm_selections[arm] = self.window_size
                    self.change_detected[arm] = False

            ucb_values = self.collected_rewards_arm / self.arm_selections + np.sqrt(2*np.log(self.t) / self.arm_selections)
            idx = np.argmax(ucb_values*(self.prices-self.cost*np.ones(self.n_arms)))
        else:
            idx = index
        return idx