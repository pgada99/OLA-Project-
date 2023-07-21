
# Importing Libraries ------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style = 'darkgrid')

# We import our plot function
from plot_function import *

# We import the other functionalities
from TS_Learner import *
from UCB1_Learner import *
from Environment import *



# INITIAL INFORMATION  ------------------------------------------------------


# Converstion probabilities

p = [
    [0.4, 0.38, 0.33, 0.25, 0.15],
    [0.22, 0.25, 0.30, 0.28, 0.21],
    [0.25, 0.22, 0.24, 0.3, 0.35]
]

#We only select C1
p_arms = p[0]


def daily_click_fun(x):
  x = x/100
  return 100 * (1.0 - np.exp(-4 * x + 2 * x ** 3))

def cost_click_fun(x):
  x = x/100
  return 400 * (1.0 - np.exp(-5 * x + 2 * x ** 3))


# Number of arms and bids
n_arms = 5
n_bids = 100

# Generate probabilities for each arm
prices= [350, 400, 450, 500, 550]
cost = 210

# Number of rounds
T = 365

# Number of experiments
n_experiment = 100





# EXERCISE 1 EXPERIMENT ------------------------------------------------------

# Lists to store rewards for each experiment
ts_rewards_per_experiment = []
ucb1_rewards_per_experiment = []

# Generate bids, cost_click, and daily_click
bids = np.linspace(0, 100, n_bids)
cost_click = cost_click_fun(bids)
daily_click = daily_click_fun(bids)

# Perform experiments
for e in range(n_experiment):
    # Create environment and learners
    env = Environment(n_arms, p_arms, prices,bids, cost_click, daily_click, cost)
    ts_learner = TS_Learner(n_arms, prices, cost)
    ucb1_learner = UCB1_Learner(n_arms, prices, cost)

    # Arrays to store collected rewards for each time step
    ts_collected_reward = np.zeros(T)
    ucb1_collected_reward = np.zeros(T)

    # Iterate over time steps
    for t in range(T):
        # TS learner
        pulled_arm = ts_learner.pull_arm()
        reward, conversion_rate, click_rate = env.round(pulled_arm)
        ts_learner.update(pulled_arm, conversion_rate, click_rate)
        ts_collected_reward[t] = reward

        # UCB1 learner
        pulled_arm = ucb1_learner.pull_arm()
        reward,  conversion_rate, click_rate= env.round(pulled_arm)
        ucb1_learner.update(pulled_arm,  conversion_rate)
        ucb1_collected_reward[t] = reward

    # Append the collected rewards for each experiment
    ts_rewards_per_experiment.append(ts_collected_reward)
    ucb1_rewards_per_experiment.append(ucb1_collected_reward)




# ANALYZE THE RESULTS  ------------------------------------------------------


# Identify the optimal result
opt = np.max([daily_click*p_arms[i]*(prices[i]-cost)-cost_click for i in range(n_arms)] )
print(opt)
optimal = opt * np.ones(T)
print(optimal)
matrix_val = np.array([ts_rewards_per_experiment, ucb1_rewards_per_experiment])
label  = ['TS','UCB1']
title = 'Exercise 1 Result'
plot_function(matrix_val, optimal, label, title)
