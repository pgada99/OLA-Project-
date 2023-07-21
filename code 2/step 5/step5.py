
# Importing Libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from UCB1_Learner import *
from UCB1_CD_Learner import *
from SWUCB1_Learner import *
from Non_Stationary_Environment import *
#from Environment import *
from plot_function import *



sns.set(style = 'darkgrid')




# INITIAL INFORMATION  ------------------------------------------------------


# Converstion probabilities

p = [
    [0.4, 0.38, 0.33, 0.25, 0.15],
    [0.22, 0.25, 0.30, 0.28, 0.21],
    [0.25, 0.22, 0.24, 0.3, 0.35]
]

#We only select C1
p_arms = np.array(p[0])


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
T = 366

# Number of experiments
n_experiments = 200


min_bid = 0
max_bid = 100
n_bids = 100
bids = np.linspace(min_bid, max_bid, n_bids)


# EXERCISE 5 EXPERIMENT ------------------------------------------------------

sigma_cc = 1
sigma_dc = 1

n_phases = 3


# Generate bids, cost_click, and daily_click
bids = np.linspace(0, 100, n_bids)
cost_click = cost_click_fun(bids)
daily_click = daily_click_fun(bids)

phases_len = int(math.ceil(T/n_phases))

ucb1_rewards_per_experiment = []
swucb1_rewards_per_experiment = []
cd_ucb1_rewards_per_experiment = []
window_size_sw = 30
window_size_cd = 30
alpha = 0.20


for e in range(n_experiments):
  print('experiment', e)
  ucb1_learner = UCB1_Learner(n_arms=n_arms, prices= prices, cost = cost)
  nonstatio_env = Non_Stationary_Environment(n_arms, p, prices, bids, cost_click, daily_click, cost, T)
  nonstatio_env_2 = Non_Stationary_Environment(n_arms, p, prices, bids, cost_click, daily_click, cost, T)
  nonstatio_env_3 = Non_Stationary_Environment(n_arms, p, prices, bids, cost_click, daily_click, cost, T)
  swucb1_learner = SWUCB1_Learner(n_arms=n_arms, prices= prices, cost = cost, window_size=window_size_sw)
  cd_ucb1_learner = UCB1_CD_Learner(n_arms=n_arms, prices= prices, cost = cost, window_size=window_size_cd, alpha=alpha)
# Arrays to store collected rewards for each time step
  swucb1_collected_reward = np.zeros(T)
  ucb1_collected_reward = np.zeros(T)
  cd_ucb1_collected_reward = np.zeros(T)

  # Iterate over time steps
  for t in range(T):

      # UCB1 learner

      pulled_arm = ucb1_learner.pull_arm()
      reward, conversion_rate = nonstatio_env.round(pulled_arm )
      ucb1_learner.update(pulled_arm, conversion_rate)
      ucb1_collected_reward[t] = reward

      # SW-UCB1 learner

      pulled_arm = swucb1_learner.pull_arm()
      reward, conversion_rate = nonstatio_env_2.round(pulled_arm)
      swucb1_learner.update(pulled_arm, conversion_rate)
      swucb1_collected_reward[t] = reward

      # UCB1 learner

      pulled_arm = cd_ucb1_learner.pull_arm()
      reward, conversion_rate = nonstatio_env_3.round(pulled_arm)
      cd_ucb1_learner.update(pulled_arm, conversion_rate)
      cd_ucb1_collected_reward[t] = reward


  # Append the collected rewards for each experiment
  swucb1_rewards_per_experiment.append(swucb1_collected_reward)
  ucb1_rewards_per_experiment.append(ucb1_collected_reward)
  cd_ucb1_rewards_per_experiment.append(cd_ucb1_collected_reward)


opt_per_phases = np.zeros(n_phases)
for phase in range(n_phases):

  opt_per_phases[phase] =  np.max([p[phase][i]*(prices[i]-cost)*daily_click-cost_click for i in range(n_arms)])
np.array(p).max(axis=1)
optimum_per_round = np.zeros(T)
for i in range(n_phases):
  t_index = range(i*phases_len, (i+1)*phases_len)
  optimum_per_round[t_index] =opt_per_phases[i]

matrix_val = np.array([ucb1_rewards_per_experiment, swucb1_rewards_per_experiment, cd_ucb1_rewards_per_experiment])
label  = ['UCB1', 'SW UCB1', 'CD UCB1']

plot_function(matrix_val, optimum_per_round,label, 'Exercise 5 result')