
# Importing Libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from advertising_curve import *
from GP_Learner import *
from GPUCB1_Learner import *
from GPTS_Learner import *
from price_curve import *
from Environment_Bids_class import *
from advertising_curve import *
from plot_function import *



sns.set(style = 'darkgrid')




# INITIAL INFORMATION  ------------------------------------------------------


# Converstion probabilities

def daily_click_fun_class(x, class_id):
  x = x/100
  if class_id == 1:
    return 100 * (1.0 - np.exp(-4 * x + 2 * x ** 3))
  elif class_id == 2:
    return 80 * (1.0 - np.exp(-3 * x + 2 * x ** 3))
  elif class_id == 3:
    return 50 * (1.0 - np.exp(-4 * x + 2 * x ** 3))
  else:
    raise Exception("class id in input not valid")

def cost_click_fun_class(x, class_id):
  x = x/100
  if class_id == 1:
    return 400 * (1.0 - np.exp(-5 * x + 2 * x ** 3))
  elif class_id == 2:
    return 400 * (1.0 - np.exp(-3 * x + 2 * x ** 3))
  elif class_id == 3:
    return 400 * (1.0 - np.exp(-4 * x + 2 * x ** 3))
  
# Number of arms and bids
n_arms = 5

prices= [[350, 400, 450, 500, 550], [300, 350, 400, 450, 500], [325, 400, 420, 480, 550]]
p_arms = np.array([[0.5,0.4,0.3,0.2,0.16], [0.4,0.3,0.25,0.2,0.16], [0.3,0.25,0.23,0.22,0.11]])
cost = 210


min_bid = 0
max_bid = 100
n_bids = 100
bids = np.linspace(min_bid, max_bid, n_bids)
sigma_cc = 1
sigma_dc = 1
T = 365
n_experiments = 30
gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment = []

n_classes = 3

for e in range (n_experiments):
  print('experiment:', e)
  envs = [Environment_Bids_class(bids, n_arms, prices[class_id-1], p_arms[class_id-1], cost, class_id) for class_id in range(1, n_classes+1)]

  gpts_learner = [GPTS_Learner(n_bids, bids, n_arms, cost, prices[class_id]) for class_id in range(0, n_classes)]
  gpucb_learner = [GPUCB1_Learner(n_bids, bids, n_arms, cost, prices[class_id]) for class_id in range(0, n_classes)]
  # Arrays to store collected rewards for each time step
  gpts_collected_reward = np.zeros((T, n_classes))
  gpucb_collected_reward = np.zeros((T, n_classes))

  for t in range (T):
    # GP Thompson Sampling
    for class_id in range(n_classes):
      pulled_arm = gpts_learner[class_id].pull_arm()
      pulled_bid, cost_click_obs, daily_click_obs, conv_rate_obs, reward  = envs[class_id].round( pulled_arm)
      gpts_learner[class_id].update(pulled_bid, cost_click_obs, daily_click_obs, pulled_arm, conv_rate_obs)
      gpts_collected_reward[t][class_id] = reward
      # GP UCB1
      pulled_arm = gpucb_learner[class_id].pull_arm()
      pulled_bid, cost_click_obs, daily_click_obs, conv_rate_obs, reward  = envs[class_id].round(pulled_arm)
      gpucb_learner[class_id].update(pulled_bid, cost_click_obs, daily_click_obs, pulled_arm, conv_rate_obs)
      gpucb_collected_reward[t][class_id] = reward


  gpts_rewards_per_experiment.append(gpts_collected_reward.sum(axis=1))
  gpucb_rewards_per_experiment.append(gpucb_collected_reward.sum(axis=1))

  # Calculate the optimal reward
opt = np.zeros(n_classes)
for class_id in range(0, n_classes):
  opt[class_id] = np.max([daily_click_fun_class(bids, class_id+1) *p_arms[class_id][i]*(prices[class_id][i]-cost)-cost_click_fun_class(bids, class_id+1) for i in range(n_arms)] )
optimal = opt.sum() * np.ones(T)
matrix_val = np.array([gpts_rewards_per_experiment, gpucb_rewards_per_experiment])
label  = ['GPTS','GPUCB1']
plot_function(matrix_val, optimal, label, 'Exercise 4 Result')
