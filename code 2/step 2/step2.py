
# Importing Libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings("ignore")

from advertising_curve import *
from GP_Learner_advert import *
from GPTS_Learner_advert import *
from GPUCB1_Learner_advert import *
from BiddingEnvironment import *
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


# def daily_click_fun(x):
#   x = x/100
#   return 100*(1.0-np.exp(-4*x+3*x**3))

# def cost_click_fun(x):
#   x = x/100
#   return (30-np.exp(-3*x+5*x**3))

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
prices= np.array([350, 400, 450, 500, 550])
cost = 210

# Number of rounds
T = 365




min_bid = 0
max_bid = 100
n_bids = 10
bids = np.linspace(min_bid, max_bid, n_bids)


# EXERCISE 2 EXPERIMENT ------------------------------------------------------

sigma_cc = 1
sigma_dc = 1
T = 365
n_experiments = 50


gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment = []

for e in range (0,n_experiments):
  print('experiment',e)
  env = BiddingEnvironment(bids,  prices, p_arms, cost)
  gpts_learner = GPTS_Learner_advert(n_bids, bids, prices, p_arms, cost)
  gpucb_learner = GPUCB1_Learner_advert(n_bids, bids, prices, p_arms, cost)
  # Arrays to store collected rewards for each time step
  gpts_collected_reward = np.zeros(T)
  gpucb_collected_reward = np.zeros(T)

  for t in range (T):
    # GP Thompson Sampling
    pulled_bid = gpts_learner.pull_arm()
    reward, cost_click_obs, daily_click_obs = env.round(pulled_bid)
    gpts_learner.update(pulled_bid, cost_click_obs, daily_click_obs)
    gpts_collected_reward[t]= reward

    # GP UCB1 Sampling
    pulled_bid = gpucb_learner.pull_arm()
    reward, cost_click_obs, daily_click_obs = env.round(pulled_bid)
    gpucb_learner.update(pulled_bid, cost_click_obs, daily_click_obs)
    gpucb_collected_reward[t]= reward

  gpts_rewards_per_experiment.append(gpts_collected_reward)
  gpucb_rewards_per_experiment.append(gpucb_collected_reward)



  # Calculate the optimal reward
opt = np.max([daily_click_fun(bids) *p_arms[i]*(prices[i]-cost)-cost_click_fun(bids) for i in range(n_arms)] )
optimal = opt * np.ones(T)
matrix_val = np.array([gpts_rewards_per_experiment, gpucb_rewards_per_experiment])
label  = ['GPTS','GPUCB1']
plot_function(matrix_val, optimal, label, 'Exercise 2 Result')