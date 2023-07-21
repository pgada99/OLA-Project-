
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


class advertising_curve():
  def __init__(self, n_bids, bids):
    self.n_bids = n_bids
    self.bids = bids
    self.means = np.zeros(self.n_bids)  # Mean cost per click for each bid
    self.sigmas = np.ones(self.n_bids)  # Standard deviation of cost per click for each bid
    self.collected = np.array([])
    self.pulled_bids = []
    self.click_bid = [[] for i in range(n_bids)]  # List to store click observations for each bid
    alpha = 10.0
    self.t= 0
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5))  # Define the kernel for Gaussian Process
    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, n_restarts_optimizer=5)  # Gaussian Process for click cost

  def update_curve(self):
    self.t+=1
    x = np.atleast_2d(self.pulled_bids).T  # Convert pulled bids to a column vector
    y = self.collected  # Collected click cost observations
    if self.t %20==0:
      self.gp.fit(x, y)  # Fit Gaussian Process for click cost

    self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.bids).T, return_std=True)  # Predict mean and standard deviation of click cost
    self.sigmas= np.maximum(self.sigmas, 1e-2)  # Ensure minimum value for standard deviation

  def update_observations(self, pulled_bid, obs):
    self.pulled_bids.append(self.bids[pulled_bid])
    self.click_bid[pulled_bid].append(obs)  # Add click cost observation for the pulled bid
    self.collected = np.append(self.collected, obs)  # Add click cost observation to collected array