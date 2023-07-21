import numpy as np
from Learner import *

# Thompson Sampling learner
class TS_Learner(Learner):
  def __init__(self, n_arms, prices, cost):
    super().__init__(n_arms, prices, cost)
    self.beta_parameters = np.ones((n_arms, 2))

  def pull_arm(self):
    idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])*(self.prices-self.cost*np.ones(self.n_arms)))
    return idx

  def update(self, pulled_arm, conversion_rate, click_day_obs):
    self.t += 1
    self.update_observations(pulled_arm, conversion_rate)
    self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + conversion_rate*click_day_obs
    self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1*click_day_obs - conversion_rate*click_day_obs