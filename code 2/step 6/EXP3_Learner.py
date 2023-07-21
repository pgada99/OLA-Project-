import numpy as np

class Exp3_Learner:
    def __init__(self, n_arms, prices, cost, gamma, reward_max=1):
        # Constructor method. Initialize the Exp3 Learner with the given parameters.
        self.n_arms = n_arms
        self.gamma = gamma  # Exploration parameter (a value between 0 and 1)
        
        
        self.weights = np.ones(n_arms)  # Weights for each arm
        
        self.pulled_arms = []  # List to store the pulled arms for later analysis
        self.collected_rewards = []  # List to store the collected rewards for later analysis
        self.cumulative_rewards = np.zeros(n_arms)  # Cumulative rewards for each arm
        self.arm_selections = np.zeros(n_arms)  # Number of times each arm has been pulled
        self.t = 0  # Current round number
        self.prices = prices  # Prices for each arm (bid)
        self.cost = cost  # Cost per click for the advertiser
        self.reward_max =reward_max
    def pull_arm(self):
        # Calculate the probabilities for selecting each arm using the Exp3 algorithm.
        probabilities_dist =np.array([ np.maximum((1 - self.gamma) * self.weights[i] / np.sum(self.weights) + self.gamma / self.n_arms , 1e-5) for i in range(self.n_arms)])
        
        #np.max((1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.n_arms * np.ones(self.n_arms), 1e-5) for i in range(n_arms)
        # Select an arm based on the calculated probabilities.
        pulled_arm = np.random.choice(np.arange(self.n_arms), p=probabilities_dist)
        #pulled_arm = draw(probabilities_dist)
        return pulled_arm

    def update(self, pulled_arm, reward):
        # Increment the current round number (t) for the Exp3 Learner.
        self.t += 1
        # Store the pulled arm and the collected reward for later analysis.
        self.pulled_arms.append(pulled_arm)
        self.collected_rewards.append(reward)

        # Update the cumulative rewards and the number of times the pulled arm has been selected.
        self.cumulative_rewards[pulled_arm] += reward
        self.arm_selections[pulled_arm] += 1
        probabilities_dist =np.array([ np.maximum((1 - self.gamma) * self.weights[i] / np.sum(self.weights) + self.gamma / self.n_arms , 1e-10) for i in range(self.n_arms)])
        
        # Calculate the scaled reward for the arm selection (to avoid potential division by zero).
        scaled = reward*(self.prices[pulled_arm]-self.cost)/self.reward_max
        # Estimate the reward for the pulled arm and update the arm's weight using the Exp3 algorithm.
        estimated_reward = scaled / probabilities_dist[pulled_arm] 
        self.weights[pulled_arm] *= np.exp((self.gamma * estimated_reward) / self.n_arms)


def draw(weights):
    choice = np.random.uniform(0, sum(weights))
    choice_index = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choice_index

        choice_index += 1