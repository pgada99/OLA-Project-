import matplotlib.pyplot as plt
import numpy as np


prices= [350, 400, 450, 500, 550]
cost = 210



p = [
    [0.4, 0.38, 0.33, 0.25, 0.15],
    [0.22, 0.25, 0.30, 0.28, 0.21],
    [0.25, 0.22, 0.24, 0.3, 0.35]
]


# # Plot the probabilities
# y1 = p[0]
# y2 = p[1]
# y3 = p[2]

# plt.plot(prices, y1, label='C1', alpha = 0.2)
# plt.plot(prices, y2, label='C2', alpha = 0.2)
# plt.plot(prices, y3, label='C3', alpha = 1)

# # Adding labels and title
# plt.xlabel("Prices")
# plt.title("Conversion probabilities per price")
# #plt.title('Three Lines Plot')
# plt.legend()

# # Show the plot
# plt.show()




n_bids = 100
bids = np.linspace(0, 100, n_bids)


# Set all the curves for every class


# def daily_click_fun(x, C):
#   x = x/100
#   if C == 1: return 100*(1.0-np.exp(-4*x+2*x**3))
#   if C == 2: return 80*(1.0-np.exp(-3*x+2*x**3))
#   if C == 3: return 50*(1.0-np.exp(-4*x+2*x**3))

# def cost_click_fun(x, C):
#   x = x/100
#   if C == 1: return 100*(30-np.exp(-3*x+5*x**3))
#   if C == 2: return 550*(30-np.exp(-3*x+5*x**3))
#   if C == 3: return 900*(30-np.exp(-3*x+5*x**3)) 


# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 7))

# # Plot on the first subplot
# axes[0,0].plot(bids, daily_click_fun(bids, 1), color='green')
# axes[0,0].set_title('number of daily click')
# #-
# axes[0,1].plot(bids, cost_click_fun(bids, 1), color='blue')
# axes[0,1].set_title('cost of click')
# #-
# axes[1,0].plot(bids, daily_click_fun(bids, 2), color='green')
# axes[1,0].set_title('number of daily click')
# #.
# axes[1,1].plot(bids, cost_click_fun(bids, 2), color='blue')
# axes[1,1].set_title('cost of click')
# #-
# axes[2,0].plot(bids, daily_click_fun(bids, 3), color='green')
# axes[2,0].set_title('number of daily click')
# #.
# axes[2,1].plot(bids, cost_click_fun(bids, 3), color='blue')
# axes[2,1].set_title('cost of click')

# plt.tight_layout()
# plt.show()











# def cost_click_fun(x):
#   x = x/100
#   return (30-np.exp(-3*x+5*x**3))


# x = np.linspace(0,100, 100)
# # Create a figure and subplots
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
# # Plot on the first subplot
# axes[0].plot(x, daily_click_fun(x), color='green')
# axes[0].set_title('number of daily click')
# # Plot on the second subplot
# axes[1].plot(x, cost_click_fun(x), color='blue')
# axes[1].set_title('cost of click')

# # Adjust spacing between subplots
# plt.tight_layout()
# # Show the figure
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

n_bids = 100
bids = np.linspace(0, 100, n_bids)

# Set all the curves for every class
def daily_click_fun(x, C):
    x = x / 100
    if C == 1:
        return 100 * (1.0 - np.exp(-4 * x + 2 * x ** 3))
    if C == 2:
        return 80 * (1.0 - np.exp(-3 * x + 2 * x ** 3))
    if C == 3:
        return 50 * (1.0 - np.exp(-4 * x + 2 * x ** 3))


# Create a single plot
plt.figure(figsize=(8, 7))

# # Plot the number of daily clicks for each class
# plt.plot(bids, daily_click_fun(bids, 1), label='Class 1', color='blue', alpha = 0.2)
# plt.plot(bids, daily_click_fun(bids, 2), label='Class 2', color='orange', alpha = 1)
# plt.plot(bids, daily_click_fun(bids, 3), label='Class 3', color='green', alpha = 0.2)


# # Adding labels and title
# plt.xlabel("Bids")
# plt.ylabel("Values")
# plt.title("Number of Daily Clicks")
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()



def cost_click_fun(x, C):
    x = x / 100
    if C == 1:
        return 400 * (1.0 - np.exp(-5 * x + 2 * x ** 3))
    if C == 2:
        return 400 * (1.0 - np.exp(-3 * x + 2 * x ** 3))
    if C == 3:
        return 400 * (1.0 - np.exp(-4 * x + 2 * x ** 3))


# Create a single plot
plt.figure(figsize=(8, 7))

# Plot the number of daily clicks for each class
plt.plot(bids, cost_click_fun(bids, 1), label='Class 1', color='blue', alpha = 0.2)
plt.plot(bids, cost_click_fun(bids, 2), label='Class 2', color='orange', alpha = 0.2)
plt.plot(bids, cost_click_fun(bids, 3), label='Class 3', color='green', alpha = 1)


# Adding labels and title
plt.xlabel("Bids")
plt.ylabel("Values")
plt.title("Cost of Clicks per bid")
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()