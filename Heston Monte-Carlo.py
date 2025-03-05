import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parameters
K = 100      # strike price
T = 1.0      # time to maturity (years)
r = 0.05     # risk-free rate
kappa = 2.0  # mean reversion speed
theta = 0.04 # long-run variance
sigma = 0.3  # volatility of volatility
rho = -0.7   # correlation between stock and variance
N = 200     # number of time steps
M = 5000    # number of monte carlo paths
dt = T / N

# define grid of (S0, V0) values
S_vals = np.linspace(50, 150, 10)  # stock price range
V_vals = np.linspace(0.01, 0.1, 10)  # variance range
S_grid, V_grid = np.meshgrid(S_vals, V_vals)
option_prices = np.zeros_like(S_grid)

# monte carlo pricing loop
for i in range(len(S_vals)):
    print(f"grid {i}/{len(S_vals)} complete.")

    for j in range(len(V_vals)):
        S0 = S_grid[i, j]
        V0 = V_grid[i, j]

        # generate correlated brownian motions
        dW_V = np.random.randn(M, N) * np.sqrt(dt)
        dW_S = rho * dW_V + np.sqrt(1 - rho**2) * np.random.randn(M, N) * np.sqrt(dt)

        # initialize paths
        S = np.zeros((M, N + 1))
        V = np.zeros((M, N + 1))
        S[:, 0] = S0
        V[:, 0] = V0

        # milstein's update for variance, euler's update for stock price
        for t in range(N):
            sqrt_V = np.sqrt(np.maximum(V[:, t], 0))
            V[:, t+1] = V[:, t] + kappa * (theta - V[:, t]) * dt + sigma * sqrt_V * dW_V[:, t] \
                        + 0.25 * sigma**2 * (dW_V[:, t]**2 - dt)
            V[:, t+1] = np.maximum(V[:, t+1], 0)
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * V[:, t]) * dt + sqrt_V * dW_S[:, t])

        # compute call option price
        call_payoffs = np.maximum(S[:, -1] - K, 0)
        option_prices[i, j] = np.exp(-r * T) * np.mean(call_payoffs)

# plot monte carlo surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, V_grid, option_prices, cmap='viridis')

ax.set_xlabel("Stock Price (S)")
ax.set_ylabel("Variance (V)")
ax.set_zlabel("Call Option Price")
ax.set_title("Monte Carlo Heston Option Price Surface")

plt.show()