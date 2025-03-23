import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline

def monte_carlo_heston_option_price(K, T, r, kappa, theta, sigma, rho, N, M, dt, S_vals, V_vals):
    # create grid of (S0, V0) values
    S_grid, V_grid = np.meshgrid(S_vals, V_vals, indexing='ij')
    option_prices = np.zeros_like(S_grid)

    # run monte carlo simulations
    for i in range(len(S_vals)):
        print(f"grid {i}/{len(S_vals)} complete")

        for j in range(len(V_vals)):
            S0 = S_grid[i, j]
            V0 = V_grid[i, j]

            # generate correlated brownian motions
            dW_V = np.random.randn(M, N) * np.sqrt(dt)
            dW_S = rho * dW_V + np.sqrt(1 - rho**2) * np.random.randn(M, N) * np.sqrt(dt)

            # initialize asset and variance paths
            S = np.zeros((M, N + 1))
            V = np.zeros((M, N + 1))
            S[:, 0] = S0
            V[:, 0] = V0

            # simulate paths
            for t in range(N):
                sqrt_V = np.sqrt(np.maximum(V[:, t], 0))
                V[:, t+1] = V[:, t] + kappa * (theta - V[:, t]) * dt + sigma * sqrt_V * dW_V[:, t] \
                            + 0.25 * sigma**2 * (dW_V[:, t]**2 - dt)
                V[:, t+1] = np.maximum(V[:, t+1], 0)
                S[:, t+1] = S[:, t] * np.exp((r - 0.5 * V[:, t]) * dt + sqrt_V * dW_S[:, t])

            # calculate discounted payoff
            call_payoffs = np.maximum(S[:, -1] - K, 0)
            option_prices[i, j] = np.exp(-r * T) * np.mean(call_payoffs)

    return S_grid, V_grid, option_prices

def plot_monte_carlo_surface(S_grid, V_grid, option_prices):
    # plot monte carlo price surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, V_grid, option_prices, cmap='viridis')
    ax.set_xlabel("stock price (S)")
    ax.set_ylabel("variance (V)")
    ax.set_zlabel("call option price")
    ax.set_title("monte carlo heston option price surface")
    plt.show()

def interpolate_price(S_vals, V_vals, option_prices, S0, V0):
    # interpolate price at specific point
    interp_func = RectBivariateSpline(S_vals, V_vals, option_prices)
    interp_price = interp_func(S0, V0)[0, 0]
    print(f"monte carlo option price at S0 ~ {S0} and V0 ~ {V0} (via spline interpolation):", interp_price)
    return interp_price

# set model parameters
K = 100      # strike price
T = 1.0      # time to maturity (years)
r = 0.05     # risk-free rate
kappa = 2.0  # mean reversion speed
theta = 0.04 # long-run variance
sigma = 0.3  # volatility of volatility
rho = -0.7   # correlation between stock and variance
N = 150      # number of time steps
M = 3000     # number of monte carlo paths
dt = T / N

# create grid of (S0, V0) values
S_vals = np.linspace(0, 200, 10)
V_vals = np.linspace(0, 0.5, 10)

# run monte carlo simulations
S_grid, V_grid, option_prices = monte_carlo_heston_option_price(K, T, r, kappa, theta, sigma, rho, N, M, dt, S_vals, V_vals)

# plot monte carlo price surface
plot_monte_carlo_surface(S_grid, V_grid, option_prices)

# interpolate price at specific point
interp_price = interpolate_price(S_vals, V_vals, option_prices, 100.0, 0.04)
