import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# simulate Heston paths using full-truncation Euler
def simulate_heston_paths(S0, v0, r, kappa, theta, sigma, rho, T, n_steps, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    S_paths = np.zeros((n_paths, n_steps + 1))
    v_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    
    for t in range(1, n_steps + 1):
        Z1 = np.random.normal(size=n_paths)
        Z2 = np.random.normal(size=n_paths)
        # generate correlated brownian motions
        dW_S = np.sqrt(dt) * Z1
        dW_v = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
        
        # update variance using full truncation (non-negative only)
        v_prev = v_paths[:, t - 1]
        v_paths[:, t] = v_prev + kappa * (theta - np.maximum(v_prev, 0)) * dt \
                        + sigma * np.sqrt(np.maximum(v_prev, 0)) * dW_v
        v_paths[:, t] = np.maximum(v_paths[:, t], 0)
        
        # update asset price using log-Euler
        S_prev = S_paths[:, t - 1]
        S_paths[:, t] = S_prev * np.exp((r - 0.5 * np.maximum(v_prev, 0)) * dt \
                                        + np.sqrt(np.maximum(v_prev, 0)) * dW_S)
    
    return S_paths, v_paths

# price American call
def american_call_price(S_paths, r, K, T):
    n_paths, n_steps_plus_one = S_paths.shape
    n_steps = n_steps_plus_one - 1
    dt = T / n_steps
    
    # initial cashflow = payoff at maturity
    cashflow = np.maximum(S_paths[:, -1] - K, 0)
    exercise_time = np.full(n_paths, n_steps)  # init with final step (no early exercise)
    
    # backward loop through time steps
    for t in range(n_steps - 1, 0, -1):
        discounted_cf = cashflow * np.exp(-r * dt)
        in_the_money = np.where((S_paths[:, t] > K) & (exercise_time == n_steps))[0]
        if len(in_the_money) == 0:
            continue
        
        # regression: fit continuation value
        immediate_payoff = np.maximum(S_paths[in_the_money, t] - K, 0)
        X = np.vstack([np.ones(len(in_the_money)),
                       S_paths[in_the_money, t],
                       S_paths[in_the_money, t]**2]).T
        Y = discounted_cf[in_the_money]
        coeff, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        continuation_value = X.dot(coeff)
        
        # compare payoff vs continuation; update if early exercise optimal
        exercise = in_the_money[immediate_payoff >= continuation_value]
        if len(exercise) > 0:
            cashflow[exercise] = immediate_payoff[np.isin(in_the_money, exercise)]
            exercise_time[exercise] = t
    
    discounts = np.exp(-r * exercise_time * dt)
    price = np.mean(cashflow * discounts)
    return price, exercise_time

# wrapper: simulate + price American call under Heston
def american_call_price_mc(S0, v0, r, kappa, theta, sigma, rho, T, K, n_steps, n_paths, seed=None):
    S_paths, _ = simulate_heston_paths(S0, v0, r, kappa, theta, sigma, rho, T, n_steps, n_paths, seed)
    price, _ = american_call_price(S_paths, r, K, T)
    return price

# build price surface over S0 and v0 grid
S_max = 200.0
v_max = 0.5
n_S = 20
n_v = 20

S_grid = np.linspace(0, S_max, n_S)
v_grid = np.linspace(0, v_max, n_v)

# model params
T = 1.0
K = 100.0
r = 0.05
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7

# monte carlo params
n_steps = 100
n_paths = 1000
seed = 50

# loop over S0, v0 grid and compute prices
price_surface = np.zeros((n_S, n_v))
total_points = n_S * n_v
counter = 0

for i, S0 in enumerate(S_grid):
    for j, v0 in enumerate(v_grid):
        price = american_call_price_mc(S0, v0, r, kappa, theta, sigma, rho, T, K, n_steps, n_paths, seed)
        price_surface[i, j] = price
        counter += 1
        if counter % 100 == 0:
            print(f"progress: {counter}/{total_points} grid points processed")

print("monte carlo simulation complete.")

# plot 3D price surface
S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_mesh, v_mesh, price_surface, cmap=cm.viridis, edgecolor='none')
ax.set_xlabel('Initial Variance v0')
ax.set_ylabel('Initial Asset Price S0')
ax.set_zlabel('Option Price')
ax.set_title('Monte Carlo American Call Option Price Surface under Heston Model')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
