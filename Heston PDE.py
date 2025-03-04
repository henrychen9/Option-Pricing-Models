import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.system('cls')

# Parameters
S_max = 200       # Maximum asset price
V_max = 1.0       # Maximum variance
T = 1.0           # Time to maturity
r = 0.05          # Risk-free rate
kappa = 2.0       # Mean reversion rate
theta = 0.04      # Long-term variance
sigma = 0.3       # Volatility of volatility
rho = -0.7        # Correlation
K = 100           # Strike price
N = 100           # Number of steps in S direction
M = 100           # Number of steps in V direction
L = 100           # Number of time steps

# Discretization
ds = S_max / N
dv = V_max / M
dt = T / L

# Stability check (CFL condition estimate)
max_dt = 1 / np.max([(S_max**2 / ds**2), (sigma**2 / dv**2), (kappa / dv)])
if dt > max_dt:
    print(f"Warning: dt = {dt:.6f} is too large! Stability limit is {max_dt:.6f}. Consider reducing L.")
    dt = max_dt * 0.9  # Slightly below the stability threshold

# Grids
S = np.linspace(0, S_max, N+1)
V = np.linspace(0, V_max, M+1)
t = np.linspace(0, T, L+1)

# Initialize option price grid
V_grid = np.zeros((N+1, M+1, L+1))

# Terminal condition (payoff at maturity)
V_grid[:, :, -1] = np.maximum(S[:, np.newaxis] - K, 0)

# Boundary conditions
V_grid[0, :, :] = 0  # S = 0
V_grid[-1, :, :] = S_max - K * np.exp(-r * (T - t))  # S = S_max
V_grid[:, 0, :] = np.maximum(S - K, 0)[:, np.newaxis]
V_grid[:, -1, :] = np.maximum(S[:, np.newaxis] - K * np.exp(-r * (T - t)), 0)


"""
print("\nBoundary Condition at S=S_max (V_grid[-1, :, :]):")
print(V_grid[-1, :, :])


# Print all boundary and terminal conditions
print("Terminal Condition (V_grid[:, :, -1]):")
print(V_grid[:, :, -1])

print("\nBoundary Condition at S=0 (V_grid[0, :, :]):")
print(V_grid[0, :, :])

print("\nBoundary Condition at S=S_max (V_grid[-1, :, :]):")
print(V_grid[-1, :, :])

print("\nBoundary Condition at V=0 (V_grid[:, 0, :]):")
print(V_grid[:, 0, :])

print("\nBoundary Condition at V=V_max (V_grid[:, -1, :]):")
print(V_grid[:, -1, :])
"""

# Explicit finite difference scheme
for l in range(L-1, -1, -1):
    for i in range(1, N-1):
        for j in range(1, M-1):
            # Partial derivatives
            dV_dS = (V_grid[i+1, j, l+1] - V_grid[i-1, j, l+1]) / (2 * ds)
            dV_dV = (V_grid[i, j+1, l+1] - V_grid[i, j-1, l+1]) / (2 * dv)
            d2V_dS2 = (V_grid[i+1, j, l+1] - 2 * V_grid[i, j, l+1] + V_grid[i-1, j, l+1]) / (ds**2)
            d2V_dV2 = (V_grid[i, j+1, l+1] - 2 * V_grid[i, j, l+1] + V_grid[i, j-1, l+1]) / (dv**2)
            d2V_dSdV = (V_grid[i+1, j+1, l+1] - V_grid[i+1, j-1, l+1] - V_grid[i-1, j+1, l+1] + V_grid[i-1, j-1, l+1]) / (4 * ds * dv)

            # Explicit Heston PDE update
            V_grid[i, j, l] = V_grid[i, j, l+1] + dt * (
                0.5 * S[i]**2 * V[j] * d2V_dS2 +
                rho * sigma * S[i] * V[j] * d2V_dSdV +
                0.5 * sigma**2 * V[j] * d2V_dV2 +
                r * S[i] * dV_dS +
                kappa * (theta - V[j]) * dV_dV - r * V_grid[i, j, l+1]
            )

        # Progress update every 10 steps
        if i % 10 == 0:
            print(f"Time step {l}, Grid row {i}/{N-1} processed.")

# Extract the option price at t=0
option_price = V_grid[:, :, 0]

# Plot the option price surface
S_grid, V_grid = np.meshgrid(S, V, indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, V_grid, option_price, cmap='viridis')
ax.set_xlabel('Asset Price (S)')
ax.set_ylabel('Variance (V)')
ax.set_zlabel('Option Price')
plt.title('Heston Model Option Price Surface')
plt.show()
