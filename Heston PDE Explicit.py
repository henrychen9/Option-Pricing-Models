import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.system('cls')

# parameters
S_max, V_max, T, r = 200, 1.0, 1.0, 0.05
kappa, theta, sigma, rho = 2.0, 0.04, 0.3, -0.7
K, N, M, L = 100, 100, 100, 100

# discretization
ds, dv, dt = S_max / N, V_max / M, T / L

# stability check using cfl condition
max_dt = 1 / np.max([(S_max**2 / ds**2), (sigma**2 / dv**2), (kappa / dv)])
if dt > max_dt:
    print(f"warning: dt = {dt:.6f} is too large! stability limit is {max_dt:.6f}. reducing dt.")
    dt = max_dt * 0.9  # set dt slightly below stability threshold

# grids for asset price (s), variance (v), and time (t)
S, V, t = np.linspace(0, S_max, N+1), np.linspace(0, V_max, M+1), np.linspace(0, T, L+1)
V_grid = np.zeros((N+1, M+1, L+1))

# terminal condition (payoff at maturity)
V_grid[:, :, -1] = np.maximum(S[:, np.newaxis] - K, 0)

# boundary conditions
V_grid[0, :, :] = 0  # s = 0
V_grid[-1, :, :] = S_max - K * np.exp(-r * (T - t))  # s = s_max
V_grid[:, 0, :] = np.maximum(S - K, 0)[:, np.newaxis]  # v = 0
V_grid[:, -1, :] = np.maximum(S[:, np.newaxis] - K * np.exp(-r * (T - t)), 0)  # v = v_max

# explicit finite difference scheme
for l in range(L-1, -1, -1):
    for i in range(1, N-1):
        for j in range(1, M-1):
            # first and second order partial derivatives
            dV_dS = (V_grid[i+1, j, l+1] - V_grid[i-1, j, l+1]) / (2 * ds)
            dV_dV = (V_grid[i, j+1, l+1] - V_grid[i, j-1, l+1]) / (2 * dv)
            d2V_dS2 = (V_grid[i+1, j, l+1] - 2 * V_grid[i, j, l+1] + V_grid[i-1, j, l+1]) / (ds**2)
            d2V_dV2 = (V_grid[i, j+1, l+1] - 2 * V_grid[i, j, l+1] + V_grid[i, j-1, l+1]) / (dv**2)
            d2V_dSdV = (V_grid[i+1, j+1, l+1] - V_grid[i+1, j-1, l+1] - V_grid[i-1, j+1, l+1] + V_grid[i-1, j-1, l+1]) / (4 * ds * dv)

            # explicit heston pde update
            V_grid[i, j, l] = V_grid[i, j, l+1] + dt * (
                0.5 * S[i]**2 * V[j] * d2V_dS2 +
                rho * sigma * S[i] * V[j] * d2V_dSdV +
                0.5 * sigma**2 * V[j] * d2V_dV2 +
                r * S[i] * dV_dS +
                kappa * (theta - V[j]) * dV_dV - r * V_grid[i, j, l+1]
            )

        # progress update every 10 rows
        if i % 10 == 0:
            print(f"time step {l}, grid row {i}/{N-1} processed.")

# extract the option price at t=0
option_price = V_grid[:, :, 0]

# plot the option price surface
S_grid, V_grid = np.meshgrid(S, V, indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, V_grid, option_price, cmap='viridis')
ax.set_xlabel('asset price (s)')
ax.set_ylabel('variance (v)')
ax.set_zlabel('option price')
plt.title('heston model option price surface')
plt.show()
