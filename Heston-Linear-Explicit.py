# ===================================================================================
# WARNING: This code would not work due to instability in explicit FD implementation!
# For reference only.
# ===================================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def heston_explicit_fd(r, kappa, theta, sigma, rho, T, K, S_max, v_max, Ns, Nv, Nt, print_interval=100):
    # solve Heston PDE using explicit finite difference method
    ds = S_max / (Ns - 1)
    dv = v_max / (Nv - 1)
    dt = T / (Nt - 1)

    S_grid = np.linspace(0, S_max, Ns)
    v_grid = np.linspace(0, v_max, Nv)

    V = np.zeros((Ns, Nv))
    for i in range(Ns):
        V[i, :] = max(S_grid[i] - K, 0)  # initial condition: payoff at maturity

    total_steps = Nt - 1
    sys.stdout.write("starting explicit time-stepping...\n")
    sys.stdout.flush()

    for n in range(total_steps, 0, -1):
        t = (n - 1) * dt
        tau = T - t  # flip time axis (working backward)

        V_new = V.copy()

        V_new[0, :] = 0.0
        V_new[-1, :] = S_max - K * np.exp(-r * tau)
        V_new[:, 0] = V_new[:, 1]
        V_new[:, -1] = V_new[:, -2]

        for i in range(1, Ns - 1):
            S_val = S_grid[i]
            for j in range(1, Nv - 1):
                v_val = v_grid[j]

                V_SS = (V[i + 1, j] - 2 * V[i, j] + V[i - 1, j]) / ds**2
                V_S  = (V[i + 1, j] - V[i - 1, j]) / (2 * ds)
                V_vv = (V[i, j + 1] - 2 * V[i, j] + V[i, j - 1]) / dv**2
                V_v  = (V[i, j + 1] - V[i, j - 1]) / (2 * dv)
                V_Sv = (V[i + 1, j + 1] - V[i + 1, j - 1] - V[i - 1, j + 1] + V[i - 1, j - 1]) / (4 * ds * dv)

                L_V = (0.5 * S_val**2 * v_val * V_SS +
                       r * S_val * V_S +
                       0.5 * sigma**2 * v_val * V_vv +
                       kappa * (theta - v_val) * V_v +
                       rho * sigma * S_val * v_val * V_Sv -
                       r * V[i, j])

                V_new[i, j] = V[i, j] - dt * L_V  # forward Euler step

        V = V_new.copy()

        if (total_steps - n) % print_interval == 0:
            sys.stdout.write(f"processed {total_steps - n} out of {total_steps} time steps\n")
            sys.stdout.flush()

    sys.stdout.write("explicit time-stepping complete\n")
    sys.stdout.flush()
    return S_grid, v_grid, V

# grid + model setup
S_max = 200.0   # max stock price
v_max = 0.5     # max variance
Ns    = 50      # number of stock grid points
Nv    = 50      # number of variance grid points
Nt    = 2000    # number of time steps
print_interval = 100

K = 100      # strike price
T = 1.0      # time to maturity
r = 0.05     # risk-free rate
kappa = 2.0  # mean reversion speed
theta = 0.04 # long-run variance
sigma = 0.3  # vol of vol
rho = -0.7   # correlation


S_grid, v_grid, V = heston_explicit_fd(r, kappa, theta, sigma, rho, T, K, S_max, v_max, Ns, Nv, Nt, print_interval)

# plot 3D surface
S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_mesh, v_mesh, V, cmap='viridis', edgecolor='none')
ax.set_xlabel('asset price S')
ax.set_ylabel('variance v')
ax.set_zlabel('option price')
ax.set_title('european call under heston model (explicit fd)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
