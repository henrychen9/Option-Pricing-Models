import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# parameters
S_max, V_max, T, r = 200, 5.0, 1.0, 0.05
kappa, theta, sigma, rho = 2.0, 0.04, 0.3, -0.7
K, N, M, L = 100, 40, 40, 100

# discretization
ds, dv, dt = S_max / N, V_max / M, T / L
if dt > 0.1 * min(ds**2, dv**2):  # ensure stability
    dt = 0.1 * min(ds**2, dv**2)
    L = int(T // dt + 1)
    dt = T / L

# grids for asset price (s), variance (v), and time (t)
S, V, t = np.linspace(0, S_max, N+1), np.linspace(0, V_max, M+1), np.linspace(0, T, L+1)
V_grid = np.zeros((N+1, M+1, L+1))

# terminal and boundary conditions
V_grid[:, :, -1] = np.maximum(S[:, np.newaxis] - K, 0)  # payoff at maturity
V_grid[0, :, :], V_grid[-1, :, :] = 0, S_max - K * np.exp(-r * (T - t))  # boundaries in s
V_grid[:, 0, :] = np.maximum(S - K, 0)[:, np.newaxis]  # variance v=0
V_grid[:, -1, :] = 2 * V_grid[:, -2, :] - V_grid[:, -3, :]  # upper v boundary

for l in range(L-1, -1, -1):
    main_diag = np.ones((N+1)*(M+1))
    lower_diag_S, upper_diag_S = np.zeros_like(main_diag), np.zeros_like(main_diag)
    lower_diag_V, upper_diag_V = np.zeros_like(main_diag), np.zeros_like(main_diag)
    b = np.zeros_like(main_diag)

    for i in range(1, N):
        for j in range(1, M):
            idx = i * (M+1) + j

            # coefficients for pde terms
            alpha = 0.5 * S[i]**2 * V[j] / (1.5 * ds**2)
            beta = 0.5 * sigma**2 * V[j] / dv**2
            alpha, beta = alpha / (1 + 2 * alpha), beta / (1 + beta)  # normalize diffusion terms

            gamma = np.clip(rho * sigma * S[i] * V[j] / (4 * ds * dv), -0.5, 0.5)  # mixed derivative stabilization
            delta, epsilon = r * S[i] / (2 * ds), kappa * (theta - V[j]) / (2 * dv)
            epsilon = epsilon / (1 + abs(epsilon))  # prevent large drifts

            # ensure diagonal dominance for numerical stability
            main_diag[idx] = max(1 + dt * (2 * alpha + 2 * beta + r), 1e-8)

            # finite difference scheme
            if i < N:
                upper_diag_S[idx] = -dt * (alpha + delta)
            if i > 0:
                lower_diag_S[idx] = -dt * (alpha - delta)
            if j < M:
                upper_diag_V[idx] = -dt * (beta + epsilon)
            if j > 0:
                lower_diag_V[idx] = -dt * (beta - epsilon)

            # handle mixed derivative term properly
            main_diag[idx] += dt * gamma if i > 0 and j > 0 else -dt * gamma
            b[idx] = V_grid[i, j, l+1]

    # apply boundary conditions
    for j in range(M+1):
        b[j] = 0  # s = 0 boundary
        b[N*(M+1) + j] = S_max - K * np.exp(-r * (T - t[l]))  # s = s_max boundary

    for i in range(N+1):
        b[i*(M+1)] = max(S[i] - K, 0)  # v = 0 boundary
        b[i*(M+1) + M] = max(S[i] - K * np.exp(-r * (T - t[l])), 0)  # v = v_max boundary

    # construct sparse matrix for implicit scheme
    A = diags([lower_diag_S, lower_diag_V, main_diag, upper_diag_V, upper_diag_S],
              [-M-1, -1, 0, 1, M+1], shape=((N+1)*(M+1), (N+1)*(M+1)), format="csr")

    # solve linear system for current time step
    try:
        V_sol = spsolve(A, b)
        if np.isnan(V_sol).any():
            print(f"warning: nan values detected at time step {l}.")
    except Exception as e:
        print(f"error solving linear system at time step {l}: {e}")
        break

    V_grid[:, :, l] = V_sol.reshape((N+1, M+1))
    print(f"time step {l} processed.")


option_price = V_grid[:, :, 0]

# plot the option price surface
S_grid, V_grid = np.meshgrid(S, V, indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, V_grid, option_price, cmap='viridis')
ax.set_xlabel('asset price (s)')
ax.set_ylabel('variance (v)')
ax.set_zlabel('option price')
plt.title('heston model option price surface (implicit method)')
plt.show()
