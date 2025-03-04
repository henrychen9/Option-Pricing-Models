import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# Parameters
S_max = 200       # Maximum asset price
V_max = 5.0       # Maximum variance
T = 1.0           # Time to maturity
r = 0.05          # Risk-free rate
kappa = 2.0       # Mean reversion rate
theta = 0.04      # Long-term variance
sigma = 0.3       # Volatility of volatility
rho = -0.7        # Correlation
K = 100           # Strike price
N = 40           # Number of steps in S direction
M = 40           # Number of steps in V direction
L = 100           # Number of time steps

# Discretization
ds = S_max / N
dv = V_max / M
dt = T / L
if dt > 0.05 * min(ds**2, dv**2):
    dt = 0.05 * min(ds**2, dv**2)
    L = int(T // dt + 1)
    dt = T / L


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
V_grid[:, 0, :] = np.maximum(S - K, 0)[:, np.newaxis]  # V = 0
V_grid[:, -1, :] = np.maximum(S[:, np.newaxis] - K * np.exp(-r * (T - t)), 0)


for l in range(L-1, -1, -1):
    # Initialize sparse matrix components
    main_diag = np.ones((N+1)*(M+1))  # Set to ones to avoid singularity
    lower_diag_S = np.zeros((N+1)*(M+1))  # Lower diagonal in S direction
    upper_diag_S = np.zeros((N+1)*(M+1))  # Upper diagonal in S direction
    lower_diag_V = np.zeros((N+1)*(M+1))  # Lower diagonal in V direction
    upper_diag_V = np.zeros((N+1)*(M+1))  # Upper diagonal in V direction
    b = np.zeros((N+1)*(M+1))  # Right-hand side vector

    for i in range(1, N):
        for j in range(1, M):
            idx = i * (M+1) + j  # Flattened index in 2D grid

            # Coefficients for the PDE terms
            alpha = 0.5 * S[i]**2 * V[j] / (1.5 * ds**2)  # Reduce by factor 1.5
            beta = 0.5 * sigma**2 * V[j] / dv**2
            alpha = alpha / (1 + 2 * alpha)  # Reduce excessive S-diffusion impact
            beta = beta / (1 + beta)  # Smooth out variance diffusion

            
            gamma = rho * sigma * S[i] * V[j] / (4 * ds * dv)
            gamma = np.clip(gamma, -5, 5)  # Prevent excessive mixed derivative influence
            delta = r * S[i] / (2 * ds)
            epsilon = kappa * (theta - V[j]) / (2 * dv)
            epsilon = epsilon / (1 + abs(epsilon))

            # Ensure diagonals are positive and nonzero
            main_diag[idx] = max(1 + dt * (2 * alpha + 2 * beta + r), 1e-8)

            # S-direction neighbors
            if i < N:  # Upper S term
                upper_diag_S[idx] = -dt * (alpha + delta)
            if i > 0:  # Lower S term
                lower_diag_S[idx] = -dt * (alpha - delta)

            # V-direction neighbors
            if j < M:  # Upper V term
                upper_diag_V[idx] = -dt * (beta + epsilon)
            if j > 0:  # Lower V term
                lower_diag_V[idx] = -dt * (beta - epsilon)

            # Mixed derivative terms (cross terms)
            if i < N and j < M:
                main_diag[idx] -= dt * gamma
            if i > 0 and j > 0:
                main_diag[idx] += dt * gamma
            """if i == 16 and 14 <= j <= 21:  # Focus on the exploding region
                print(f"🚨 i={i}, j={j}, alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}, epsilon={epsilon}")
            if np.isnan(V_grid[:, :, l+1]).any():
                nan_indices = np.argwhere(np.isnan(V_grid[:, :, l+1]))
                for idx in nan_indices:
                    i, j = idx
                    print(f"🚨 NaN at V_grid[{i}, {j}, 1013]. Previous value: {V_grid[i, j, 1014]}")
                print(f"🚨 First NaNs at time step {l+1}, at indices: {nan_indices}")
        
                break"""

            # Right-hand side vector
            b[idx] = V_grid[i, j, l+1]

    # Apply boundary conditions (EXPLICITLY)
    for j in range(M+1):
        b[j] = 0  # S = 0 boundary
        b[N*(M+1) + j] = S_max - K * np.exp(-r * (T - t[l]))  # S = S_max boundary

    for i in range(N+1):
        b[i*(M+1)] = max(S[i] - K, 0)  # V = 0 boundary
        b[i*(M+1) + M] = max(S[i] - K * np.exp(-r * (T - t[l])), 0)  # V = V_max boundary

    # Construct sparse matrix A (Block-Tridiagonal Form)
    diagonals = [
        lower_diag_S,  # Lower diagonal in S
        lower_diag_V,  # Lower diagonal in V
        main_diag,     # Main diagonal
        upper_diag_V,  # Upper diagonal in V
        upper_diag_S   # Upper diagonal in S
    ]
    offsets = [-M-1, -1, 0, 1, M+1]  # Offsets for block-tridiagonal structure

    A = diags(diagonals, offsets, shape=((N+1)*(M+1), (N+1)*(M+1)), format="csr")

    # Solve the linear system A * V_sol = b
    try:
        V_sol = spsolve(A, b)
        if np.isnan(V_sol).any():
            print(f"Warning: NaN values detected at time step {l}.")
    except Exception as e:
        print(f"Error solving linear system at time step {l}: {e}")
        break

    # Reshape the solution into grid format
    V_grid[:, :, l] = V_sol.reshape((N+1, M+1))

    print(f"Time step {l} processed.")



print("Option price at S=200, V=0.1:", V_grid[N//1, M//10, 0])
print("Option price at S=200, V=0.5:", V_grid[N//1, M//2, 0])
print("Option price at S=200, V=0.9:", V_grid[N//1, int(0.9 * M), 0])

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
plt.title('Heston Model Option Price Surface (Implicit Method)')
plt.show()