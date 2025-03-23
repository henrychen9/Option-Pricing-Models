import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d surface plots
from scipy.interpolate import RectBivariateSpline  # for interpolation

def heston_implicit_fd(r, kappa, theta, sigma, rho, T, K, S_max, v_max, Ns, Nv, Nt, print_interval=10):
    # solves the heston pde for a european call option using a fully implicit finite difference scheme
    
    # define grid steps
    ds = S_max / (Ns - 1)
    dv = v_max / (Nv - 1)
    dt = T / (Nt - 1)
    
    # create grids for asset price and variance
    S_grid = np.linspace(0, S_max, Ns)
    v_grid = np.linspace(0, v_max, Nv)
    
    # set terminal condition for european call option
    V = np.zeros((Ns, Nv))
    for i in range(Ns):
        payoff = max(S_grid[i] - K, 0)
        V[i, :] = payoff
    
    total_steps = Nt - 1
    sys.stdout.write("Starting time-stepping...\n")
    sys.stdout.flush()
    
    # perform backward time-stepping
    for n in range(total_steps, 0, -1):
        t = (n-1) * dt
        tau = T - t  # time to maturity
        
        # apply boundary conditions
        V[0, :] = 0.0
        V[-1, :] = S_max - K * np.exp(-r * tau)
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]
        
        # prepare linear system
        N_interior = (Ns-2) * (Nv-2)
        b = np.zeros(N_interior)
        rows = []
        cols = []
        data = []
        
        # function to map 2d indices to 1d
        def idx(i, j):
            return (j-1) * (Ns-2) + (i-1)
        
        # loop over interior grid nodes
        for j in range(1, Nv-1):
            for i in range(1, Ns-1):
                index = idx(i, j)
                S_val = S_grid[i]
                v_val = v_grid[j]
                
                # coefficients for s direction
                aS_plus  = 0.5 * S_val**2 * v_val / ds**2 + r * S_val / (2 * ds)
                aS_minus = 0.5 * S_val**2 * v_val / ds**2 - r * S_val / (2 * ds)
                aS_center = - S_val**2 * v_val / ds**2
                
                # coefficients for v direction
                aV_plus  = 0.5 * sigma**2 * v_val / dv**2 + kappa*(theta - v_val) / (2 * dv)
                aV_minus = 0.5 * sigma**2 * v_val / dv**2 - kappa*(theta - v_val) / (2 * dv)
                aV_center = - sigma**2 * v_val / dv**2
                
                # coefficient for mixed derivative
                a_mixed = rho * sigma * S_val * v_val / (4 * ds * dv)
                
                # central term
                central = aS_center + aV_center - r
                A_center = 1 - dt * central
                
                # fill in right-hand side vector
                b[index] = V[i, j]
                
                # fill matrix entries and boundary adjustments
                if i+1 <= Ns-2:
                    rows.append(index)
                    cols.append(idx(i+1, j))
                    data.append(- dt * aS_plus)
                else:
                    b[index] -= - dt * aS_plus * V[i+1, j]
                
                if i-1 >= 1:
                    rows.append(index)
                    cols.append(idx(i-1, j))
                    data.append(- dt * aS_minus)
                else:
                    b[index] -= - dt * aS_minus * V[i-1, j]
                
                if j+1 <= Nv-2:
                    rows.append(index)
                    cols.append(idx(i, j+1))
                    data.append(- dt * aV_plus)
                else:
                    b[index] -= - dt * aV_plus * V[i, j+1]
                
                if j-1 >= 1:
                    rows.append(index)
                    cols.append(idx(i, j-1))
                    data.append(- dt * aV_minus)
                else:
                    b[index] -= - dt * aV_minus * V[i, j-1]
                
                # mixed derivative entries
                if (i+1 <= Ns-2) and (j+1 <= Nv-2):
                    rows.append(index)
                    cols.append(idx(i+1, j+1))
                    data.append(- dt * a_mixed)
                else:
                    b[index] -= - dt * a_mixed * V[i+1, j+1]
                
                if (i+1 <= Ns-2) and (j-1 >= 1):
                    rows.append(index)
                    cols.append(idx(i+1, j-1))
                    data.append(dt * a_mixed)
                else:
                    b[index] -= dt * a_mixed * V[i+1, j-1]
                
                if (i-1 >= 1) and (j+1 <= Nv-2):
                    rows.append(index)
                    cols.append(idx(i-1, j+1))
                    data.append(dt * a_mixed)
                else:
                    b[index] -= dt * a_mixed * V[i-1, j+1]
                
                if (i-1 >= 1) and (j-1 >= 1):
                    rows.append(index)
                    cols.append(idx(i-1, j-1))
                    data.append(- dt * a_mixed)
                else:
                    b[index] -= - dt * a_mixed * V[i-1, j-1]
                
                # diagonal entry
                rows.append(index)
                cols.append(index)
                data.append(A_center)
        
        # assemble sparse matrix and solve
        A = sp.coo_matrix((data, (rows, cols)), shape=(N_interior, N_interior)).tocsr()
        x = spla.spsolve(A, b)
        
        # update solution grid
        for j in range(1, Nv-1):
            for i in range(1, Ns-1):
                V[i, j] = x[idx(i, j)]
        
        if (total_steps - n) % print_interval == 0:
            sys.stdout.write(f"Processed {total_steps - n} out of {total_steps} time steps\n")
            sys.stdout.flush()
    
    sys.stdout.write("Time-stepping complete\n")
    sys.stdout.flush()
    return S_grid, v_grid, V

# set up model parameters
S_max = 200.0
v_max = 0.5
Ns    = 50
Nv    = 30
Nt    = 100
print_interval = 10

K = 100
T = 1.0
r = 0.05
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7

# solve the pde
S_grid, v_grid, V = heston_implicit_fd(r, kappa, theta, sigma, rho, T, K, S_max, v_max, Ns, Nv, Nt, print_interval)

# create meshgrid for 3d plotting
S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')

# plot the option price surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_mesh, v_mesh, V, cmap='viridis', edgecolor='none')
ax.set_xlabel('Asset Price S')
ax.set_ylabel('Variance v')
ax.set_zlabel('Option Price')
ax.set_title('European Call Option Price under Heston Model')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# interpolate price at specific point
interp_func = RectBivariateSpline(S_grid, v_grid, V)
interp_price = interp_func(100.0, 0.04)[0, 0]
print("More accurate interpolated implicit FD option price at S0=100 and v0=0.04:", interp_price)
