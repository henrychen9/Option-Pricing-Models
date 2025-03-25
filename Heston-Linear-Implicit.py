import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline

def heston_implicit_fd(r, kappa, theta, sigma, rho, T, K, S_max, v_max, Ns, Nv, Nt, print_interval=10):
    # fully implicit FD for pricing European call (Heston)
    
    # define grid steps
    ds = S_max / (Ns - 1)
    dv = v_max / (Nv - 1)
    dt_fixed = T / (Nt - 1)  # initial guess if needed, but we'll adapt dt
    
    # S and v grid
    S_grid = np.linspace(0, S_max, Ns)
    v_grid = np.linspace(0, v_max, Nv)
    
    # terminal condition: payoff at maturity
    V = np.zeros((Ns, Nv))
    for i in range(Ns):
        payoff = max(S_grid[i] - K, 0)
        V[i, :] = payoff
    
    tol = 1e-4
    dt_min = 1e-6
    dt_max = dt_fixed

    # initialize time step
    current_dt = dt_fixed
    t = T
    
    # one implicit step backward in time
    def single_time_step(V_in, dt, t):
        # t is the current time, dt is the step backward so that new time is t_new = t - dt
        V_new = V_in.copy()
        t_new = t - dt
        tau = T - t_new  # time to maturity
        
        # apply boundary conditions
        V_new[0, :] = 0.0
        V_new[-1, :] = S_max - K * np.exp(-r * tau)
        V_new[:, 0] = V_new[:, 1]
        V_new[:, -1] = V_new[:, -2]
        
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
                
                # RHS = previous time step
                b[index] = V_in[i, j]
                
                # fill matrix entries and boundary adjustments
                if i+1 <= Ns-2:
                    rows.append(index)
                    cols.append(idx(i+1, j))
                    data.append(- dt * aS_plus)
                else:
                    b[index] -= - dt * aS_plus * V_in[i+1, j]
                
                if i-1 >= 1:
                    rows.append(index)
                    cols.append(idx(i-1, j))
                    data.append(- dt * aS_minus)
                else:
                    b[index] -= - dt * aS_minus * V_in[i-1, j]
                
                if j+1 <= Nv-2:
                    rows.append(index)
                    cols.append(idx(i, j+1))
                    data.append(- dt * aV_plus)
                else:
                    b[index] -= - dt * aV_plus * V_in[i, j+1]
                
                if j-1 >= 1:
                    rows.append(index)
                    cols.append(idx(i, j-1))
                    data.append(- dt * aV_minus)
                else:
                    b[index] -= - dt * aV_minus * V_in[i, j-1]
                
                # mixed derivative entries
                if (i+1 <= Ns-2) and (j+1 <= Nv-2):
                    rows.append(index)
                    cols.append(idx(i+1, j+1))
                    data.append(- dt * a_mixed)
                else:
                    b[index] -= - dt * a_mixed * V_in[i+1, j+1]
                
                if (i+1 <= Ns-2) and (j-1 >= 1):
                    rows.append(index)
                    cols.append(idx(i+1, j-1))
                    data.append(dt * a_mixed)
                else:
                    b[index] -= dt * a_mixed * V_in[i+1, j-1]
                
                if (i-1 >= 1) and (j+1 <= Nv-2):
                    rows.append(index)
                    cols.append(idx(i-1, j+1))
                    data.append(dt * a_mixed)
                else:
                    b[index] -= dt * a_mixed * V_in[i-1, j+1]
                
                if (i-1 >= 1) and (j-1 >= 1):
                    rows.append(index)
                    cols.append(idx(i-1, j-1))
                    data.append(- dt * a_mixed)
                else:
                    b[index] -= - dt * a_mixed * V_in[i-1, j-1]
                
                # diagonal entry
                rows.append(index)
                cols.append(index)
                data.append(A_center)
        
        # build sparse matrix and solve
        A = sp.coo_matrix((data, (rows, cols)), shape=(N_interior, N_interior)).tocsr()
        x = spla.spsolve(A, b)
        
        # update solution grid
        for j in range(1, Nv-1):
            for i in range(1, Ns-1):
                V_new[i, j] = x[idx(i, j)]
        return V_new
    
    step_count = 0
    accepted_steps = 0
    while t > 0:
        current_dt = min(current_dt, t)
        
        # compute one full step and two half steps
        V_full = single_time_step(V, current_dt, t)
        V_half1 = single_time_step(V, current_dt/2, t)
        V_half  = single_time_step(V_half1, current_dt/2, t - current_dt/2)

        error = np.linalg.norm(V_full - V_half) / (np.linalg.norm(V_half) + 1e-10)
        
        # reject the step and reduce current_dt if error is too large
        if error > tol:
            current_dt = max(current_dt/2, dt_min)
            sys.stdout.write(f"Step {step_count}: error {error:.2e} too high, reducing dt to {current_dt:.2e}\n")
            sys.stdout.flush()
        else:
            # accept the half-step solution as it is more accurate
            V = V_half.copy()
            t = t - current_dt
            accepted_steps += 1
            step_count += 1

            # increase current_dt if error is much lower than tolerance
            if error < tol/2 and current_dt < dt_max:
                current_dt = min(current_dt * 2, dt_max)

            if accepted_steps % print_interval == 0:
                sys.stdout.write(f"Processed {accepted_steps} time steps; current t = {t:.4f}\n")
                sys.stdout.flush()
    
    return S_grid, v_grid, V

# params
S_max = 200.0       # maximum asset price
v_max = 0.5         # maximum variance 
Ns    = 50          # number of asset price grid points
Nv    = 30          # number of variance grid points
Nt    = 100         # number of time steps
print_interval = 10 # how often to print iteration info

K = 100             # strike price
T = 1.0             # time to maturity
r = 0.05            # risk-free interest rate
kappa = 2.0         # mean reversion rate of variance
theta = 0.04        # long-term mean of variance
sigma = 0.3         # volatility of volatility
rho = -0.7          # correlation between S and v

# solve the PDE
S_grid, v_grid, V = heston_implicit_fd(r, kappa, theta, sigma, rho, T, K, S_max, v_max, Ns, Nv, Nt, print_interval)

# create meshgrid for 3D plotting
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
print("Interpolated implicit FD option price at S0=100 and v0=0.04:", interp_price)
