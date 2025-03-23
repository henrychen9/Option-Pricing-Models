import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline

def psor_solver(A, b, payoff, x, omega_init, tol, max_iter):
    # solve LCP using projected SOR (x >= payoff)
    A_dense = A.toarray()
    N = len(x)
    omega = omega_init
    prev_error = None

    for it in range(max_iter):
        x_old = x.copy()

        # perform a PSOR iteration
        for i in range(N):
            sum_val = 0.0
            for j in range(N):
                if j < i:
                    sum_val += A_dense[i, j] * x[j]
                elif j > i:
                    sum_val += A_dense[i, j] * x_old[j]
            x_i_new = (b[i] - sum_val) / A_dense[i, i]

            # relaxation and projection
            x[i] = max(payoff[i], (1 - omega) * x_old[i] + omega * x_i_new)
        
        # compute error
        error = np.linalg.norm(x - x_old, ord=np.inf)
        
        # update omega adaptively
        if prev_error is not None:
            ratio = error / prev_error
            if ratio > 0.95:
                omega = max(1.0, omega * 0.9)
            elif ratio < 0.80:
                omega = min(1.95, omega * 1.1)
        
        if error < tol:
            break
        
        prev_error = error
    
    return x


def build_operator_interior(S_grid, v_grid, ds, dv, dt, r, kappa, theta, sigma, rho, Ns, Nv):
    # build sparse FD matrix for Heston PDE (interior only)
    N_interior = (Ns - 2) * (Nv - 2)
    rows = []
    cols = []
    data = []
    
    def idx(i, j):
        return (j - 1) * (Ns - 2) + (i - 1)
    
    for j in range(1, Nv - 1):
        for i in range(1, Ns - 1):
            index = idx(i, j)
            S_val = S_grid[i]
            v_val = v_grid[j]

            # S-direction terms
            aS_plus  = 0.5 * S_val**2 * v_val / ds**2 + r * S_val / (2 * ds)
            aS_minus = 0.5 * S_val**2 * v_val / ds**2 - r * S_val / (2 * ds)
            aS_center = - S_val**2 * v_val / ds**2

            # v-direction terms
            aV_plus  = 0.5 * sigma**2 * v_val / dv**2 + kappa * (theta - v_val) / (2 * dv)
            aV_minus = 0.5 * sigma**2 * v_val / dv**2 - kappa * (theta - v_val) / (2 * dv)
            aV_center = - sigma**2 * v_val / dv**2

            # mixed derivative
            a_mixed = rho * sigma * S_val * v_val / (4 * ds * dv)

            # center
            central = aS_center + aV_center - r
            A_center = 1 - dt * central

            # neighbors
            if i + 1 <= Ns - 2:
                rows.append(index)
                cols.append(idx(i + 1, j))
                data.append(- dt * aS_plus)

            if i - 1 >= 1:
                rows.append(index)
                cols.append(idx(i - 1, j))
                data.append(- dt * aS_minus)

            if j + 1 <= Nv - 2:
                rows.append(index)
                cols.append(idx(i, j + 1))
                data.append(- dt * aV_plus)

            if j - 1 >= 1:
                rows.append(index)
                cols.append(idx(i, j - 1))
                data.append(- dt * aV_minus)

            # mixed terms
            if (i + 1 <= Ns - 2) and (j + 1 <= Nv - 2):
                rows.append(index)
                cols.append(idx(i + 1, j + 1))
                data.append(- dt * a_mixed)

            if (i + 1 <= Ns - 2) and (j - 1 >= 1):
                rows.append(index)
                cols.append(idx(i + 1, j - 1))
                data.append(dt * a_mixed)

            if (i - 1 >= 1) and (j + 1 <= Nv - 2):
                rows.append(index)
                cols.append(idx(i - 1, j + 1))
                data.append(dt * a_mixed)

            if (i - 1 >= 1) and (j - 1 >= 1):
                rows.append(index)
                cols.append(idx(i - 1, j - 1))
                data.append(- dt * a_mixed)

            # diag
            rows.append(index)
            cols.append(index)
            data.append(A_center)
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(N_interior, N_interior)).tocsr()
    return A

def restrict(fine):
    # restrict fine to coarse (4-point avg)
    coarse = 0.25 * (fine[0::2, 0::2] + fine[1::2, 0::2] + fine[0::2, 1::2] + fine[1::2, 1::2])
    return coarse

def prolong(coarse):
    # prolong coarse to fine (2x2 duplication)
    return np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)


def mg_psor_solver(A, b, payoff, x, omega, tol, max_iter, num_pre, num_post,
                   S_grid, v_grid, ds, dv, dt, r, kappa, theta, sigma, rho, Ns, Nv):
    # handle multigrid PSOR with the existing PSOR solver
    x = psor_solver(A, b, payoff, x, omega, tol, num_pre)
    r_vec = b - A.dot(x)
    
    Nx = Ns - 2
    Ny = Nv - 2
    r_2d = np.reshape(r_vec, (Nx, Ny))
    x_2d = np.reshape(x, (Nx, Ny))
    payoff_2d = np.reshape(payoff, (Nx, Ny))
    
    r_coarse = restrict(r_2d)
    x_coarse = restrict(x_2d)
    payoff_coarse = restrict(payoff_2d)
    
    S_interior = S_grid[1:-1]
    v_interior = v_grid[1:-1]
    S_coarse = S_interior[::2]
    v_coarse = v_interior[::2]
    Ns_coarse = len(S_coarse) + 2
    Nv_coarse = len(v_coarse) + 2
    ds_coarse = ds * 2
    dv_coarse = dv * 2
    dt_coarse = dt
    S_coarse_full = np.linspace(S_grid[0], S_grid[-1], Ns_coarse)
    v_coarse_full = np.linspace(v_grid[0], v_grid[-1], Nv_coarse)
    A_coarse = build_operator_interior(S_coarse_full, v_coarse_full, ds_coarse, dv_coarse, dt_coarse,
                                       r, kappa, theta, sigma, rho, Ns_coarse, Nv_coarse)
    
    e_coarse = np.zeros(A_coarse.shape[0])
    e_coarse = psor_solver(A_coarse, r_coarse.flatten(), 
                           np.zeros_like(r_coarse.flatten()), e_coarse, omega, tol, max_iter)
    
    e_coarse_2d = np.reshape(e_coarse, r_coarse.shape)
    e_fine = prolong(e_coarse_2d)
    e_fine = e_fine[:Nx, :Ny]
    
    x_2d += e_fine
    x = x_2d.flatten()
    
    x = psor_solver(A, b, payoff, x, omega, tol, num_post)
    return x

def heston_american_fd(r, kappa, theta, sigma, rho, T, K, S_max, v_max, Ns, Nv, Nt,
                         omega=1.6, tol=1e-3, max_iter=500, use_multigrid=True, print_interval=10):
    # solve American call under Heston using FD + PSOR + MG
    ds = S_max / (Ns - 1)
    dv = v_max / (Nv - 1)
    dt = T / (Nt - 1)
    
    S_grid = np.linspace(0, S_max, Ns)
    v_grid = np.linspace(0, v_max, Nv)
    
    # terminal payoff
    V = np.zeros((Ns, Nv))
    for i in range(Ns):
        V[i, :] = max(S_grid[i] - K, 0)
    
    total_steps = Nt - 1
    
    for n in range(total_steps, 0, -1):
        t = (n - 1) * dt
        tau = T - t
        
        # boundary conditions
        V[0, :] = 0.0
        V[-1, :] = S_max - K
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]
        
        A = build_operator_interior(S_grid, v_grid, ds, dv, dt, r, kappa, theta, sigma, rho, Ns, Nv)
        N_interior = (Ns - 2) * (Nv - 2)

        # build vectors for b (rhs), payoff, and initial guess
        def idx(i, j): return (j - 1) * (Ns - 2) + (i - 1)

        b = np.zeros(N_interior)
        payoff_vec = np.zeros(N_interior)
        x_init = np.zeros(N_interior)

        for j in range(1, Nv - 1):
            for i in range(1, Ns - 1):
                b[idx(i, j)] = V[i, j]
                payoff_vec[idx(i, j)] = max(S_grid[i] - K, 0)
                x_init[idx(i, j)] = V[i, j]
        
        if use_multigrid:
            x_sol = mg_psor_solver(A, b, payoff_vec, x_init, omega, tol, max_iter,
                                    num_pre=5, num_post=5,
                                    S_grid=S_grid, v_grid=v_grid, ds=ds, dv=dv, dt=dt,
                                    r=r, kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                                    Ns=Ns, Nv=Nv)
        else:
            x_sol = psor_solver(A, b, payoff_vec, x_init, omega, tol, max_iter)
        
        for j in range(1, Nv - 1):
            for i in range(1, Ns - 1):
                V[i, j] = x_sol[idx(i, j)]
        
        if (total_steps - n) % print_interval == 0:
            sys.stdout.write(f"Processed {total_steps - n} out of {total_steps} time steps\n")
            sys.stdout.flush()
    return S_grid, v_grid, V

# params
S_max = 200.0
v_max = 0.5
Ns    = 20
Nv    = 20
Nt    = 100

K = 100
T = 1.0
r = 0.05
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7

# solve PDE
S_grid, v_grid, V_american = heston_american_fd(r, kappa, theta, sigma, rho, T, K,
                                                S_max, v_max, Ns, Nv, Nt,
                                                use_multigrid=True)

# plot surface
S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_mesh, v_mesh, V_american, cmap='viridis', edgecolor='none')
ax.set_xlabel('Asset Price S')
ax.set_ylabel('Variance v')
ax.set_zlabel('Option Price')
ax.set_title('American Call Option Price under the Heston Model')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# interpolation at S0=100, v0=0.04
interp_func = RectBivariateSpline(S_grid, v_grid, V_american)
interp_price = interp_func(100.0, 0.04)[0, 0]
print("Interpolated American option price at S0=100 and v0=0.04:", interp_price)
