import numpy as np
from scipy.integrate import quad

def heston_charfun(u, S0, v0, T, r, kappa, theta, sigma, rho, j):
    # computes the characteristic function under the Heston model

    i = 1j
    x = np.log(S0)

    # for j=1, we use a shift
    if j == 1:
        u_shift = u - i
    else:
        u_shift = u

    d = np.sqrt((rho * sigma * i * u_shift - kappa)**2 + sigma**2 * (i*u_shift + u_shift**2))
    g = (kappa - rho*sigma*i*u_shift - d) / (kappa - rho*sigma*i*u_shift + d)
    
    # compute C and D as before
    C = r * i * u_shift * T + (kappa * theta / sigma**2) * ((kappa - rho*sigma*i*u_shift - d)*T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho*sigma*i*u_shift - d) / sigma**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    # normalization for j=1 only
    if j == 1:
        # subtract rT from C and ln(S0) in the final exponent
        C = C - r * T
        return np.exp(C + D * v0 + i * u_shift * x - np.log(S0))
    
    else:
        return np.exp(C + D * v0 + i * u_shift * x)

def integrand(u, S0, v0, T, r, kappa, theta, sigma, rho, K, j):
    # Computes the integrand for the probability P_j
    i = 1j
    phi = heston_charfun(u, S0, v0, T, r, kappa, theta, sigma, rho, j)
    return np.real(np.exp(-i*u*np.log(K)) * phi / (i*u))

def heston_price_fourier(S0, v0, T, r, kappa, theta, sigma, rho, K, lower_limit=1e-8, upper_limit=50):
    # prices a European call option under the Heston model using Fourier integration
    P1 = 0.5 + (1/np.pi) * quad(lambda u: integrand(u, S0, v0, T, r, kappa, theta, sigma, rho, K, 1),
                                lower_limit, upper_limit, limit=500)[0]
    P2 = 0.5 + (1/np.pi) * quad(lambda u: integrand(u, S0, v0, T, r, kappa, theta, sigma, rho, K, 2),
                                lower_limit, upper_limit, limit=500)[0]
    
    return S0 * P1 - K * np.exp(-r*T) * P2


S0    = 100.0  # initial stock price
v0    = 0.04   # initial variance
K     = 100.0  # strike price
T     = 1.0    # time to maturity (in years)
r     = 0.05   # risk-free interest rate
kappa = 2.0    # mean reversion speed
theta = 0.04   # long-run variance
sigma = 0.3    # volatility of variance (vol of vol)
rho   = -0.7   # correlation between stock and variance

price = heston_price_fourier(S0, v0, T, r, kappa, theta, sigma, rho, K, lower_limit=1e-8, upper_limit=50)

print("European Call Price under Heston model:", price)
