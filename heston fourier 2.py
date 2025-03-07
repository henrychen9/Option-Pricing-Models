import numpy as np
import scipy.integrate as integrate

def heston_characteristic_function(u, S0, v0, kappa, theta, sigma, rho, r, tau):
    d = np.sqrt((rho * sigma * 1j * u - kappa)**2 + sigma**2 * (1j * u + u**2))
    g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)
    C = r * 1j * u * tau + (kappa * theta / sigma**2) * ((kappa - rho * sigma * 1j * u - d) * tau - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    D = ((kappa - rho * sigma * 1j * u - d) / sigma**2) * ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
    return np.exp(C + D * v0 + 1j * u * np.log(S0))

def heston_option_price(S0, K, v0, kappa, theta, sigma, rho, r, tau, option_type='call'):
    def integrand(u):
        return np.real(np.exp(-1j * u * np.log(K)) * heston_characteristic_function(u - 0.5j, S0, v0, kappa, theta, sigma, rho, r, tau)) / (u**2 + 0.25)
    
    integral, _ = integrate.quad(integrand, 0, np.inf)
    option_price = (S0 - K * np.exp(-r * tau) * integral / np.pi)
    
    if option_type == 'put':
        option_price = option_price - S0 + K * np.exp(-r * tau)
    
    return option_price

# Input parameters
S0 = 100
K = 100
v0 = 0.04
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.5
r = 0.05
tau = 1.0

# Compute call price
call_price = heston_option_price(S0, K, v0, kappa, theta, sigma, rho, r, tau, option_type='call')
print(f"Call Price: {call_price}")