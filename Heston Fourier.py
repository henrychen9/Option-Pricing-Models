import numpy as np
from scipy.fft import fft


S0, K, v0, r, T = 100, 100, 0.04, 0.05, 1
kappa, theta, sigma, rho = 2.0, 0.04, 0.3, -0.5
N = 64
L = 10


def char_function(u, t, S0, v0, r, kappa, theta, sigma, rho):
    """ Heston model characteristic function """
    xi = kappa - 1j * rho * sigma * u
    d = np.sqrt(xi**2 + (sigma**2) * (u**2 + 1j * u))
    g = (xi - d) / (xi + d)
    
    A = (1j * u * (np.log(S0) + (r - 0.5 * v0) * t) +
         (kappa * theta / sigma**2) * ((xi - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))))
    
    B = ((xi - d) / sigma**2) * ((1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t)))
    
    return np.exp(A + B * v0)

 
x0 = np.log(S0 / K)

# Truncation range for Fourier COS
a, b = -L * np.sqrt(T), L * np.sqrt(T)

# Define Fourier-Cosine coefficients
k = np.arange(N)
u = k * np.pi / (b - a)

# Characteristic function of the log-price
cf_values = char_function(u, T, S0, v0, r, kappa, theta, sigma, rho)

# Fourier coefficients for payoff function
chi_k = (np.sin(u * (b - a)) / u) * np.exp(-r * T)  
chi_k[0] = (b - a) * np.exp(-r * T)  # Correct for k=0 to avoid division by zero

# Compute price using COS formula
price = K * np.real(np.dot(cf_values, chi_k)) * 2 / (b - a)


print(f"Heston Fourier-COS Price: {price:.4f}")