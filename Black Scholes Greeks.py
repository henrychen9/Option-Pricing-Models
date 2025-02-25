# Black Scholes Greeks

import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:  # put option
        delta = norm.cdf(d1) - 1

    # Gamma (same for calls & puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (same for calls & puts, scaled by 100 to match standard convention)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  

    # Theta
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        theta = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta /= 365  # Convert to per-day decay

    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  
    else:  # put option
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }

# Example usage:
S = 100   # Stock price
K = 100   # Strike price
T = 1     # Time to expiration (in years)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

# Compute Greeks for a call option
greeks = black_scholes_greeks(S, K, T, r, sigma, option_type="call")
print(greeks)
