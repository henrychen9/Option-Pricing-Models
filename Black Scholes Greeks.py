# Black Scholes Greeks

import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (scaled by 100)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  

    # Theta
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta /= 365

    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }
