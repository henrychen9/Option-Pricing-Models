# Black Scholes calculator
# Henry Chen

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, t, r, sigma):
    "Computes the Black-Scholes price for a European call option."
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    C = S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    return C
S = 100    # Stock price
K = 100    # Strike price
T = 1.0    # Expiration time (1 year)
t = 0.0    # Current time
r = 0.05   # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)

price = black_scholes_call(S, K, T, t, r, sigma)
print("Black-Scholes Call Price:", price)