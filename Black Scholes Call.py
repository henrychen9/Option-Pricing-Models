# Black Scholes Call

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, t, r, sigma):
    "Computes the Black-Scholes price for a European call option."
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    C = S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    return C


