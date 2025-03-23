import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class MonteCarloSimulator:
    # initialize the simulator with parameters
    def __init__(self, S0, T, n, r, sigma, M, is_store_paths=True):
        self.S0 = S0        # initial stock price
        self.T = T          # time to maturity
        self.n = n          # number of time steps
        self.r = r          # risk-free interest rate
        self.sigma = sigma  # volatility
        self.M = M          # number of simulations
        self.dt = T / n     # time step size
        self.is_store_paths = is_store_paths  # flag for storing full paths
        self.paths = None
        self.final_prices = None

    # simulate stock price paths using gbm
    def simulate(self):
        Z = np.random.randn(self.M, self.n)  # standard normal shocks

        if self.is_store_paths:
            S = np.zeros((self.M, self.n + 1))
            S[:, 0] = self.S0  # set initial prices

            for t in range(self.n):
                drift = (self.r - 0.5 * self.sigma**2) * self.dt
                diffusion = self.sigma * np.sqrt(self.dt) * Z[:, t]
                S[:, t+1] = S[:, t] * np.exp(drift + diffusion)  # update path

            self.paths = S
            self.final_prices = S[:, -1]  # store terminal values
        else:
            S_t = np.full(self.M, float(self.S0))  # vector of initial prices
            for t in range(self.n):
                drift = (self.r - 0.5 * self.sigma**2) * self.dt
                diffusion = self.sigma * np.sqrt(self.dt) * Z[:, t]
                S_t *= np.exp(drift + diffusion)
            self.final_prices = S_t

    # price american option using least-squares monte carlo
    def price_american_option(self, K, option_type="call"):
        if self.paths is None:
            raise ValueError("run simulate() before pricing an option")

        discount_factor = np.exp(-self.r * self.dt)
        payoffs = np.zeros_like(self.paths)  # option value at each time

        # set final payoffs at maturity
        if option_type == "call":
            payoffs[:, -1] = np.maximum(self.paths[:, -1] - K, 0)
        elif option_type == "put":
            payoffs[:, -1] = np.maximum(K - self.paths[:, -1], 0)
        else:
            raise ValueError("invalid option type")

        for t in range(self.n - 1, 0, -1):
            in_money = payoffs[:, t] > 0  # eligible for early exercise

            if np.any(in_money):
                X = self.paths[in_money, t].reshape(-1, 1)  # spot prices
                Y = payoffs[in_money, t+1] * discount_factor  # discounted future payoff

                model = LinearRegression()
                model.fit(X, Y)  # fit continuation value
                continuation_value = model.predict(X)

                if option_type == "call":
                    immediate_exercise = np.maximum(self.paths[in_money, t] - K, 0)
                else:
                    immediate_exercise = np.maximum(K - self.paths[in_money, t], 0)

                exercise = immediate_exercise > continuation_value
                payoffs[in_money, t] = np.where(exercise, immediate_exercise,
                                                payoffs[in_money, t+1] * discount_factor)

        option_price = np.mean(payoffs[:, 1] * discount_factor)
        return option_price

    # compare simulated expected value to theoretical
    def expected_value_comparison(self):
        if self.final_prices is None:
            raise ValueError("run simulate() before computing expectations")

        empirical_mean = np.mean(self.final_prices)
        theoretical_mean = self.S0 * np.exp(self.r * self.T)  # E[S_T] under gbm

        print(f"simulated E[S_T]: {empirical_mean:.4f}")
        print(f"theoretical E[S_T]: {theoretical_mean:.4f}")
        print(f"percent error: {100 * abs(empirical_mean - theoretical_mean) / theoretical_mean:.4f}%")

    # plot a subset of simulated stock price paths
    def plot_paths(self, num_paths=10):
        if not self.is_store_paths or self.paths is None:
            raise ValueError("run simulate() with is_store_paths=True before plotting")

        plt.figure(figsize=(10, 5))
        time_grid = np.linspace(0, self.T, self.n + 1)  # x-axis timeline

        for i in range(min(num_paths, self.M)):
            plt.plot(time_grid, self.paths[i, :], lw=1)  # draw path

        plt.xlabel("time (years)")
        plt.ylabel("stock price")
        plt.title("monte carlo simulated price paths")
        plt.grid()
        plt.show()


simulator = MonteCarloSimulator(S0=100, T=1, n=252, r=0.05, sigma=0.2, M=10000, is_store_paths=True)
simulator.simulate()
simulator.plot_paths()
simulator.expected_value_comparison()
call_price = simulator.price_american_option(K=100, option_type="call")
put_price = simulator.price_american_option(K=100, option_type="put")
print(f"american call option price: {call_price:.4f}")
print(f"american put option price: {put_price:.4f}")
