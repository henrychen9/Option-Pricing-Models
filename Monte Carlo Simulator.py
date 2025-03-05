import numpy as np
import matplotlib.pyplot as plt

class MonteCarloSimulator:
    def __init__(self, S0, T, n, r, sigma, M, is_store_paths=True):
        """
        Initialize the simulator with the given parameters.

        parameters:
        - S0: initial stock price
        - T: time to maturity (in years)
        - n: number of time steps
        - r: risk-free rate
        - sigma: volatility
        - M: number of simulations
        - is_store_paths: whether to store full paths (default: True)
        """
        self.S0 = S0  # initial stock price
        self.T = T  # time to maturity
        self.n = n  # number of time steps
        self.r = r  # risk-free interest rate
        self.sigma = sigma  # volatility
        self.M = M  # number of simulations
        self.dt = T / n  # time step size
        self.is_store_paths = is_store_paths  # flag for storing full paths
        self.paths = None  # will hold the simulated price paths if needed
        self.final_prices = None  # will hold the final prices of the simulations

    def simulate(self):
        """Simulate asset price paths using the GBM model."""
        # generate random values for the standard normal process
        Z = np.random.randn(self.M, self.n) 

        if self.is_store_paths:
            # if storing paths, create an array to hold the price paths for all simulations
            S = np.zeros((self.M, self.n + 1))
            S[:, 0] = self.S0  # set initial stock prices

            # loop through each time step and update stock prices
            for t in range(self.n):
                S[:, t+1] = S[:, t] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt +
                                             self.sigma * np.sqrt(self.dt) * Z[:, t])

            self.paths = S  # store all paths
            self.final_prices = S[:, -1]  # store the final price for each simulation
        else:
            # if not storing paths, just calculate the final price for each simulation
            S_t = np.full(self.M, float(self.S0))
            for t in range(self.n):
                S_t *= np.exp((self.r - 0.5 * self.sigma**2) * self.dt +
                              self.sigma * np.sqrt(self.dt) * Z[:, t])

            self.final_prices = S_t  # store the final prices

    def price_american_option(self, K, option_type="call"):
        """
        Price an American option using Least-Squares Monte Carlo (LSMC).

        parameters:
        - K: strike price
        - option_type: 'call' or 'put' (default: 'call')

        returns:
        - Estimated American option price.
        """
        if self.paths is None:
            raise ValueError("Run simulate() before pricing an option.")

        discount_factor = np.exp(-self.r * self.dt)
        payoffs = np.zeros_like(self.paths)

        # compute option payoffs at maturity
        if option_type == "call":
            payoffs[:, -1] = np.maximum(self.paths[:, -1] - K, 0)
        elif option_type == "put":
            payoffs[:, -1] = np.maximum(K - self.paths[:, -1], 0)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        # 2ork backward in time (LSMC)
        for t in range(self.n - 1, 0, -1):
            in_money = payoffs[:, t] > 0  # select paths where exercising is possible

            if np.any(in_money):
                X = self.paths[in_money, t].reshape(-1, 1)  # stock prices at t
                Y = payoffs[in_money, t+1] * discount_factor  # discounted future payoffs

                # fit regression model to estimate continuation value
                model = LinearRegression()
                model.fit(X, Y)
                continuation_value = model.predict(X)

                # determine early exercise decision
                immediate_exercise = np.maximum(self.paths[in_money, t] - K, 0) if option_type == "call" else np.maximum(K - self.paths[in_money, t], 0)
                exercise = immediate_exercise > continuation_value

                # update payoffs for exercised paths
                payoffs[in_money, t] = np.where(exercise, immediate_exercise, payoffs[in_money, t+1] * discount_factor)

        # compute option price as the mean of discounted payoffs
        option_price = np.mean(payoffs[:, 1] * discount_factor)
        return option_price
    
    def expected_value_comparison(self):
        """Compare simulated E[S_T] with theoretical E[S_T]."""
        if self.final_prices is None:
            raise ValueError("Run simulate() before computing expectations.")  # ensure simulation has run

        # calculate simulated and theoretical expected values
        empirical_mean = np.mean(self.final_prices)
        theoretical_mean = self.S0 * np.exp(self.r * self.T)

        # print the comparison between simulated and theoretical values
        print(f"Simulated E[S_T]: {empirical_mean:.4f}")
        print(f"Theoretical E[S_T]: {theoretical_mean:.4f}")
        print(f"Percent Error: {100 * abs(empirical_mean - theoretical_mean) / theoretical_mean:.4f}%")

    def plot_paths(self, num_paths=10):
        """Plot a subset of the simulated price paths."""
        if not self.is_store_paths or self.paths is None:
            raise ValueError("Run simulate() with is_store_paths=True before plotting.")  # ensure paths are stored

        plt.figure(figsize=(10, 5))
        time_grid = np.linspace(0, self.T, self.n + 1)
        for i in range(min(num_paths, self.M)):  # plot the requested number of paths
            plt.plot(time_grid, self.paths[i, :], lw=1)

        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")
        plt.title("Monte Carlo Simulated Price Paths")
        plt.grid()
        plt.show()

# example usage
if __name__ == "__main__":
    # create simulator instance with sample parameters
    simulator = MonteCarloSimulator(S0=100, T=1, n=252, r=0.05, sigma=0.2, M=10000, is_store_paths=True)
    simulator.simulate()  # run the simulation
    simulator.plot_paths()  # plot a few of the simulated paths
    simulator.expected_value_comparison()  # compare simulated vs theoretical expected value
    american_call_price = simulator.price_american_option(K=100, option_type="call")
    american_put_price = simulator.price_american_option(K=100, option_type="put")
    print(f"American Call Option Price: {american_call_price:.4f}")
    print(f"American Put Option Price: {american_put_price:.4f}")