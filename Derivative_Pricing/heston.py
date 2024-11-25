import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from MonteCarlo import MonteCarloMixin
from mixins import OptionMixin
from decorators import timer
import logging
logger = logging.getLogger(__name__)

ENABLE_TIMER = True

class HestonModel(MonteCarloMixin, OptionMixin):
    """
    np.float is deprecated for default float. just use that for now
    """
    def __init__(self, S0, strike, time_to_maturity, rate, vov, rho, v0, kappa, theta, option_type, nb_steps=10000,
                 nb_iters=10000):
        self.S0 = S0
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.rate = rate
        self.vov = vov
        self.rho = rho
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.option_type = option_type
        self._validateOptionType()
        self.nb_steps = nb_steps
        self.nb_iters = nb_iters
        self.vol_random_index = 1
        self.stock_random_index = 0
        self.initDependencies()

    def initDependencies(self):
        self.cho_matrix = None
        self.vol = None
        self.random_num_arr = None
        self.stock = None
        self.dt = self.time_to_maturity / self.nb_steps  # Time step
        self.sq_dt = np.sqrt(self.dt)
        self.discount_factor = np.exp(-self.rate * self.dt)

    @timer(ENABLE_TIMER, logger)
    def choleskyMatrix(self):
        if self.cho_matrix is not None:
            return self.cho_matrix
        covariance_matrix = np.zeros((2, 2), dtype=float)
        covariance_matrix[0] = [1.0, self.rho]
        covariance_matrix[1] = [self.rho, 1.0]
        self.cho_matrix = np.linalg.cholesky(covariance_matrix)
        return self.cho_matrix

    @timer(ENABLE_TIMER, logger)
    def random_number_gen(self):
        if self.random_num_arr is not None:
            return self.random_num_arr
        self.random_num_arr = np.random.standard_normal((2, self.nb_steps + 1, self.nb_iters))
        return self.random_num_arr

    @timer(ENABLE_TIMER, logger)
    def volSimulation(self):
        if self.vol is not None:
            return self.vol
        self.vol = np.zeros((self.nb_steps + 1, self.nb_iters), dtype=float)
        rand = self.random_number_gen()
        cho_matrix = self.choleskyMatrix()
        self.vol[0] = self.v0
        for t in range(1, self.nb_steps + 1):
            ran = np.dot(cho_matrix, rand[:, t])
            self.vol[t] = np.maximum(0, self.vol[t - 1] +
                                     self.kappa * (self.theta - self.vol[t - 1]) * self.dt
                                     + self.vov * self.sq_dt * ran[self.vol_random_index] * np.sqrt(self.vol[t - 1])
                                     )
        return self.vol

    @timer(ENABLE_TIMER, logger)
    def stockSimulation(self):
        if self.stock is not None:
            return self.stock
        self.volSimulation()
        self.stock = np.zeros((self.nb_steps + 1, self.nb_iters), dtype=float)
        rand = self.random_number_gen()
        cho_matrix = self.choleskyMatrix()
        self.stock[0] = self.S0
        for t in range(1, self.nb_steps + 1):
            ran = np.dot(cho_matrix, rand[:, t])
            self.stock[t] = self.stock[t - 1] * np.exp(
                (self.rate - self.vol[t - 1] / 2) * self.dt
                + np.sqrt(self.vol[t]) * ran[self.stock_random_index] * self.sq_dt
            )

    @timer(ENABLE_TIMER, logger)
    def plot_paths(self, n):
        self.volSimulation()
        self.stockSimulation()
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(range(len(self.stock)), self.stock[:, :n])
        ax1.grid()
        ax1.set_title("Heston Price paths")
        ax1.set_ylabel("Price")
        ax1.set_xlabel("Timestep")

        ax2.plot(range(len(self.vol)), self.vol[:, :n])
        ax2.grid()
        ax2.set_title("Heston Volatility paths")
        ax2.set_ylabel("Volatility")
        ax2.set_xlabel("Timestep")
        plt.show()

    @timer(ENABLE_TIMER, logger)
    def price(self):
        self.stockSimulation()
        payoff = self._payoff(self.stock[-1, :], self.strike)
        return np.exp(-self.rate * self.time_to_maturity) * np.mean(payoff)


if __name__ == '__main__':
    v0 = 0.04
    kappa_v = 2
    sigma_v = 0.3
    theta_v = 0.04
    rho = -0.9
    strike = 90
    time_to_maturity = 1.0

    S0 = 100  # Current underlying asset price
    r = 0.05  # Risk-free rate
    M0 = 50  # Number of time steps in a year
    T = 1  # Number of years
    M = int(M0 * T)  # Total time steps
    Ite = 10000  # Number of simulations
    dt = T / M  # Length of time step

    heston = HestonModel(S0=S0, strike=strike, time_to_maturity=time_to_maturity, rate=r, vov=sigma_v, rho=rho, v0=v0,
                         kappa=kappa_v, theta=theta_v, option_type="call", nb_iters=Ite, nb_steps=M)
    # heston.plot_paths(Ite)
    # print(heston.choleskyMatrix())
    print(heston.price())
