import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import pandas as pd
from sklearn.linear_model import LinearRegression
from mixins import OptionMixin
from MonteCarlo import MonteCarloMixin



class MertonModel(OptionMixin, MonteCarloMixin):
    """
    TODO 1. see what happens when we change the poisson distribution. mu and lambda. look at the stock returns log distribution
    TODO 2. the paths visualization
    TODO 3. compare with closed form, heston. prices as well as the graphs.
    """
    def __init__(self,
                 strike, time, S0, rate, option_type, sigma, expiry,
                 lambda_, mu, delta, nb_steps, nb_iters):
        self.lambda_ = lambda_
        self.mu = mu
        self.delta = delta
        self.rate = rate
        self.sigma = sigma
        self.expiry = expiry
        self.S0 = S0
        self.strike = strike
        self.nb_steps = nb_steps
        self.nb_iters = nb_iters
        self.time = time
        self.option_type = option_type
        self.initDependencies()
        
    def initDependencies(self):
        self.time_to_maturity = self.expiry - self.time
        self.dt = self.time_to_maturity / self.nb_steps
        self.z1 = None
        self.z2 = None
        self.y = None
        self.rj = None
        self.stock = None
        

    def calculate_rj(self):
        self.rj = self.lambda_ * (np.exp(self.mu + 0.5 * self.delta**2) - 1)

    def genRandomNumbers(self):
        if self.z1 is not None:
            return 
        self.z1 = np.random.standard_normal((self.nb_steps + 1, self.nb_iters))
        self.z2 = np.random.standard_normal((self.nb_steps + 1, self.nb_iters))
        self.y = np.random.poisson(self.lambda_ * self.dt, (self.nb_steps + 1, self.nb_iters))
    
    def simulate(self):
        if self.stock is not None:
            return self.stock
        self.stock = np.zeros((self.nb_steps + 1, self.nb_iters), dtype=float)
        self.genRandomNumbers()
        self.calculate_rj()
        self.stock[0] = self.S0
        for t in range(1, self.nb_steps + 1):
            self.stock[t] = self.stock[t - 1] * (
                    np.exp((self.rate - self.rj - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * self.z1[t])
                    + (np.exp(self.mu + self.delta * self.z2[t]) - 1) * self.y[t]
            )
            self.stock[t] = np.maximum(
                self.stock[t], 0.00001
            ) 
        return self.stock

    def price(self):
        self.simulate()
        payoff = self._payoff(self.stock[-1, :], self.strike)
        return np.exp(-self.rate * self.time_to_maturity) * np.mean(payoff)


if __name__ == "__main__":
    lamb = 0.75  # Lambda of the model
    mu = -0.6  # Mu
    delta = 0.25  # Delta

    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0  # Maturity/time period (in years)
    S0 = 100  # Current Stock Price

    Ite = 10000  # Number of simulations (paths)
    M = 50  # Number of steps
    strike = 90
    t = 0

    merton = MertonModel(strike=strike, time=t, S0=S0, rate=r, option_type="call", sigma=sigma, expiry=T,
                         lambda_=lamb, mu=mu, delta=delta, nb_steps=M, nb_iters=Ite)
    print(merton.price())