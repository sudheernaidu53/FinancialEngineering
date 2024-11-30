"""Exotic Options
1. Barrier Options. Up-and-Out Option
"""
from MonteCarlo import MonteCarloGBM
import numpy as np
import numpy.random as npr
from heston import HestonModel
from merton import MertonModel


class BarrierOptionsGBM(MonteCarloGBM):
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry, barrier, nb_iters=100000, nb_steps=50):
        self.barrier = barrier
        self.nb_steps = nb_steps
        super().__init__(strike, time, S0, rate, option_type, sigma, expiry, nb_iters)

    def initDependencies(self):
        super().initDependencies()
        self.stock = None
        self.knocked = None
        self.dt = self.time_to_maturity / self.nb_steps
        self.vol = self.sigma * np.sqrt(self.dt)

    def setup(self):
        if self.stock is not None:
            return self.stock, self.knocked
        self.stock = np.zeros((self.nb_steps + 1, self.nb_iters), dtype=float)
        self.stock[0] = self.S0

        self.knocked = np.zeros(self.nb_iters, dtype=bool)


class UpAndOutEuropeanGBM(BarrierOptionsGBM):
    """
     interested in buying calls Apple Inc. (AAPL) because they believe the price will rise.
     buy 100 contracts with the cost as low as possible.
     Apple stock is trading at $200, investors believe it will go up but probably won't rise above $240.
     They decide to buy an at-the-money up-and-out option with a strike price of $200, an expiration in three months, and a knock-out level of $240.
     vanilla call: $11.80
     up and out call: $8.80
     If, at any time before expiration, the price of Apple stock touches $240, the options cease to exist and the investor loses the premium they paid ($88,000).
    """

    def stockPriceGrid(self):
        self.setup()
        for t in range(1, self.nb_steps + 1):
            z = npr.randn(self.nb_iters)
            self.stock[t] = self.stock[t - 1] * np.exp(self.drift * self.dt + self.vol * z)

            # if barrier is met, then set those  paths to zero
            self.knocked |= (self.stock[t] >= self.barrier)
            self.stock[t, self.knocked] = 0
        return self.stock, self.knocked

    def price(self):
        stock, knocked = self.stockPriceGrid()
        payoff = np.where(knocked, 0,
                          self._payoff(stock[-1], self.strike))  # payoffs for only the paths that are not knocked out
        return np.exp(-self.rate * self.time_to_maturity) * np.mean(payoff)


class UpAndInEuropeanGBM(BarrierOptionsGBM):
    pass  # TO DO


class BarrierOptionsHeston(HestonModel):
    def __init__(self, S0, strike, time_to_maturity, rate, vov, rho, v0, kappa, theta, option_type, barrier,
                 nb_steps=10000, seed=42, nb_iters=10000):
        self.barrier = barrier
        super().__init__(S0, strike, time_to_maturity, rate, vov, rho, v0, kappa, theta, option_type, nb_steps=nb_steps,
                         seed=seed, nb_iters=nb_iters)


class KnockedInMixin:

    def initDependencies(self):
        super().initDependencies()
        self.knocked_in = None

    def getKnocked(self):
        if self.knocked_in is not None:
            return self.knocked_in
        self.knocked_in = np.zeros(self.nb_iters, dtype=bool)
        self.simulate()
        for t in range(1, self.nb_steps + 1):
            if self.BARRIER_TYPE == "DOWN_AND_IN":
                self.knocked_in |= (self.stock[t] <= self.barrier)
            elif self.BARRIER_TYPE == "UP_AND_IN":
                self.knocked_in |= (self.stock[t] >= self.barrier)
            else:
                raise ValueError("Unknown barrier type")
        return self.knocked_in

    def price(self):
        knocked_in = self.getKnocked()
        payoff = np.where(knocked_in, self._payoff(self.stock[-1], self.strike), 0)
        return np.exp(-self.rate * self.time_to_maturity) * np.mean(payoff)


class UpAndInEuropeanHeston(KnockedInMixin, BarrierOptionsHeston):
    BARRIER_TYPE = "UP_AND_IN"


class BarrierOptionsMerton(MertonModel):
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry,
                 lambda_, mu, delta_, nb_steps, nb_iters, barrier, seed=42):
        self.barrier = barrier
        super().__init__(strike, time, S0, rate, option_type, sigma, expiry,
                         lambda_, mu, delta_, nb_steps, nb_iters, seed=seed)


class DownAndInEuropeanMerton(KnockedInMixin, BarrierOptionsMerton):
    BARRIER_TYPE = "DOWN_AND_IN"
