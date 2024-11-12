"""Exotic Options
1. Barrier Options. Up-and-Out Option
"""
from MonteCarlo import MonteCarloPricing
import numpy as np
import numpy.random as npr

class BarrierOptions(MonteCarloPricing):
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry, barrier, nb_iters=100000, nb_steps=50):
        self.barrier = barrier
        self.nb_steps = nb_steps
        super().__init__(strike, time, S0, rate, option_type, sigma, expiry, nb_iters)

    def initDependencies(self):
        super().initDependencies()
        self.S = None
        self.knocked = None
        self.dt = self.time_to_maturity / self.nb_steps
        self.vol = self.sigma * np.sqrt(self.dt)

    def setupStockPriceGrid(self):
        if self.S is not None:
            return self.S, self.knocked
        self.S = np.zeros((self.nb_steps + 1, self.nb_iters))
        self.S[0] = self.S0

        self.knocked = np.zeros(self.nb_iters, dtype=bool)

class UpAndOutEuropeanMC(BarrierOptions):
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
        self.setupStockPriceGrid()
        for t in range(1, self.nb_steps + 1):
            z = npr.randn(self.nb_iters)
            self.S[t] = self.S[t - 1] * np.exp(self.drift * self.dt + self.vol * z)

            # if barrier is met, then set those  paths to zero
            self.knocked |= (self.S[t] >= self.barrier)
            self.S[t, self.knocked] = 0
        return self.S, self.knocked

    def price(self):
        S, knocked = self.stockPriceGrid()
        payoff = np.where(knocked, 0, self._payoff(S[-1], self.strike)) # payoffs for only the paths that are not knocked out
        return np.exp(-self.rate * self.time_to_maturity) * np.mean(payoff)


class UpAndInEuropeanMC(BarrierOptions):
    pass # TO DO
