import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from mixins import OptionMixin

class ClosedFormPricing(OptionMixin):
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry):
        self.strike = strike
        self.time = time
        self.S0 = S0
        self.rate = rate
        self.option_type = option_type
        self.sigma = sigma
        self.expiry = expiry
        self.time_to_maturity = self.expiry - self.time
        self._volCalculation()
        self._validateOptionType()

    def _volCalculation(self):
        self.vol =  self.sigma * np.sqrt(self.time_to_maturity)

    @property
    def dMinus(self):
        return (np.log(self.S0 / self.strike) + (self.rate - 0.5 * self.sigma**2) * self.time_to_maturity )/ self.vol

    @property
    def dPlus(self):
        return self.dMinus + self.vol

    def price(self):
        if self.time_to_maturity<0:
            return 0.
        elif self.time_to_maturity == 0:
            return self._payoff(self.S0)
        if self.option_type == "call":
            return self.S0 * norm.cdf(self.dPlus) - self.strike * np.exp(-self.rate * self.time_to_maturity) * norm.cdf(self.dMinus)
        elif self.option_type == "put":
            return self.strike * np.exp(-self.rate * self.time_to_maturity) * norm.cdf(-self.dMinus) - self.S0 * norm.cdf(-self.dPlus)


    def delta(self):
        if self.option_type == "call":
            return norm.cdf(self.dPlus)
        elif self.option_type == "put":
            return -norm.cdf(-self.dPlus)

    def gamma(self):
        return norm.pdf(self.dPlus) / (self.S0 * self.vol)

    def vega(self):
        return self.S0 * norm.pdf(self.dPlus) * np.sqrt(self.time_to_maturity)

    def theta(self):
        if self.option_type == "call":
            return -self.S0 * norm.pdf(self.dPlus) * self.vol / (2 * self.time_to_maturity) - self.rate * self.strike * np.exp(-self.rate * self.time_to_maturity) * norm.cdf(self.dMinus)
        elif self.option_type == "put":
            return -self.S0 * norm.pdf(self.dPlus) * self.vol / (2 * self.time_to_maturity) + self.rate * self.strike * np.exp(-self.rate * self.time_to_maturity) * norm.cdf(-self.dMinus)

    def rho(self):
        if self.option_type == "call":
            return self.strike * self.time_to_maturity * np.exp(-self.rate * self.time_to_maturity) * norm.cdf(self.dMinus)
        elif self.option_type == "put":
            return -self.strike * self.time_to_maturity * np.exp(-self.rate * self.time_to_maturity) * norm.cdf(-self.dMinus)


if __name__ == "__main__":
    T = 2.0
    S = 100.0
    K = 105.0
    r = 0
    vol = 0.20
    option_type = "call"
    option = ClosedFormPricing(K, 0, S, r, option_type, vol, T)

    print_all = lambda option: print(f"Price: {option.price()}, Delta: {option.delta()}, Gamma: {option.gamma()}, Vega: {option.vega()}, Theta: {option.theta()}, Rho: {option.rho()}")
    print_all(option)

    # class notes
    # Option price = 9.197350649294513
    # Delta = 0.4876036978454982
    # Gamma = 0.014097929791127266
    # Vega = 56.39171916450907
    # Theta = -2.819585958225453
    # Rho = 79.1260382705106

    # Replica:
    # Price: 9.19735064929452, Delta: 0.48760369784549823, Gamma: 0.014097929791127265, Vega: 56.39171916450907, Theta: -2.8195859582254537, Rho: 79.1260382705106
    #


    print_all(ClosedFormPricing(K, 0, S, r, "put", vol, T))

    # class notes
    # Option price = 14.197350649294513
    # Delta = -0.5123963021545018
    # Gamma = 0.014097929791127266
    # Vega = 56.39171916450907
    # Theta = -2.819585958225453
    # Rho = -130.8739617294894

    # Replica:
    # Price: 14.197350649294513, Delta: -0.5123963021545018, Gamma: 0.014097929791127265, Vega: 56.39171916450907, Theta: -2.8195859582254537, Rho: -130.8739617294894
    #