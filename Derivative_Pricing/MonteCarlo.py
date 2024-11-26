import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from closedForm import ClosedFormPricing
import numpy.random as npr
from scipy.stats import norm


class MonteCarloMixin(object):
    @classmethod
    def setSeed(cls, seed):
        """There is no direct way to unset this? """
        npr.seed(seed)

    def reset(self):
        self.initDependencies()


    def secondOrder(self, attrib, epsilon=0.01, multiplicative=False):
        original_val = getattr(self, attrib)
        if multiplicative:
            assert original_val, "original value cannot be zero if the mode is multiplicative"

        up_val = (original_val * (1 + epsilon)) if multiplicative else (original_val + epsilon)
        down_val = (original_val * (1 - epsilon)) if multiplicative else (original_val - epsilon)
        mid_val = original_val

        # Reset and compute prices
        self.reset()
        setattr(self, attrib, up_val)
        price_up = self.price()

        self.reset()
        setattr(self, attrib, down_val)
        price_down = self.price()

        self.reset()
        setattr(self, attrib, mid_val)
        price_mid = self.price()

        diff = (up_val - down_val)/2
        # Second derivative approximation: f''(x) ≈ (f(x+ε) - 2*f(x) + f(x-ε)) / ε²
        # it is (delta_up - delta_down)/ diff, which leads to above formula
        return (price_up - 2 * price_mid + price_down) / ( diff ** 2)

    def gamma(self, epsilon=0.01, multiplicative=False):
        return self.secondOrder("S0", epsilon, multiplicative)

    def firstOrder(self, attrib, epsilon=0.01, multiplicative=False):

        original_val = getattr(self, attrib)
        if multiplicative:
            assert original_val, "original value cannot be zero if the mode is multiplicative"
        up_val = (original_val * (1 + epsilon)) if multiplicative else (original_val + epsilon)
        down_val = (original_val * (1 - epsilon)) if multiplicative else (original_val - epsilon)
        self.reset()

        setattr(self, attrib, up_val)
        price_up = self.price()

        self.reset()
        setattr(self, attrib, down_val)
        price_down = self.price()

        self.reset()

        return (price_up - price_down) / (up_val - down_val)

    def delta(self, epsilon=0.01, multiplicative=False):
        return self.firstOrder("S0", epsilon, multiplicative)

    def vega(self, epsilon=0.01, multiplicative=False):
        return self.firstOrder("sigma", epsilon, multiplicative)

    def theta(self, epsilon=0.01, multiplicative=False):
        return self.firstOrder("time", epsilon, multiplicative)

    def rho(self, epsilon=0.01, multiplicative=False):
        return self.firstOrder("rate", epsilon, multiplicative)


class MonteCarloGBM(MonteCarloMixin):
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry, nb_iters=100000):
        self.strike = strike
        self.time = time
        self.S0 = S0
        self.rate = rate
        self.option_type = option_type
        self.sigma = sigma
        self.expiry = expiry
        self.nb_iters = nb_iters
        self.initDependencies()

    @property
    def drift(self):
        return self.rate - 0.5 * self.sigma ** 2

    @property
    def gbmVol(self):
        return self.sigma*np.sqrt(self.time_to_maturity)

    def initDependencies(self):
        self.time_to_maturity = self.expiry - self.time

    def terminalStockPrice(self):
        return self.S0 * np.exp(self.drift * self.time_to_maturity +self.gbmVol * npr.randn(self.nb_iters))

    def _payoff(self, S, K):
        if self.option_type == "call":
            return np.maximum(S - K, 0)
        elif self.option_type == "put":
            return np.maximum(K - S, 0)

    def terminalPayoffs(self):
        return self._payoff(self.terminalStockPrice(), self.strike)

    def discount(self):
        return np.exp(-self.rate * self.time_to_maturity)

    def price(self):
        return self.discount() * np.mean(self.terminalPayoffs())



class MonteCarloConvergence:
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry, nb_iters_list, tolerance=0.01,
                 klass=MonteCarloGBM, **kw):
        self.strike = strike
        self.time = time
        self.S0 = S0
        self.rate = rate
        self.option_type = option_type
        self.sigma = sigma
        self.expiry = expiry
        self.nb_iters_list = nb_iters_list
        self.prices = None
        self.tolerance = tolerance
        self.klass = klass
        self.kw = kw

    def getPrices(self):
        if self.prices is not None:
            return self.prices

        self.prices = []
        for nb_iters in self.nb_iters_list:
            mc = self.klass(strike=self.strike, time=self.time, S0=self.S0, rate=self.rate,
                            option_type=self.option_type, sigma=self.sigma, expiry=self.expiry, nb_iters=nb_iters,
                            **self.kw)
            self.prices.append(mc.price())
        return self.prices

    def plot(self, overlay_closed_form=False):
        plt.plot(self.nb_iters_list, self.getPrices(), label=self.klass.__name__)
        plt.title("Convergence of " + self.klass.__name__)
        plt.xlabel("Number of iterations")
        plt.ylabel("Option price")
        if overlay_closed_form:
            cf_pricer = ClosedFormPricing(self.strike, self.time, self.S0, self.rate, self.option_type, self.sigma,
                                          self.expiry)
            cf_price = cf_pricer.price()
            plt.hlines(cf_price, xmin=self.nb_iters_list[0], xmax=self.nb_iters_list[-1], colors="red",
                       label="closedForm BlackScholes")
        plt.vlines(self.toleranceAchievement(), ymin=self.getPrices()[0], ymax=self.getPrices()[-1], colors="green",
                   label="convergence", linestyles="dotted")
        plt.grid()
        plt.legend()
        plt.show()

    def toleranceAchievement(self):
        prices = self.getPrices()
        for i, nb_iters in enumerate(self.nb_iters_list):
            if i == 0:
                continue
            if np.abs(prices[i] - prices[i-1]) < self.tolerance:
                return nb_iters


if __name__ == "__main__":
    MonteCarloMixin.setSeed(42)
    mc = MonteCarloGBM(95, 0, 100, 0.06, "call", 0.3, 1)

    print(mc.price())

    # mc_convergence = MonteCarloConvergence(95, 0, 100, 0.06, "call", 0.3, 1, range(1, 100000, 500))
    # mc_convergence.plot(overlay_closed_form=True)
    # print(mc_convergence.toleranceAchievement())

    # american_option = AmericanOptionGBM(95, 0, 100, 0.06, "call", 0.3, nb_steps=1000, expiry=1, nb_iters=100000)
    # print(american_option.price())
    #
    # american_option_convergence = MonteCarloConvergence(95, 0, 100, 0.06, "call", 0.3, 1, range(1, 100000, 500),
    #                                                    klass=AmericanOptionGBM)
    # american_option_convergence.plot(overlay_closed_form=True)
    # print(american_option_convergence.toleranceAchievement())
