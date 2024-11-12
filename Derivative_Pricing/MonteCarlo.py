from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from closedForm import ClosedFormPricing
import numpy.random as npr
from scipy.stats import norm

class MonteCarlo(object):
    @classmethod
    def setSeed(cls, seed):
        """There is no direct way to unset this? """
        npr.seed(seed)

class MonteCarloPricing(MonteCarlo):
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


    def initDependencies(self):
        self.time_to_maturity = self.expiry - self.time

    def terminalStockPrice(self):
        return self.S0 * np.exp((self.rate - 0.5 * self.sigma ** 2) * self.time_to_maturity + self.sigma * np.sqrt(
            self.time_to_maturity) * npr.randn(self.nb_iters))

    def terminalPayoffs(self):
        if self.option_type == "call":
            return np.maximum(self.terminalStockPrice() - self.strike, 0)
        elif self.option_type == "put":
            return np.maximum(self.strike - self.terminalStockPrice(), 0)

    def discount(self):
        return np.exp(-self.rate * self.time_to_maturity)

    def price(self):
        return self.discount() * np.mean(self.terminalPayoffs())

    def reset(self):
        self.initDependencies()

    def firstOrder(self, attrib, epsilon = 0.1, multiplicative=True):

        original_val = getattr(self, attrib)
        if multiplicative:
            assert original_val, "original value cannot be zero if the mode is multiplicative"
        self.reset()

        setattr(self, attrib, (original_val * (1+epsilon)) if multiplicative else (original_val + epsilon))
        price_up = self.price()

        setattr(self, attrib, (original_val * (1-epsilon)) if multiplicative else (original_val - epsilon))
        price_down = self.price()

        self.reset()

        return (price_up - price_down) / (2 * epsilon)

    def delta(self, epsilon = 0.1, multiplicative=True):
        return self.firstOrder("S0", epsilon, multiplicative)

    def vega(self, epsilon = 0.1):
        return self.firstOrder("sigma", epsilon)

    def theta(self, epsilon = 0.1):
        return self.firstOrder("time", epsilon)

    def rho(self, epsilon = 0.1):
        return self.firstOrder("rate", epsilon)


class EuropeanOptionMonteCarlo(MonteCarloPricing):
    pass

class AmericanOptionMonteCarlo(MonteCarloPricing):
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry, nb_iters=100000, nb_steps=50):
        self.nb_steps = nb_steps
        super().__init__(strike, time, S0, rate, option_type, sigma, expiry, nb_iters)


    def initDependencies(self):
        super().initDependencies()
        self.dt = self.time_to_maturity / self.nb_steps  # Time step
        self.discount_factor = np.exp(-self.rate * self.dt)
        self.vol = self.sigma * np.sqrt(self.dt)
        self.S = None
        self.payoff = None

    def simulatedStockPrices(self):
        if self.S is not None:
            return self.S
        self.S = np.zeros((self.nb_steps + 1, self.nb_iters))
        self.S[0] = self.S0
        for t in range(1, self.nb_steps + 1):
            z = npr.randn(self.nb_iters)
            self.S[t] = self.S[t - 1] * np.exp((self.rate - 0.5 * self.sigma ** 2) * self.dt + self.vol * z)
        return self.S

    def _payoff(self, S, K):
        if self.option_type == "call":
            return np.maximum(S - K, 0)
        elif self.option_type == "put":
            return np.maximum(K - S, 0)

    def __SimplePayoff(self, S=None):
        """don't use. has bugs.
        treats each path as independent.
        value on a path as of some date is max(discounted value of next day payoff, exercise value)
        Above will lead to a bug? say the intrinsic values are [4,2,1,0] the code will exercise on all time steps but the last. this shouldn't be a bug.
        1. The continuation value is the expected payoff if the option is held, not just at the next step, but across all possible future paths from that point onward.
        2. it should be the conditional expectation of the future payoff, given the stock price at the current step.
        3. Stock price paths do not linearly correspond to future payoffs. For instance, when a call option is deep out of the money, the future payoff is likely zero regardless of slight changes in the stock price.
        4. By fitting to multiple paths' payoffs at later steps, the regression approximates a broader future scenario for each stock price level, providing a better decision criterion for early exercise.
        """
        S = self.simulatedStockPrices() if S is None else S
        if self.payoff is not None:
            return self.payoff
        self.payoff = np.zeros_like(S)
        # expiry
        self.payoff[-1] = self._payoff(S[-1], self.strike)

        for t in range(self.nb_steps - 1, -1, -1):
            exercise_value = self._payoff(S[t], self.strike)
            continuation_value = self.discount_factor * self.payoff[t + 1]  # discount self.payoff value to present.

            # max of exercise and continuation value.
            # print(t, sum(exercise_value > continuation_value))
            self.payoff[t] = np.where(exercise_value > continuation_value, exercise_value, continuation_value)
        return self.payoff[0]

    def longstaffSchwartzPayoff(self, S=None):
        if self.payoff is not None:
            return self.payoff
        S = self.simulatedStockPrices() if S is None else S

        payoffs = self._payoff(S[-1], self.strike)
        cashflows = payoffs.copy()

        for t in range(self.nb_steps - 1, -1, -1):
            # in-the-money paths
            in_the_money = S[t] < self.strike if (self.option_type == "put") else S[t] > self.strike
            in_the_money_paths = np.where(in_the_money)[0]
            # print("in the money paths for time step ", t, ":", in_the_money_paths)

            # regression
            X = S[ t, in_the_money_paths]
            Y = self.discount_factor* cashflows[in_the_money_paths]

            if len(X) > 0:
                regression = np.polyfit(X, Y, 2)
                continuation_value = np.polyval(regression, X)

                intrinsic_value = self._payoff(X, self.strike)
                exercise = intrinsic_value > continuation_value

                cashflows[in_the_money_paths[exercise]] = intrinsic_value[exercise]

            cashflows = self.discount_factor*cashflows

        self.payoff = cashflows
        return self.payoff

    def price(self):
        t0_price_vector = self.longstaffSchwartzPayoff()
        return np.mean(t0_price_vector)


class MonteCarloConvergence:
    def __init__(self, strike, time, S0, rate, option_type, sigma, expiry, nb_iters_list, tolerance=0.01,
                 klass=EuropeanOptionMonteCarlo, **kw):
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
            if np.abs(prices[i] - prices[-1]) < self.tolerance:
                return nb_iters


if __name__ == "__main__":
    MonteCarlo.setSeed(42)
    mc = MonteCarloPricing(95, 0, 100, 0.06, "call", 0.3, 1)

    print(mc.price())

    # mc_convergence = MonteCarloConvergence(95, 0, 100, 0.06, "call", 0.3, 1, range(1, 100000, 500))
    # mc_convergence.plot(overlay_closed_form=True)
    # print(mc_convergence.toleranceAchievement())

    american_option = AmericanOptionMonteCarlo(95, 0, 100, 0.06, "call", 0.3, nb_steps=1000, expiry=1, nb_iters=100000)
    print(american_option.price())

    american_option_convergence = MonteCarloConvergence(95, 0, 100, 0.06, "call", 0.3, 1, range(1, 100000, 500),
                                                       klass=AmericanOptionMonteCarlo)
    american_option_convergence.plot(overlay_closed_form=True)
    print(american_option_convergence.toleranceAchievement())
