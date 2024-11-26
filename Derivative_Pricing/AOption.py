"""American Option"""
from binomialPricing import (
    BinomialOptionModel,

)
import numpy as np
import numpy.random as npr
from trinomialPricing import TrinomialOptionModel
from MonteCarlo import MonteCarloGBM
from heston import HestonModel
from merton import MertonModel


class AmericanMixin:
    def optionPriceAtNode(self, step, d_index, ):
        opt_price = super().optionPriceAtNode(step, d_index)
        return max(opt_price, self._payoff(self.underlying_price[step, d_index]))


class AmericanOptionBinomial(AmericanMixin, BinomialOptionModel):
    pass


class AmericanOptionTrinomial(AmericanMixin, TrinomialOptionModel):
    pass


class AmericanMCMixin:
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
        S = self.stockSimulation() if S is None else S
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
        S = self.stockSimulation() if S is None else S

        payoffs = self._payoff(S[-1], self.strike)
        cashflows = payoffs.copy()

        for t in range(self.nb_steps - 1, 0, -1):
            # in-the-money paths
            in_the_money = S[t] < self.strike if (self.option_type == "put") else S[t] > self.strike
            in_the_money_paths = np.where(in_the_money)[0]
            # print("in the money paths for time step ", t, ":", in_the_money_paths)

            # regression
            X = S[t, in_the_money_paths]
            Y = self.discount_factor * cashflows[in_the_money_paths]

            if len(X) > 0:
                regression = np.polyfit(X, Y, 2)
                continuation_value = np.polyval(regression, X)

                intrinsic_value = self._payoff(X, self.strike)
                exercise = intrinsic_value > continuation_value

                cashflows[in_the_money_paths[exercise]] = intrinsic_value[exercise]

            cashflows = self.discount_factor * cashflows

        self.payoff = cashflows
        return self.payoff

    def price(self):
        t0_price_vector = self.longstaffSchwartzPayoff()
        return np.mean(t0_price_vector)

    def vega(self, epsilon=0.01, multiplicative=False):
        """There seems to be a bug in generic vega.. TODO.
        so manually calculate vega"""
        up_val = (self.sigma * (1 + epsilon)) if multiplicative else (self.sigma + epsilon)
        down_val = (self.sigma * (1 - epsilon)) if multiplicative else (self.sigma - epsilon)

        option_up = self.__class__(self.strike, self.time, self.S0, self.rate, self.option_type, up_val, self.expiry)
        option_down = self.__class__(self.strike, self.time, self.S0, self.rate, self.option_type, down_val,
                                     self.expiry)

        return (option_up.price() - option_down.price()) / (up_val - down_val)


class AmericanOptionGBM(AmericanMCMixin, MonteCarloGBM):
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

    def stockSimulation(self):
        if self.S is not None:
            return self.S
        self.S = np.zeros((self.nb_steps + 1, self.nb_iters))
        self.S[0] = self.S0
        for t in range(1, self.nb_steps + 1):
            z = npr.randn(self.nb_iters)
            self.S[t] = self.S[t - 1] * np.exp(self.drift * self.dt + self.vol * z)
        return self.S


class AmericanOptionHeston(AmericanMCMixin, HestonModel):
    pass


class AmericanOptionMerton(AmericanMCMixin, MertonModel):
    pass


if __name__ == "__main__":
    # a = AmericanOptionBinomial(45, 5, 5, 45, 0., 'call', 1.2, 1/1.2)
    a = AmericanOptionBinomial(45, 50, 50, 45, 0., 'put', 1.5, 1 / 1.5)
    a.calculateOptionPrice()
    print(a.underlying_price)
    print("Option Price: ", a.getOptionPrice())
    a.visualizeAllGrids()
