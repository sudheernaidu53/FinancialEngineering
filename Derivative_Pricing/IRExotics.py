import numpy as np
import matplotlib.pyplot as plt


class IRCap:
    """
    provides protection against rising interest rates.
    The payoff of an IR Cap is calculated as
    the difference between the floating interest rate (e.g., LIBOR) and the strike rate,
    multiplied by the notional amount and the accrual period,
    but only if the floating rate exceeds the strike rate.
    If the floating rate is below the strike rate, the payoff is zero.
    """

    def __init__(self, r0, strike_rate, expiry, side="long", numsamples=1000):
        self.side = side
        self.r0 = r0
        self.strike_rate = strike_rate
        self.expiry = expiry
        self.numsamples = numsamples

    def payOff(self, r_array=None):
        r_array = r_array if r_array is not None else np.linspace(self.r0 - self.strike_rate,
                                                                  self.r0 + self.strike_rate, self.numsamples)

        payoff = np.maximum(0, r_array - self.strike_rate)
        return payoff if self.side == "long" else -payoff

    def plotPayoff(self):
        r_array = np.linspace(self.r0 - self.strike_rate, self.r0 + self.strike_rate, self.numsamples)
        payoff = self.payOff(r_array)
        plt.plot(r_array, payoff)
        plt.xlabel('Interest Rate')
        plt.ylabel('Payoff of Cap with strike rate ' + str(self.strike_rate))
        plt.show()


class IRFloor:
    """
    provides protection against falling interest rates.
    The payoff of an IR Floor is calculated as
    the difference between the floating interest rate (e.g., LIBOR) and the strike rate,
    multiplied by the notional amount and the accrual period,
    but only if the floating rate falls below the strike rate.
    If the floating rate is above the strike rate, the payoff is zero.
    """

    def __init__(self, r0, strike_rate, expiry, side="long", numsamples=1000):
        self.r0 = r0
        self.strike_rate = strike_rate
        self.expiry = expiry
        self.side = side
        self.numsamples = numsamples

    def payOff(self, r_array=None):
        r_array = r_array if r_array is not None else np.linspace(self.r0 - self.strike_rate,
                                                                  self.r0 + self.strike_rate, self.numsamples)
        payoff = np.maximum(0, self.strike_rate - r_array)
        return payoff if self.side == "long" else -payoff

    def plotPayoff(self):
        r_array = np.linspace(self.r0 - self.strike_rate, self.r0 + self.strike_rate, self.numsamples)
        payoff = self.payOff(r_array)
        plt.plot(r_array, payoff)
        plt.xlabel('Interest Rate')
        plt.ylabel('Payoff of Floor with strike rate ' + str(self.strike_rate))
        plt.show()


class RangeAccrual:
    """
    A Range Accrual Instrument is a type of Interest Rate (IR) derivative that pays a fixed rate of interest
    (accrual rate) on a notional amount,
    but only for the periods when the underlying interest rate (e.g., LIBOR) falls within a
    predetermined range (the "range").
    """

    def __init__(self, r0, upper_rate, lower_rate, accrual_rate, expiry, side="long", numsamples=1000):
        self.r0 = r0
        self.upper_rate = upper_rate
        self.lower_rate = lower_rate
        self.accrual_rate = accrual_rate
        self.expiry = expiry
        self.side = side

    def payOff(self, r_array=None):
        r_array = r_array if r_array is not None else np.linspace(self.lower_rate - self.r0,
                                                                  self.r0 + self.upper_rate, self.numsamples)
        payoff = np.where((r_array < self.upper_rate) & (r_array > self.lower_rate), self.accrual_rate, 0)
        return payoff if self.side == "long" else -payoff

    def plotPayoff(self):
        r_array = np.linspace(self.lower_rate - self.r0, self.r0 + self.upper_rate, self.numsamples)
        payoff = self.payOff(r_array)
        plt.plot(r_array, payoff)
        plt.xlabel('Interest Rate')
        plt.ylabel('Payoff of Range Accrual with range ' + str(self.lower_rate) + ' to ' + str(self.upper_rate))
        plt.show()


class RangeAccrualAsBasket:
    """
    Range Accrual could be approximated as a basket of IR caps and IR floors.
    2 IR caps around the Lower barrier. the one with lower_rate - epsilon as long and the lower_rate+epsilon as short, this creates an upward slope
    2 IR floors around the Upper barrier. the one with upper_rate + epsilon as long and the upper_rate-epsilon as short, this creates as downward slope
    
    scale them up by the accrual rate/epsilon
    need a fixed cashflow to complete the setup of replication?    
    """

    def __init__(self, r0, upper_rate, lower_rate, accrual_rate, expiry, epsilon=0.0001, numsamples=None):
        self.r0 = r0
        self.upper_rate = upper_rate
        self.lower_rate = lower_rate
        self.accrual_rate = accrual_rate
        self.expiry = expiry
        self.epsilon = epsilon
        self.scaling_factor = self.accrual_rate / self.epsilon
        self.diff_bounds = self.upper_rate - self.lower_rate
        self.lower_bound_for_graph = self.lower_rate - self.diff_bounds/5
        self.upper_bound_for_graph = self.upper_rate + self.diff_bounds/5
        self.numsamples = numsamples or int((self.upper_bound_for_graph - self.lower_bound_for_graph)/self.epsilon)

    def replicatingBasket(self):
        self.long_cap = IRCap(self.r0, self.lower_rate - self.epsilon / 2, self.expiry, "long")
        self.short_cap = IRCap(self.r0, self.lower_rate + self.epsilon / 2, self.expiry, "short")
        self.long_floor = IRFloor(self.r0, self.upper_rate + self.epsilon / 2, self.expiry, "long")
        self.short_floor = IRFloor(self.r0, self.upper_rate - self.epsilon / 2, self.expiry, "short")
        self.fixed_cashflow = -self.epsilon #self.accrual_rate/self.epsilon
        return [self.long_cap, self.short_cap, self.long_floor, self.short_floor]

    def payOff(self, r_array=None):
        r_array = r_array if r_array is not None else np.linspace(self.lower_bound_for_graph,
                                                                  self.upper_bound_for_graph, self.numsamples)
        total_payoff = np.zeros(r_array.shape, dtype=float)
        for opt in self.replicatingBasket():
            total_payoff += opt.payOff(r_array)
        return (total_payoff+ self.fixed_cashflow)*self.scaling_factor

    def plotPayoff(self):
        r_array = np.linspace(self.lower_bound_for_graph,
                              self.upper_bound_for_graph, self.numsamples)
        payoff = self.payOff(r_array)
        color_scheme = ['blue', 'green', 'red', 'black']
        for color, opt_type, opt in zip(color_scheme, ["long_cap", "short_cap", "long_floor", "short_floor"],
                                        self.replicatingBasket()):
            plt.plot(r_array, opt.payOff(r_array), label=opt_type, color=color, linestyle='--', alpha=0.5) #self.scaling_factor*

        plt.plot(r_array, payoff, label = "replication basket", color = 'purple')

        range_accrual = RangeAccrual(self.r0, self.upper_rate, self.lower_rate, self.accrual_rate, self.expiry)
        plt.plot(r_array, range_accrual.payOff(r_array), label = "range accrual", color = 'orange')

        plt.xlabel('Interest Rate')
        plt.ylabel('Payoff of Range Accrual with range ' + str(self.lower_rate) + ' to ' + str(self.upper_rate))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    expiry = 1
    r0 = 0.1
    upper_rate = 0.15
    lower_rate = 0.05
    accrual_rate = 0.04
    epsilon = 0.0001
    # range_accrual = RangeAccrual(r0, upper_rate, lower_rate, accrual_rate, expiry)
    # range_accrual.plotPayoff()
    range_accrual_as_basket = RangeAccrualAsBasket(r0, upper_rate, lower_rate, accrual_rate, expiry, epsilon)
    range_accrual_as_basket.plotPayoff()
