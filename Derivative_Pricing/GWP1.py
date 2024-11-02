import pandas as pd

from AOption import AmericanOptionBinomial, AmericanOptionTrinomial
from treePricing import TreeOptionDependenceOnFactor
from EOption import EuropeanOptionBinomial, EuropeanOptionTrinomial
# import numpy as np
# import matplotlib.pyplot as plt


def docstringDecorator(func):
    def wrapper(*args, **kwargs):
        print(func.__doc__)
        return func(*args, **kwargs)

    return wrapper


class GWP1:

    def __init__(self, S0=100, moneyness=0, T=0.25, r=0.05, sigma=0.2):
        self.__initBinomialSetup(S0, moneyness, T, r, sigma)

    def __initBinomialSetup(self, S0=100, moneyness=0, T=0.25, r=0.05, sigma=0.2):
        """trinomial step is the same"""
        self.S0 = S0
        self.moneyness = moneyness
        self.K = S0 * (1 + moneyness)
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = 2000

    @docstringDecorator
    def stepsVariation(self, klass=EuropeanOptionBinomial, option_type="call"):
        """Question 5 a of GWP1
        Price an ATM European call and put using a binomial tree:
        a. Choose the number of steps in the tree you see convenient to achieve reliable estimates.
        """
        option = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type=option_type,
            steps=self.steps
        )
        option_style, model_type = klass.__name__.split("Option")
        variator = TreeOptionDependenceOnFactor(
            option=option, parameter_name="steps",
            val_range=[1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500],
            plot_title="{} {} price vs {}".format(model_type, option_type, "number of steps")
        )
        variator.plot()
        print(variator.asDict(one2one=True))
        print(variator.asDataFrame())

    @docstringDecorator
    def strikeVariation(self, klass=EuropeanOptionBinomial, option_type="call"):
        """Question 15 and 16 of GWP1
        15.Select 5 strike prices so that Call options are: Deep OTM, OTM, ATM, ITM, and
        Deep ITM. (E.g., you can do this by selecting moneyness of 90%, 95%, ATM, 105%,
        110%; where moneyness is measured as K/S0):
        a. Using the trinomial tree, price the Call option corresponding to the 5 different strikes selected.
        (Unless stated otherwise, consider input data given in Step 1).
        b. Comment on the trend you observe (e.g., increasing/decreasing in moneyness) in option prices and whether it makes sense.
        16.Repeat Q15 for 5 different strikes for Put options. (Make sure you also answer sections a and b of Q15).
        """
        option = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type=option_type,
            steps=self.steps
        )
        option_style, model_type = klass.__name__.split("Option")
        variator = TreeOptionDependenceOnFactor(
            option=option, parameter_name="strike", val_range=[self.S0 *x for x in [0.9, 0.95, 1, 1.05, 1.1]],
            plot_title="{} {} price vs {}".format(model_type, option_type, "strike"))
        variator.plot()
        print(variator.asDict(one2one=True))
        print(variator.asDataFrame())

    @docstringDecorator
    def callAndPutPrice(self, klass=EuropeanOptionBinomial):
        """Question 5 of GWP1.
        Price an ATM European call and put using a binomial tree:
        b. Brieﬂy describe the overall process, as well as a reason why you choose that number of steps in the tree.
        """
        option = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type="call",
            steps=self.steps
        )
        poption = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type="put",
            steps=self.steps
        )

        option.calculateOptionPrice()
        poption.calculateOptionPrice()
        print(
            "Call price is {:,.2f} and put price is {:,.2f}".format(option.getOptionPrice(), poption.getOptionPrice()))


    @docstringDecorator
    def deltaOfCallAndPut(self, klass=EuropeanOptionBinomial):
        """Question 6 of GWP1.
        Compute the delta of an ATM European call and put using a binomial tree:
        a.  How do they compare?
        b. Comment brieﬂy on the differences and signs of Delta for both options.
        What does delta proxy for?
        Why does it make sense to obtain a positive/negative delta for each option?
        """
        option = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type="call",
            steps=self.steps
        )
        option.calculateOptionPrice()
        option.fillDeltaGrid()
        print("Delta of {} call option is {:,.2f}".format(klass.__name__, option.getT0Delta()))

        poption = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type="put",
            steps=self.steps
        )
        poption.calculateOptionPrice()
        poption.fillDeltaGrid()
        print("Delta of {} put option is {:,.2f}".format(klass.__name__, poption.getT0Delta()))

    @docstringDecorator
    def vegaOfCallAndPut(self, klass=EuropeanOptionBinomial):
        """
        Question 7 of GWP1.
        Delta measures one sensitivity of the option price. 
        But there are other important sensitivities we will look at throughout the course. 
        An important one is the sensitivity of the option price to the underlying volatility (vega)..
        a. Compute the sensitivity of previous put and call option prices to a 5% increase in volatility (from 20% to 25%). 
        How do prices change with respect to the change in volatility?
        b. Comment on the potential differential impact of this change for call and put options
        :return:
        """
        s1 = self.sigma
        option = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type="call",
            steps=self.steps
        )
        poption = klass(
            S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=self.sigma, option_type="put",
            steps=self.steps
        )
        p1 = poption.calculateOptionPrice()
        c1 = option.calculateOptionPrice()
        s2 = 0.25
        option.setAttribute("sigma", s2)
        c2 = option.calculateOptionPrice()
        poption.setAttribute("sigma", s2)
        p2 = poption.calculateOptionPrice()
        # just a check
        # o2 = klass(
        #     S0=self.S0, strike=self.K, time=self.T, rate=self.r, sigma=s2, option_type="call",
        #     steps=self.steps
        # )
        # print("option value with setattr is {} and from actual initialization is {}".format(c2,
        #
        #                                                                         o2.calculateOptionPrice()))
        vega = (c2 - c1) / ((s2 - s1) * 100)
        p_vega = (p2 - p1) / ((s2 - s1) * 100)  # normalize to percentage change than unit change.
        pd.DataFrame({
            "type": ["call", "call", "put", "put"],
            "volatility": [s1, s2, s1, s2],
            "option value": [c1, c2, p1, p2],
            "vega": [vega, 0, p_vega, 0]
        }
        ).to_excel("{}_vega_table.xlsx".format(klass.__name__)
                   # engine="openpyxl"
                   )
        print("Vega of call and put options respectively are {:,.2f} and {:,.2f}".format(vega, p_vega))


if __name__ == "__main__":
    gwp1 = GWP1()

    # Q5,6,7
    gwp1.stepsVariation(option_type="call")
    gwp1.callAndPutPrice()
    gwp1.deltaOfCallAndPut()
    gwp1.vegaOfCallAndPut()

    #Q8,9,10
    gwp1.stepsVariation(klass=AmericanOptionBinomial, option_type="put")
    gwp1.callAndPutPrice(klass=AmericanOptionBinomial)
    gwp1.deltaOfCallAndPut(klass=AmericanOptionBinomial)
    gwp1.vegaOfCallAndPut(klass=AmericanOptionBinomial)

    #Q15,16
    gwp1.strikeVariation(klass=EuropeanOptionTrinomial, option_type="call")
    gwp1.strikeVariation(klass=EuropeanOptionTrinomial, option_type="put")

    #Q17, 18
    gwp1.strikeVariation(klass=AmericanOptionTrinomial, option_type="call")
    gwp1.strikeVariation(klass=AmericanOptionTrinomial, option_type="put")
