from GWP1 import docstringDecorator
from decorators import timer
from AOption import AmericanOptionHeston, AmericanOptionMerton
from EOption import EuropeanOptionHeston, EuropeanOptionMerton
from tabulate import tabulate
from IPython.display import display, HTML
import pandas as pd
import numpy as np
from exotics import UpAndInEuropeanHeston, DownAndInEuropeanMerton


class GWP3:
    """
        Assumptions
        1. vega is reported as change in 1% of vol.
    """
    COMMAND_LINE = True
    SEED = 42

    def __init__(self, S0=80, rate=0.055, sigma=0.35, time_to_matuirty=3 / 12, nb_steps_per_year=252 * 8,
                 nb_iters=10000,
                 moneyness=0,
                 option_type="call",
                 # heston parameters
                 v0=0.032,
                 kappa=1.85,
                 theta=0.045,
                 rho1=-0.3,
                 rho2=-0.7,
                 # merton parameters
                 mu=-0.5,
                 delta_=0.22,
                 lambda1=0.75,
                 lambda2=0.25
                 ):
        self.S0 = S0
        self.rate = rate
        self.sigma = sigma
        self.time_to_matuirty = time_to_matuirty
        self.nb_steps = int(nb_steps_per_year * time_to_matuirty)
        self.nb_iters = nb_iters
        self.option_type = option_type
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.mu = mu
        self.delta_ = delta_
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.rho1 = rho1
        self.rho2 = rho2
        self.strike = (1 + moneyness) * S0

    @classmethod
    def displayDataFrame(cls, df):
        if cls.COMMAND_LINE:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        else:  # jupyter notebook
            display(HTML(df.to_html(index=False)))

    def getStats(self, call_opt, put_opt, exer_type, model_type):
        call_opt.setSeed(self.SEED)
        put_opt.setSeed(self.SEED)
        call_price = call_opt.price()
        put_price = put_opt.price()
        call_delta = call_opt.delta()
        put_delta = put_opt.delta()
        call_gamma = call_opt.gamma()
        put_gamma = put_opt.gamma()
        df = pd.DataFrame(
            {
                "Type": ["ATM Call", "ATM Put"],
                "Exer": [exer_type, exer_type],
                "Price": [call_price, put_price],
                "Delta": [call_delta, put_delta],
                "Gamma": [call_gamma, put_gamma],
                "Model": [model_type, model_type]
            }

        )
        self.displayDataFrame(df)
        return df

    def heston(self, rho):
        kw = dict(S0=self.S0, strike=self.strike, time_to_maturity=self.time_to_matuirty,
                  rate=self.rate, vov=self.sigma, v0=self.v0, kappa=self.kappa,
                  theta=self.theta, rho=rho,
                  nb_steps=self.nb_steps, nb_iters=self.nb_iters)
        heston_call = EuropeanOptionHeston(option_type="call", **kw)
        heston_put = EuropeanOptionHeston(option_type="put", **kw)

        return self.getStats(heston_call, heston_put, "European", model_type="Heston")

    @timer()
    @docstringDecorator
    def hestonRho1(self):
        """
        5 and 7a: Using the Heston Model and Monte-Carlo simulation, price an ATM European call and put, using a correlation value of -0.30.
        Calculate delta and gamma for each of the options.
        """
        return self.heston(self.rho1)

    @timer()
    @docstringDecorator
    def hestonRho2(self):
        """
        6 and 7b: Using the Heston Model and Monte-Carlo simulation, price an ATM European call and put, using a correlation value of -0.70.
        Calculate delta and gamma for each of the options.
        """
        return self.heston(self.rho2)

    def merton(self, lambda_):

        kw = dict(S0=self.S0, strike=self.strike, time=0, expiry=self.time_to_matuirty, sigma=self.sigma,
                  rate=self.rate, mu=self.mu, delta_=self.delta_, lambda_=lambda_,
                  nb_steps=self.nb_steps, nb_iters=self.nb_iters)
        merton_call = EuropeanOptionMerton(option_type="call", **kw)
        merton_put = EuropeanOptionMerton(option_type="put", **kw)
        return self.getStats(merton_call, merton_put, "European", model_type="Merton")

    @timer()
    @docstringDecorator
    def mertonLambda1(self):
        """
        8 and 10 a: Using the Merton Model, price an ATM European call and put with jump intensity parameter equal to 0.75. Calculate delta and gamma for each of the options
        """
        return self.merton(self.lambda1)

    @timer()
    @docstringDecorator
    def mertonLambda2(self):
        """
        9 and 10 b: Using the Merton Model, price an ATM European call and put with jump intensity parameter equal to 0.25. Calculate delta and gamma for each of the options
        """
        return self.merton(self.lambda2)

    # @timer()
    # @docstringDecorator
    # def hestonAmerican(self):
    #     """
    #     13. a. Using the Heston Model, price an ATM American call and put
    #     """
    #     kw = dict(S0=self.S0, strike=self.strike, time_to_maturity=self.time_to_matuirty,
    #               rate=self.rate, vov=self.sigma, v0=self.v0, kappa=self.kappa,
    #               theta=self.theta, rho=self.rho1,
    #               nb_steps=self.nb_steps, nb_iters=self.nb_iters)
    #     heston_call = AmericanOptionHeston(option_type="call", **kw)
    #     heston_put = AmericanOptionHeston(option_type="put", **kw)
    #
    #     return self.getStats(heston_call, heston_put, "American", model_type="Heston")
    #
    # @timer()
    # @docstringDecorator
    # def mertonAmerican(self):
    #     """
    #     13. b. Using the Merton Model, price an ATM American call and put
    #     """
    #     kw = dict(S0=self.S0, strike=self.strike, time=0, expiry=self.time_to_matuirty, sigma=self.sigma,
    #               rate=self.rate, mu=self.mu, delta_=self.delta_, lambda_=self.lambda1,
    #               nb_steps=self.nb_steps, nb_iters=self.nb_iters)
    #     merton_call = AmericanOptionMerton(option_type="call", **kw)
    #     merton_put = AmericanOptionMerton(option_type="put", **kw)
    #     return self.getStats(merton_call, merton_put, "American", model_type="Merton")

    @timer()
    @docstringDecorator
    def americanCall(self):
        """
        13. Repeat Questions 5 and 8 for the case of an American call option. Comment on
the differences you observe from original Questions 5 and 8.
        """
        kw = dict(S0=self.S0, strike=self.strike, time_to_maturity=self.time_to_matuirty,
                  rate=self.rate, vov=self.sigma, v0=self.v0, kappa=self.kappa,
                  theta=self.theta, rho=self.rho1,
                  nb_steps=self.nb_steps, nb_iters=self.nb_iters)
        heston_european = EuropeanOptionHeston(option_type="call", **kw)
        heston_call = AmericanOptionHeston(option_type="call", **kw)
        kw = dict(S0=self.S0, strike=self.strike, time=0, expiry=self.time_to_matuirty, sigma=self.sigma,
                  rate=self.rate, mu=self.mu, delta_=self.delta_, lambda_=self.lambda1,
                  nb_steps=self.nb_steps, nb_iters=self.nb_iters)
        merton_european = EuropeanOptionMerton(option_type="call", **kw)
        merton_call = AmericanOptionMerton(option_type="call", **kw)
        self.displayDataFrame(
            pd.DataFrame(
                {"Exer": ["American", "European", "American", "European"],
                 "Model": ["Heston", "Heston", "Merton", "Merton"],
                 "Type": ["Call", "Call", "Call", "Call"],
                 "Price": [heston_call.price(), heston_european.price(), merton_call.price(), merton_european.price()],
                 }
            )

        )

    @timer()
    @docstringDecorator
    def upAndInHestonEuropean(self):
        """
        14. Using Heston model data from Question 6, price a European up-and-in call option
        (UAI) with a barrier level of $95 and a strike price of $95 as well. This UAI option
        becomes alive only if the stock price reaches (at some point before maturity) the
        barrier level (even if it ends below it). Compare the price obtained to the one from
        the simple European call.
        rho is -0.70
        """
        barrier = 85
        opt = UpAndInEuropeanHeston(S0=self.S0, strike=barrier, time_to_maturity=self.time_to_matuirty,
                                    rate=self.rate, vov=self.sigma, v0=self.v0, kappa=self.kappa,
                                    theta=self.theta, rho=self.rho2, option_type="call",
                                    nb_steps=self.nb_steps, nb_iters=self.nb_iters, barrier=barrier
                                    )
        kw = dict(S0=self.S0, strike=barrier, time_to_maturity=self.time_to_matuirty,
                  rate=self.rate, vov=self.sigma, v0=self.v0, kappa=self.kappa,
                  theta=self.theta, rho=self.rho2,
                  nb_steps=self.nb_steps, nb_iters=self.nb_iters)
        heston_call = EuropeanOptionHeston(option_type="call", **kw)
        self.displayDataFrame(
            pd.DataFrame(
                {"Exer": ["European", "European"],
                 "Model": ["Heston", "Heston"],
                 "Price": [opt.price(), heston_call.price()],
                 "Type": ["Up and In", "ATM Call"],

                 }
            )

        )

    @timer()
    @docstringDecorator
    def downAndInMertonEuropean(self):
        """
15. Using Merton model data from Question 8, price a European down-and-in put
option (DAI) with a barrier level of $65 and a strike price of $65 as well. This UAO
option becomes alive only if the stock price reaches (at some point before
maturity) the barrier level (even if it ends above it). Compare the price obtained to
the one from the simple European put.
        """
        barrier = 65
        opt = DownAndInEuropeanMerton(S0=self.S0, strike=barrier, time=0,
                                      expiry=self.time_to_matuirty,
                                      rate=self.rate, sigma=self.sigma, mu=self.mu, delta_=self.delta_,
                                      lambda_=self.lambda1, option_type="put",
                                      nb_steps=self.nb_steps, nb_iters=self.nb_iters, barrier=barrier
                                      )
        kw = dict(S0=self.S0, strike=barrier, time=0,
                  expiry=self.time_to_matuirty,
                  rate=self.rate, sigma=self.sigma, mu=self.mu, delta_=self.delta_,
                  lambda_=self.lambda1, option_type="put",
                  nb_steps=self.nb_steps, nb_iters=self.nb_iters)
        merton_put = EuropeanOptionMerton(**kw)
        self.displayDataFrame(
            pd.DataFrame(
                {"Exer": ["European", "European"],
                 "Model": ["Merton", "Merton"],
                 "Price": [opt.price(), merton_put.price()],
                 "Type": ["Down and In", "ATM Put"],

                 }
            )

        )


if __name__ == "__main__":
    np.random.seed(42)
    gwp3 = GWP3()
    # gwp3.hestonRho1()
    # gwp3.hestonRho2()
    # gwp3.mertonLambda1()
    # gwp3.mertonLambda2()
    # gwp3.americanCall()
    gwp3.upAndInHestonEuropean()
    # gwp3.downAndInMertonEuropean()
