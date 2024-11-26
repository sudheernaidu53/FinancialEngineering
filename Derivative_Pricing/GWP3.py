from GWP1 import docstringDecorator
from decorators import timer
from AOption import AmericanOptionHeston, AmericanOptionMerton
from EOption import EuropeanOptionHeston, EuropeanOptionMerton
from tabulate import tabulate
from IPython.display import display, HTML
import pandas as pd
import numpy as np

class GWP3:
    """
        Assumptions
        1. vega is reported as change in 1% of vol.
    """
    COMMAND_LINE = True

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
        Using the Heston Model and Monte-Carlo simulation, price an ATM European call and put, using a correlation value of -0.30.
        Calculate delta and gamma for each of the options.
        """
        return self.heston(self.rho1)

    @timer()
    @docstringDecorator
    def hestonRho2(self):
        """
        Using the Heston Model and Monte-Carlo simulation, price an ATM European call and put, using a correlation value of -0.70.
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
        Using the Merton Model, price an ATM European call and put with jump intensity parameter equal to 0.75. Calculate delta and gamma for each of the options
        """
        return self.merton(self.lambda1)

    @timer()
    @docstringDecorator
    def mertonLambda2(self):
        """
        Using the Merton Model, price an ATM European call and put with jump intensity parameter equal to 0.25. Calculate delta and gamma for each of the options
        """
        return self.merton(self.lambda2)


if __name__ == "__main__":
    np.random.seed(42)
    gwp3 = GWP3()
    gwp3.hestonRho1()
    gwp3.hestonRho2()
    gwp3.mertonLambda1()
    gwp3.mertonLambda2()
