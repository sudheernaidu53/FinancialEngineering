from GWP1 import docstringDecorator, GWP1
import pandas as pd
from MonteCarlo import MonteCarloConvergence
from EOption import EuropeanOptionBinomial, EuropeanOptionTrinomial, EuropeanOptionMC, EuropeanOptionBS
from AOption import AmericanOptionBinomial, AmericanOptionTrinomial, AmericanOptionMC
from exotics import UpAndOutEuropeanMC
from tabulate import tabulate
from IPython.display import display

class GWP2(GWP1):
    """
    Assumptions
    1. vega is reported as change in 1% of vol.
    """
    COMMAND_LINE=True

    def __init__(self, S0=100, moneyness=0, T=0.25, r=0.05, sigma=0.2, nb_iters=100000):
        self.nb_iters = nb_iters
        super().__init__(S0=S0, moneyness=moneyness, T=T, r=r, sigma=sigma)

    def prettifyComparisonDF(self, df, column_suffix="Price"):
        df["% Diff"] = (df["GWP1 {}".format(column_suffix)] - df["GWP2 {}".format(column_suffix)]) / df[
            "GWP1 {}".format(column_suffix)]
        df["GWP1 {}".format(column_suffix)] = df["GWP1 {}".format(column_suffix)].map("{:.2f}".format)
        df["GWP2 {}".format(column_suffix)] = df["GWP2 {}".format(column_suffix)].map("{:.2f}".format)
        df["% Diff"] = df["% Diff"].map("{:.2%}".format)

    @classmethod
    def displayDataFrame(cls, df):
        if cls.COMMAND_LINE:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        else: # jupyter notebook
            display(df.to_html(index=False))

    # def getBinomialAttributes(self, klass=None):
    #     bin_call, bin_put = self.callAndPutPrice(klass=klass or EuropeanOptionBinomial)
    #     bin_call_delta, bin_put_delta = self.deltaOfCallAndPut(klass= klass or EuropeanOptionBinomial)
    #     bin_call_vega, bin_put_vega = self.vegaOfCallAndPut()

    @docstringDecorator
    def EuropeanBS(self):
        """
        Q1:
        Team member A will repeat questions 5, 6, and 7 of GWP#1 using the
        Black-Scholes closed-form solution to price the different European Options. For
        Q7 on vega, you can use Black-Scholes closed-form solution.

        Question 5 of GWP1.
        Price an ATM European call and put using a binomial tree:
        b. Brieﬂy describe the overall process, as well as a reason why you choose that number of steps in the tree.

        Q6 of GWP1.
         Compute the Greek Delta for the European call and European put at time 0:
        a. How do they compare?
        b. Comment brieﬂy on the differences and signs of Delta for both options.
        What does delta proxy for? Why does it make sense to obtain a
        positive/negative delta for each option?

         Delta measures one sensitivity of the option price. But there are other important
        sensitivities we will look at throughout the course. An important one is the
        sensitivity of the option price to the underlying volatility (vega)..
        a. Compute the sensitivity of previous put and call option prices to a 5%
        increase in volatility (from 20% to 25%). How do prices change with
        respect to the change in volatility?
        b. Comment on the potential differential impact of this change for call and
        put options.
        """
        bin_call, bin_put = self.callAndPutPrice()
        bs_call_obj = EuropeanOptionBS(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                       option_type="call")
        bs_put_obj = EuropeanOptionBS(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                      option_type="put")
        bs_call, bs_put = bs_call_obj.price(), bs_put_obj.price()
        df = pd.DataFrame(
            {"Q": [5, 5],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Eur", "Eur"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["BS", "BS"],
             "GWP1 Price": [bin_call, bin_put],
             "GWP2 Price": [bs_call, bs_put],

             }

        )
        self.prettifyComparisonDF(df)
        self.displayDataFrame(df)

        # delta
        bs_call_delta = bs_call_obj.delta()
        bs_put_delta = bs_put_obj.delta()
        bin_call_delta, bin_put_delta = self.deltaOfCallAndPut()
        df = pd.DataFrame(
            {"Q": [6, 6],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Eur", "Eur"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["BS", "BS"],
             "GWP1 delta": [bin_call_delta, bin_put_delta],
             "GWP2 delta": [bs_call_delta, bs_put_delta],
             }

        )
        self.prettifyComparisonDF(df, column_suffix="delta")
        self.displayDataFrame(df)

        # vega
        bs_call_vega = bs_call_obj.vega() / 100.0
        bs_put_vega = bs_put_obj.vega() / 100.0
        bin_call_vega, bin_put_vega = self.vegaOfCallAndPut()
        df = pd.DataFrame(
            {"Q": [7, 7],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Eur", "Eur"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["BS", "BS"],
             "GWP1 vega": [bin_call_vega, bin_put_vega],
             "GWP2 vega": [bs_call_vega, bs_put_vega],
             }

        )
        self.prettifyComparisonDF(df, column_suffix="vega")
        self.displayDataFrame(df)

    @docstringDecorator
    def EuropeanMC(self):
        """
        2. Team member B will repeat questions 5, 6, and 7 of GWP#1 using Monte-Carlo
        methods under a general GBM equation with daily time-steps in the simulations.
        As was the case with the number of time steps in the trees, make sure you run a
        large enough number of simulations. For Q7 here you can rely on the same
        intuition as in the trees, just ‘shock’ the volatility parameter and recalculate
        things.
        :return:
        """
        bin_call, bin_put = self.callAndPutPrice()
        mc_call_obj = EuropeanOptionMC(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                       option_type="call")
        mc_put_obj = EuropeanOptionMC(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                      option_type="put")
        mc_call, mc_put = mc_call_obj.price(), mc_put_obj.price()
        df = pd.DataFrame(
            {"Q": [5, 5],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Eur", "Eur"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["MC", "MC"],
             "GWP1 Price": [bin_call, bin_put],
             "GWP2 Price": [mc_call, mc_put],
             }

        )
        self.prettifyComparisonDF(df)
        self.displayDataFrame(df)

        # delta
        mc_call_delta = mc_call_obj.delta()
        mc_put_delta = mc_put_obj.delta()
        bin_call_delta, bin_put_delta = self.deltaOfCallAndPut()
        df = pd.DataFrame(
            {"Q": [6, 6],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Eur", "Eur"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["MC", "MC"],
             "GWP1 delta": [bin_call_delta, bin_put_delta],
             "GWP2 delta": [mc_call_delta, mc_put_delta],
             }

        )
        self.prettifyComparisonDF(df, column_suffix="delta")
        self.displayDataFrame(df)

        # vega
        mc_call_vega = mc_call_obj.vega(epsilon=0.05, multiplicative=False) / 100.0
        mc_put_vega = mc_put_obj.vega(epsilon=0.05, multiplicative=False) / 100.0
        bin_call_vega, bin_put_vega = self.vegaOfCallAndPut()
        df = pd.DataFrame(
            {"Q": [7, 7],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Eur", "Eur"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["MC", "MC"],
             "GWP1 vega": [bin_call_vega, bin_put_vega],
             "GWP2 vega": [mc_call_vega, mc_put_vega],
             }

        )
        self.prettifyComparisonDF(df, column_suffix="vega")
        self.displayDataFrame(df)

    def EuropeanCallConvergence(self):
        european_option_convergence = MonteCarloConvergence(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                                            option_type="call", klass=EuropeanOptionMC, nb_iters_list=range(1, 100000, 500))
        european_option_convergence.plot(overlay_closed_form=False)
        print(european_option_convergence.toleranceAchievement())

    def AmericanMCConvergence(self):
        american_option_convergence = MonteCarloConvergence(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                                            option_type="put", klass=AmericanOptionMC, nb_iters_list=range(1, 100000, 500))
        american_option_convergence.plot(overlay_closed_form=False)
        print(american_option_convergence.toleranceAchievement())

    @docstringDecorator
    def AmericanMC(self):
        """
        4. use Monte-Carlo methods with regular GBM process and
        daily simulations on an American Call option. Remember to answer the different
        questions in the original GWP#1: price (Q8), calculate delta (Q9) and vega (Q10)
        only for the Call option case.
        5. Monte-Carlo methods with regular GBM process and
        daily simulations on an American Put option. Remember to answer the different
        questions in the original GWP#1: price (Q5), calculate delta (Q6) and vega (Q7)
        only for the Put option case

        GWP1:
        8. Repeat Q5, but this time consider options (call and put) of American style.
        (Answer sections a and b of Q5 as well)
        9. Repeat Q6, but considering American-style options. (Answer/comment on
        sections a and b of Q6 as well).
        10.Repeat Q7, but considering American-style options. (Answer/comment on
        sections a and b of Q7 as well).
        """
        bin_call, bin_put = self.callAndPutPrice(klass=AmericanOptionBinomial)
        mc_call_obj = AmericanOptionMC(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                       option_type="call")
        mc_put_obj = AmericanOptionMC(S0=self.S0, strike=self.K, time=0, expiry=self.T, rate=self.r, sigma=self.sigma,
                                      option_type="put")
        mc_call, mc_put = mc_call_obj.price(), mc_put_obj.price()
        df = pd.DataFrame(
            {"Q": [8, 8],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Amer", "Amer"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["MC", "MC"],
             "GWP1 Price": [bin_call, bin_put],
             "GWP2 Price": [mc_call, mc_put],
             }

        )
        self.prettifyComparisonDF(df)
        self.displayDataFrame(df)

        # delta
        mc_call_delta = mc_call_obj.delta()
        mc_put_delta = mc_put_obj.delta()
        bin_call_delta, bin_put_delta = self.deltaOfCallAndPut(klass=AmericanOptionBinomial)
        df = pd.DataFrame(
            {"Q": [9, 9],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Amer", "Amer"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["MC", "MC"],
             "GWP1 delta": [bin_call_delta, bin_put_delta],
             "GWP2 delta": [mc_call_delta, mc_put_delta],
             }

        )
        self.prettifyComparisonDF(df, column_suffix="delta")
        self.displayDataFrame(df)

        # vega
        mc_call_vega = mc_call_obj.vega(epsilon=0.05, multiplicative=False) / 100.0
        mc_put_vega = mc_put_obj.vega(epsilon=0.05, multiplicative=False) / 100.0
        bin_call_vega, bin_put_vega = self.vegaOfCallAndPut(klass=AmericanOptionBinomial)
        df = pd.DataFrame(
            {"Q": [10, 10],
             "Type": ["ATM Call", "ATM Put"],
             "Exer": ["Amer", "Amer"],
             "GWP1 Method": ["Binomial", "Binomial"],
             "GWP2 Method": ["MC", "MC"],
             "GWP1 vega": [bin_call_vega, bin_put_vega],
             "GWP2 vega": [mc_call_vega, mc_put_vega],
             }

        )
        self.prettifyComparisonDF(df, column_suffix="vega")
        self.displayDataFrame(df)

    @docstringDecorator
    def deltaHedging(self):
        """
         Team member A will work with European options with same characteristics as
        GWP#1 under different levels of moneyness:
        a. Price an European Call option with 110% moneyness and an European Put
        with 95%  moneyness using Black-Scholes. Both have 3 months maturity.
        b. You build a portfolio that buys the previous Call and Put options. What is
        the delta of the portfolio? How would you delta-hedge this portfolio?
        c. You build a second portfolio that buys the previous Call option and sells
        the Put. What is the delta of the portfolio? How would you delta-hedge this
        portfolio?
        """
        call_moneyness = 0.1
        put_moneyness = -0.05

        bs_call_obj = EuropeanOptionBS(S0=self.S0, strike=self.K * (1 + call_moneyness), time=0, expiry=self.T,
                                       rate=self.r,
                                       sigma=self.sigma,
                                       option_type="call")
        bs_put_obj = EuropeanOptionBS(S0=self.S0, strike=self.K * (1 + put_moneyness), time=0, expiry=self.T,
                                      rate=self.r,
                                      sigma=self.sigma,
                                      option_type="put")
        bs_call_price, bs_put_price = bs_call_obj.price(), bs_put_obj.price()

        trinomial_call = EuropeanOptionTrinomial(S0=self.S0, strike=self.K * (1 + call_moneyness), time=self.T,
                                                 rate=self.r, sigma=self.sigma, steps=2000,
                                                 option_type="call")
        trinomial_put = EuropeanOptionTrinomial(S0=self.S0, strike=self.K * (1 + put_moneyness), time=self.T,
                                                rate=self.r, sigma=self.sigma, steps=2000,
                                                option_type="put")
        trinomial_call_price, trinomial_put_price = trinomial_call.calculateOptionPrice(), trinomial_put.calculateOptionPrice()
        # TODO rename all calculateOptionPrice to price for consistency
        df = pd.DataFrame(
            {"Q": [15, 16],
             "Type": ["110% M call", "95% M put"],
             "Exer": ["Eur", "Eur"],
             "GWP1 Method": ["Trinomial", "Trinomial"],
             "GWP2 Method": ["BS", "BS"],
             "GWP1 Price": [trinomial_call_price, trinomial_put_price],
             "GWP2 Price": [bs_call_price, bs_put_price],
             }

        )
        self.prettifyComparisonDF(df)
        self.displayDataFrame(df)

        # delta
        bs_call_delta = bs_call_obj.delta()
        bs_put_delta = bs_put_obj.delta()
        # buy the bs call, and buy the bs put. what is the total delta?
        total_delta = bs_call_delta + bs_put_delta
        df = pd.DataFrame(
            {"call delta": ["{:3f}".format(bs_call_delta)],
             "put delta": ["{:3f}".format(bs_put_delta)],
             "total delta": ["{:3f}".format(total_delta)],
             "strategy": ["Buy {:.3f} units of the underlying asset".format(-total_delta)], }
        )
        self.displayDataFrame(df)

        #  buys the previous Call option and sells the Put.
        total_delta = bs_call_delta - bs_put_delta
        df = pd.DataFrame(
            {
             "call delta": ["{:3f}".format(bs_call_delta)],
             "put delta": ["{:3f}".format(bs_put_delta)],
             "total delta": ["{:3f}".format(total_delta)],

             "strategy": ["Sell {:.3f} units of the underlying asset".format(total_delta)], }
        )
        self.displayDataFrame(df)

    @docstringDecorator
    def upAndOutOption(self):
        """
        8. Team member B will work with Monte-Carlo methods with daily time steps to
        price an Up-and-Out (UAO)  barrier option. The option is currently ATM with a
        barrier level of 141 and:
        𝑆
        0

        = 120; 𝑟 = 6%; σ = 30%; 𝑇 = 8 𝑚𝑜𝑛𝑡ℎ𝑠
        :return:
        """
        S0 = 120
        rate = 0.06
        sigma = 0.3
        T = 8.0 / 12.0
        barrier = 141
        K = S0

        uao = UpAndOutEuropeanMC(S0=S0, rate=rate, sigma=sigma, time=0, expiry=T, barrier=barrier, strike=K, option_type="call")
        # print("price of Up and out European style call option is {:.2f}".format(uao.price()))

        uao_put = UpAndOutEuropeanMC(S0=S0, rate=rate, sigma=sigma, time=0, expiry=T, barrier=barrier, strike=K, option_type="put")
        # print("price of Up and out European style put option is {:.2f}".format(uao_put.price()))

        bs_call = EuropeanOptionBS(S0=S0, strike=K, time=0, expiry=T, rate=rate, sigma=sigma, option_type="call")
        bs_put = EuropeanOptionBS(S0=S0, strike=K, time=0, expiry=T, rate=rate, sigma=sigma, option_type="put")

        df = pd.DataFrame(
            {
                "Exer": ["Eur", "Eur"],
                "Type": ["Call", "Put"],
            "UAO price": [uao.price(), uao_put.price()],
                "BS Price": [ bs_call.price(), bs_put.price()],
            }
        )
        self.displayDataFrame(df)

if __name__ == '__main__':
    gwp2 = GWP2()
    # gwp2.EuropeanBS()
    # gwp2.EuropeanMC()
    # gwp2.AmericanMC()
    # gwp2.deltaHedging()
    gwp2.upAndOutOption()
    # gwp2.AmericanMCConvergence()
