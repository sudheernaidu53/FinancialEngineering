from GWP2 import *
import numpy.random as npr
import numpy as np
from interestRate import VasicekIRModel


class GradedQuizM4(object):
    @classmethod
    def Q1(cls):
        strike = 42
        nb_iters = 4500
        S0 = 38.75
        sigma = 0.55
        rate = 0.01
        time = 0
        expiry = 5 / 12
        option_type = "call"
        EuropeanOptionMC.setSeed(42)
        mc = EuropeanOptionMC(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry, nb_iters=nb_iters)
        bs = EuropeanOptionBS(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        print(mc.price(), bs.price())
        # return mc.price()

    @classmethod
    def Q3(cls):
        strike = 27.5
        nb_iters = 4500
        S0 = 32.5
        sigma = 0.45
        rate = 0.0275
        time = 0
        expiry = 4 / 12
        option_type = "put"
        EuropeanOptionMC.setSeed(42)
        mc = EuropeanOptionMC(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry, nb_iters=nb_iters)
        bs = EuropeanOptionBS(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        print(mc.price(), bs.price())
        # return mc.price()

    @classmethod
    def Q4(cls):
        vasicek = VasicekIRModel(
            r0=2.85,
            mean_rev_mult=0.12,
            theta=2.5,
            sigma=0.75,
            time=6 / 12.0,
            nb_steps=250,
            nb_iters=25

        )
        vasicek.setSeed(1)

        print(vasicek.getAvgIR())
        print(vasicek.getRatesGrid()[1])

    @classmethod
    def Q6(cls):
        """
        what is the price tomorrow if we use 1 simulation with standard GBM
        assume there are 255 days in year
        """
        npr.seed(2)
        S0 = 385.0
        mu = 0.0725
        sigma = 0.34
        dt = 1.0 / 255.0
        Z = np.random.normal()
        St = S0 * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

        print(St)

    @classmethod
    def Q7(cls):
        S0 = 32.5
        strike = 32.5
        sigma = 0.35
        rate = 0.0275
        time = 0
        expiry = 5 / 12
        option_type = "call"
        EuropeanOptionMC.setSeed(42)
        mc = EuropeanOptionMC(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        bs = EuropeanOptionBS(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        print(mc.price(), bs.price())

    @classmethod
    def Q8(cls):
        VasicekIRModel.setSeed(42)
        vasicek = VasicekIRModel(
            r0=0.985,
            mean_rev_mult=0.22,
            theta=0.018,
            sigma=0.0175,
            time=12 / 12.0,
            nb_steps=250,
            nb_iters=1

        )

        print(vasicek.getAvgIR())
        print(vasicek.getRatesGrid()[1])

    @classmethod
    def Q11(cls):
        S0 = 32.5
        strike = 32.5
        sigma = 0.35
        rate = 0.0275
        time = 0
        expiry = 5 / 12
        option_type = "put"
        EuropeanOptionMC.setSeed(42)
        mc = EuropeanOptionMC(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        bs = EuropeanOptionBS(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        print(mc.price(), bs.price())

    @classmethod
    def Q12(cls):
        VasicekIRModel.setSeed(42)
        vasicek = VasicekIRModel(
            r0=2.85,
            mean_rev_mult=0.15,
            theta=0.0255,
            sigma=0.85,
            time=5 / 12.0,
            nb_steps=250,
            nb_iters=25

        )

        print(vasicek.getAvgIR())
        print(vasicek.getRatesGrid()[-1])
        print(max(vasicek.getRatesGrid()[-1]))
        print(vasicek.getRatesGrid().max().max())

    @classmethod
    def Q13(cls):
        strike = 122
        S0 = 118.75
        sigma = 0.25
        rate = 0.015
        time = 0
        expiry = 18 / 12
        option_type = "put"
        EuropeanOptionMC.setSeed(42)
        mc = EuropeanOptionMC(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry, nb_iters=5000)
        bs = EuropeanOptionBS(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        mcp, bsp = mc.price(), bs.price()
        print(mcp, bsp, mcp - bsp)
        # 14.578789975350213 14.750889429871933 0.2727028694989393

    @classmethod
    def Q15(cls):
        S0 =32.5
        strike = 35
        sigma = 0.45
        rate = 0.01
        time = 0
        expiry = 2 / 12
        option_type = "put"
        bs = EuropeanOptionBS(strike=strike, time=time, S0=S0, rate=rate, option_type=option_type, sigma=sigma,
                              expiry=expiry)
        print(bs.rho())


if __name__ == '__main__':
    # GradedQuizM4.Q1()
    # GradedQuizM4.Q3()
    # GradedQuizM4.Q6()
    # GradedQuizM4.Q7()
    # GradedQuizM4.Q8()
    # GradedQuizM4.Q11()
    # GradedQuizM4.Q12()
    # GradedQuizM4.Q13()

    GradedQuizM4.Q15()