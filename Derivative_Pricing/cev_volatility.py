import numpy as np
from scipy.stats import ncx2
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class CEVVolatilityModel:
    """TODO: not tested yet
    """
    def __init__(self, strike, S0, time_to_maturity, rfr, sigma, beta, ):
        self.S0 = S0
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.rfr = rfr
        self.sigma = sigma
        self.beta = beta

    def price(self):
        zb = 2 + 2 / (2 - self.beta)
        kappa = 2 * self.rfr / (self.sigma ** 2 * (2 - self.beta) * (
                    np.exp(self.rfr * (2 - self.beta) * self.time_to_maturity) - 1))
        x = kappa * self.S0 ** (2 - self.beta) * np.exp(self.rfr * (2 - self.beta) * self.time_to_maturity)
        y = kappa * self.strike ** (2 - self.beta)
        return self.S0 * (1 - ncx2.cdf(2 * y, zb, 2 * x)) - self.strike * np.exp(-self.rfr * self.time_to_maturity) * (
            ncx2.cdf(2 * x, zb - 2, 2 * y)
        )

    def plotPrices(self, df_call):
        self.strike = df_call["Strike"]
        modelprices = self.price()
        realprices = df_call["Last Price"]
        plt.plot(self.strike, modelprices, "o", label="Model")
        plt.plot(self.strike, realprices, "o", label="Real")
        plt.xlabel("Stike")
        plt.ylabel("Option price")
        plt.legend()
        err = mean_squared_error(modelprices.values, realprices)
        print("Mean Squared Error is ", err)
