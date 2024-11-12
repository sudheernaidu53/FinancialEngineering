import numpy as np
from abc import abstractmethod, ABC
import pandas as pd
import matplotlib.pyplot as plt
from mixins import  OptionMixin

class AbstractTreeOptionModel(ABC, OptionMixin):
    def __init__(self, strike, steps, time, S0, rate, option_type, up=None, sigma=None, yax_align="y"):
        """

        :param strike: strike of the option
        :param steps:how many steps in the process
        :param time: time to maturity
        :param S0: T0 underlying price
        :param rate: interest rate, risk free and constant
        :param option_type: "call" or "put"
        :param up: the multiplicative factor to be applied on underlying when it goes up
        :param sigma: the volatility of the underlying
        :param yax_align: when plotting the grid, how to align? "index" will keep the graph symmetric, by using matrix indices in cartesian coordinates
        "y" will just plot the price on the y axis, so it could be very skewed, but a good indicator of moneyness of option.
        """
        self.strike = strike
        self.time = time
        self.steps = steps
        self.S0 = S0
        self.rate = rate
        self.option_type = option_type
        self.yax_align = yax_align
        self.sigma = sigma
        self.up = up
        self.setDeltaT()
        self.setDiscount()
        self._initUpDown()
        self._validateOptionType()

    def setDiscount(self):
        self.discount = np.exp(-self.rate * self.delta_t)

    def setDeltaT(self):
        self.delta_t = self.time / self.steps

    @abstractmethod
    def _initUpDown(self):
        raise NotImplementedError

    def initSetup(self):
        self.resetGrids()
        self.calcRiskNeutralProbab()

    def fillOptionPricesAtExpiry(self):
        for i in range(len(self.underlying_price[self.steps])): #self.steps
            self.option_prices[self.steps, i] = self._payoff(self.underlying_price[self.steps, i])

    def setAttribute(self, attrib, value):
        if not hasattr(self, attrib):
            raise AttributeError(attrib + " is not an attribute of this object")
        self.__setattr__(attrib, value)
        if attrib == "sigma" :
            self.setUpDownFromSigma()
            self.calcRiskNeutralProbab()
        if attrib in [ "steps", "time"]:

            self.setDeltaT()
            self.setDiscount()
            self.calcRiskNeutralProbab()
            if self.sigma:
                self.setUpDownFromSigma()
                self.calcRiskNeutralProbab()
        if attrib in ["rate"]:
            self.setDiscount()
            self.calcRiskNeutralProbab()
        if attrib in ["up", "down"]:
            self.sigma = None
        self.resetGrids()

    def getOptionPrice(self):
        return self.option_prices[0, 0]


    def setUpDownFromSigma(self):
        self.up = None
        self.down = None
        self._initUpDown()

    @abstractmethod
    def calcRiskNeutralProbab(self):
        raise NotImplementedError

    @abstractmethod
    def resetGrids(self):
        raise NotImplementedError

    def calculateOptionPrice(self):
        self.initSetup()
        self.fillUnderlyingGrid()
        self.fillOptionPricesAtExpiry()
        self.fillOptionPricesRemaining()
        return self.getOptionPrice()

    @abstractmethod
    def fillUnderlyingGrid(self):
        raise NotImplementedError

    @abstractmethod
    def fillOptionPricesRemaining(self):
        raise NotImplementedError

    def visualizeOptionGrid(self):
        self.visualizeGrid(self.option_prices, "Option Price")

    def visualizeUnderlyingGrid(self):
        self.visualizeGrid(self.underlying_price, "Underlying Price")

    @abstractmethod
    def visualizeGrid(self, grid, label = None):
        raise NotImplementedError


class ObservableNumber:
    T0Price = "T0 Option Price"

    @classmethod
    def supported(cls):
        return [cls.T0Price]


class TreeOptionDependenceOnFactor(object):
    def __init__(self, option, parameter_name, val_range, to_plot=ObservableNumber.T0Price, plot_title=None):
        """

        :param option: instance of BinomialOptionModel
        :param parameter_name: a parameter that is part of object
        :param val_range: what values to tweak, iterable
        """
        assert to_plot in ObservableNumber.supported()
        self.option = option
        self.parameter_name = parameter_name
        self.val_range = val_range
        self.results = []
        self.to_plot = to_plot
        self.plot_title = plot_title or "{} as a function of {}".format(self.to_plot, self.parameter_name)

    def getANumber(self):
        if self.to_plot == ObservableNumber.T0Price:
            return self.option.calculateOptionPrice()
        raise NotImplementedError

    def tweakAndRecord(self):
        for v in self.val_range:
            self.option.setAttribute(self.parameter_name, v)
            self.results.append(self.getANumber())

    def plot(self):
        if not self.results:
            self.tweakAndRecord()
        plt.plot(self.val_range, self.results)
        plt.xlabel(self.parameter_name)
        plt.ylabel(self.to_plot)
        plt.title(self.plot_title)
        plt.grid(True)
        plt.show()

    def asDict(self, one2one=False):
        if not self.results:
            self.tweakAndRecord()
        return dict(zip(self.val_range, ["{:,.2f}".format(x) for x in self.results])) if one2one else {
            self.parameter_name: self.val_range, self.to_plot: self.results}

    def asDataFrame(self):
        return pd.DataFrame(self.asDict())