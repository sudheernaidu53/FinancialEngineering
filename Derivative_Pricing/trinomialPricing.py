import numpy as np
import matplotlib.pyplot as plt
from treePricing import AbstractTreeOptionModel


class TrinomialOptionModel(AbstractTreeOptionModel):
    # TODO: implement delta calculation


    def setUpDownFromSigma(self):
        self.up = None
        self.down = None
        self._initUpDown()

    def resetGrids(self):
        """
        first dimension is time steps, the second dimension the spawning of nodes. There are 2n+1 nodes on nth level.
        for 0:1
        for 1: u,m ,d
        for 2: u^2, um, m^2 = ud, dm, dd
        so, self.underlying_price[self.steps][2*self.steps+1] should give me the highest value of the underlying
        self.underlying_price[self.steps][0] should give me the lowest value of the underlying
        from grid[x][y] you would go an upstep to arive at grid[x+1][y+2], a middle step at grid[x+1][y+1] and a downstep at grid[x+1][y]
        """
        self.option_prices = np.zeros((self.steps + 1, 2* self.steps + 1))
        self.underlying_price = np.zeros((self.steps + 1, 2*self.steps + 1))
        # self.deltas = np.zeros((self.steps, self.steps))

    def _initUpDown(self):
        if self.up is None:
            self.up = np.exp(self.sigma * np.sqrt(2 * self.delta_t))
        assert self.up > 0.0, "self.up should be non negative"
        self.down = 1 / self.up
        assert self.down < self.up, "self.up <= 1. / self.up = down"

    def calcRiskNeutralProbab(self):
        self.pu = (
                          (
                                  np.exp(self.rate * self.delta_t / 2)
                                  - np.exp(-self.sigma * np.sqrt(self.delta_t / 2))
                          )
                          / (
                                  np.exp(self.sigma * np.sqrt(self.delta_t / 2))
                                  - np.exp(-self.sigma * np.sqrt(self.delta_t / 2))
                          )
                  ) ** 2
        self.pd = (
                          (
                                  -np.exp(self.rate * self.delta_t / 2)
                                  + np.exp(self.sigma * np.sqrt(self.delta_t / 2))
                          )
                          / (
                                  np.exp(self.sigma * np.sqrt(self.delta_t / 2))
                                  - np.exp(-self.sigma * np.sqrt(self.delta_t / 2))
                          )
                  ) ** 2
        self.pm = 1 - self.pu - self.pd

        assert 0 <= self.pu <= 1.0, "p_u should lie in [0, 1] given %s" % self.pu
        assert 0 <= self.pd <= 1.0, "p_d should lie in [0, 1] given %s" % self.pd
        assert 0 <= self.pm <= 1.0, "p_m should lie in [0, 1] given %s" % self.pm

    def fillUnderlyingGrid(self):
        for i in range(self.steps, -1, -1):
            self.underlying_price[i][:2*i+1] = self.fillUnderlyingVector(i)

    def optionPriceAtNode(self, step, d_index, ):
        """TODO. Except for the terminal nodes"""
        T1Up = self.option_prices[step + 1, d_index + 2]
        T1mid = self.option_prices[step + 1, d_index + 1]
        T1Down = self.option_prices[step + 1, d_index]
        opt_price = self.discount * (
                self.pu * T1Up + self.pm * T1mid + self.pd * T1Down)
        return opt_price

    def fillOptionPricesRemaining(self):
        for step in range(self.steps - 1, -1, -1):
            for d_index in range(2*step + 1):
                self.option_prices[step, d_index] = self.optionPriceAtNode(step, d_index)

    def fillUnderlyingVector(self, nb):
        """
        get the price of underlying at some particular layer. For ex: for up = 1.1 and down 0.9, with level 3 we get a vector of 7 values
        [0.9^3, 0.9^2, 0.9^1, 1.0, 1.1, 1.1^2, 1.1^3]
        :param nb: level 2nb + 1 nodes present in this level
        :return:
        """
        vec_u = self.up * np.ones(nb)
        np.cumprod(vec_u, out=vec_u)

        vec_d = self.down * np.ones(nb)
        np.cumprod(vec_d, out=vec_d)

        res = np.concatenate((vec_d[::-1], [1.0], vec_u))
        res *= self.S0

        return res

    def _payoff(self, stock_price, strike=None):
        if self.option_type == 'call':
            return max(0, stock_price - self.strike)
        elif self.option_type == 'put':
            return max(0, self.strike - stock_price)
        raise ValueError("option_type must be 'call' or 'put'")

    def visualizeGrid(self, grid, label=None):
        """
        Visualize grid of shape (self.steps+1, 2*self.steps+1)
        Draw a line from (x,y) to (x+1, y+2) in green, 
        Draw a line from (x,y) to (x+1, y+1) in blue,
        Draw a line from (x,y) to (x+1, y) in red
        
        also plot the value grid[x,y] at (x,y)
        :return:
        """
        fig, ax = plt.subplots()
        for x in range(self.steps +1 ):
            for k in range(2*x+1):
                if self.yax_align == "index":  # align y axis with index, symmetric graph, Cartesian plane
                    y = k-x
                elif self.yax_align == "y":  # align y axis with price
                    y = grid[x, k]
                else:
                    raise ValueError("yax_align must be 'index' or 'y'")
                ax.plot(x, y, 'bo')
                ax.text(x, y, f'{grid[x, k]:.2f}', ha='center', va='bottom', color='blue', fontsize=8)
                if x < self.steps:
                    y_up = y+1 if self.yax_align == "index" else grid[x+1, k+2]
                    y_mid = y if self.yax_align == "index" else grid[x+1, k+1]
                    y_down = y-1 if self.yax_align == "index" else grid[x+1, k]
                    ax.plot([x, x+1], [y, y_up],'g--', lw=0.5)
                    ax.plot([x, x+1], [y, y_mid], 'b--', lw=0.5)
                    ax.plot([x, x+1], [y, y_down], 'r--', lw=0.5)
        ax.set_title("Binomial Tree for grid" + label or "")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Price")
        ax.grid()
        return ax

    def visualizeAllGrids(self):
        self.visualizeGrid(self.underlying_price, "Underlying Price")
        self.visualizeGrid(self.option_prices, "Option Price")
        plt.show()


if __name__ == "__main__":
    a = TrinomialOptionModel(strike=90, steps=2, time=1.0, S0=100, rate=0.0,
                             option_type="call", up=None, sigma=0.3, yax_align="index")
    a.calculateOptionPrice()
    print(a.underlying_price)
    print("Option Price: ", a.getOptionPrice())
    a.visualizeAllGrids()