"""American Option"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # or 'TkAgg', 'Qt5Agg', depending on your installation

class AmericanOptionBinomial:
    def __init__(self, strike, steps,time, S0, rate,  option_type, up, down):
        self.strike = strike
        self.time = time
        self.steps = steps
        self.delta_t = time/steps
        self.S0 = S0
        self.rate = rate
        self.up = up
        self.down = down
        self.option_type = option_type

    def calcRiskNeutralProbab(self):
        self.risk_neutral_prob = (np.exp(-self.rate* self.delta_t) - self.down)/(self.up - self.down)

    def initSetup(self):
        """
        first dimension is time steps, the second dimension the spawning
        so, self.underlying_price[self.steps][self.steps] should give me the highest value of the underlying
        self.underlying_price[self.steps][0] should give me the lowest value of the underlying
        from matrix[x][y] you would go an upstep to arive at matrix[x+1][y+1] and a downstep at matrix[x+1][y]
        """
        self.option_prices = np.zeros((self.steps+1, self.steps+1))
        self.underlying_price = np.zeros((self.steps+1, self.steps+1))
        self.deltas = np.zeros((self.steps, self.steps))
        self.calcRiskNeutralProbab()

    def _payoff(self, stock_price, strike = None):
        if self.option_type == 'call':
            return max(0, stock_price - self.strike)
        elif self.option_type == 'put':
            return max(0, self.strike - stock_price)
        raise ValueError("option_type must be 'call' or 'put'")

    def fillUnerlyingGrid(self):
        for i in range(self.steps, -1, -1):
            for j in range(i+1):
                try:
                    self.underlying_price[i, j] = self.S0 * (self.up ** j) * (self.down ** (i - j))
                except Exception as e:
                    print(e, i, j)
                    raise Exception


    def _americanStep(self, option_price, stock_price):
        # print("option_price", option_price, "stock_price", stock_price)
        if self.option_type == 'call':
            return max(option_price, stock_price - self.strike)
        elif self.option_type == 'put':
            return max(option_price, self.strike - stock_price)
        raise ValueError("option_type must be 'call' or 'put'")


    def fillOptionPricesAtExpiry(self):
        for i in range(self.steps+1):
            self.option_prices[self.steps, i] = self._payoff(self.underlying_price[self.steps, i])

    def eurOptionPrice(self,  T1Up, T1Down,):
        return np.exp(-self.rate*self.delta_t) *(self.risk_neutral_prob*T1Up + (1-self.risk_neutral_prob)*T1Down )

    def fillOptionPricesRemaining(self):
        for t in range(self.steps-1, -1, -1):
            for i in range(t+1):
                self.option_prices[t, i] = self.eurOptionPrice(self.option_prices[t+1, i+1],
                                                               self.option_prices[t+1, i])
                self.option_prices[t, i] = self._americanStep(
                    self.option_prices[t, i], self.underlying_price[t, i])

    def fillDeltaGrid(self):
        """ delta is defined as  (up option price - down option price)/ (up underlying price- down underlying price)"""
        for step in range(self.steps-1, -1, -1):
            for i in range(step+1):
                self.deltas[step, i] = (self.option_prices[step+1, i+1] - self.option_prices[step+1, i]) / (self.underlying_price[step+1, i+1] - self.underlying_price[step+1, i])

    def calculateOptionPrice(self):
        self.initSetup()
        self.fillUnerlyingGrid()
        self.fillOptionPricesAtExpiry()
        self.fillOptionPricesRemaining()


    def getOptionPrice(self):
        return self.option_prices[0, 0]

    def visualizeGrid(self, matrix, label=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        len_ = len(matrix) -1
        # print(len_)
        # Plot stock pric   e tree
        for j in range(len_ + 1):
            for k in range(j + 1):
                x = j
                y = matrix[j, k]
                ax.plot(x, y, 'bo')
                ax.text(x, y, f'{matrix[j, k]:.2f}',
                        ha='center', va='bottom', color='blue', fontsize=8)
                if j < len_:
                    ax.plot([x, x + 1], [y, matrix[j + 1, k + 1]], 'b--', lw=0.5)
                    ax.plot([x, x + 1], [y, matrix[j + 1, k]], 'b--', lw=0.5)

        ax.set_title("Binomial Tree for matrix" + label or "")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Price")
        ax.grid()
        return ax

    def visualizeOptionGrid(self):
        self.visualizeGrid(self.option_prices, "Option Price")

    def visualizeUnderlyingGrid(self):
        self.visualizeGrid(self.underlying_price, "Underlying Price")

    def visualizeDeltaGrid(self):
        self.visualizeGrid(self.deltas, "Delta")

    def visualizeAllGrids(self):
        self.visualizeOptionGrid()
        self.visualizeUnderlyingGrid()
        plt.show()


if __name__ == "__main__":
    # a = AmericanOptionBinomial(45, 5, 5, 45, 0., 'call', 1.2, 1/1.2)
    a = AmericanOptionBinomial(45, 50, 50, 45, 0., 'put', 1.5, 1/1.5)
    a.calculateOptionPrice()
    print(a.underlying_price)
    print("Option Price: ", a.getOptionPrice())
    a.visualizeAllGrids()
    a.fillDeltaGrid()
    a.visualizeDeltaGrid()
    plt.show()