import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from enum import Enum, unique
from treePricing import AbstractTreeOptionModel, TreeOptionDependenceOnFactor


# import matplotlib
# matplotlib.use('Agg')  # or 'TkAgg', 'Qt5Agg', depending on your installation

class BinomialOptionModel(AbstractTreeOptionModel):  # AmericanOptionBinomial
    def __init__(self, strike, steps, time, S0, rate, option_type, up=None, down=None, sigma=None, yax_align="y"):
        self.down = down
        super().__init__(strike, steps, time, S0, rate, option_type, up=up, sigma=sigma, yax_align=yax_align)

    def getAPath(self, path):
        assert len(path) == self.steps
        current_node = (0, 0)
        option_prices = [self.option_prices[current_node]]
        underlying_prices = [self.underlying_price[current_node]]
        deltas = [self.deltas[current_node]]

        for i, p in enumerate(path):
            if p == "u":
                current_node = (current_node[0] + 1, current_node[1] + 1)
            elif p == "d":
                current_node = (current_node[0] + 1, current_node[1])
            else:
                raise ValueError
            option_prices.append(self.option_prices[current_node])
            underlying_prices.append(self.underlying_price[current_node])
            if i < len(path) - 1:
                deltas.append(self.deltas[current_node])
            else:
                deltas.append(0.0)

        return pd.DataFrame({
            "time": list(range(self.steps + 1)),
            "option_price": option_prices,
            "underlying_price": underlying_prices,
            "delta": deltas,
            "previous step": [np.nan] + list(path)
        })

    def explainDeltaHedging(self, df):

        # df = {
        #     "time": [0, 1, 2, 3],
        #     "option_price": [13.82, 22.41, 34.63, 19.46],
        #     "underlying_price": [180.00, 162.54, 146.77, 162.54],
        #     "delta": [-0.47, -0.74, -1.00, 0.00],
        #     "previous step": ["", "d", "d", "u"],
        # }
        cash_accounts = []  # [-row['delta'] * row["underlying_price"]]
        stock_portfolios = []  # [row['delta'] * row["underlying_price"]]
        discounted_cash_accounts = []

        for i, row in df.iterrows():

            if i == 0:
                # Initial stock portfolio
                stock_portfolio = row['delta'] * row["underlying_price"]
                # Initial cash account from selling delta shares
                cash_account = -stock_portfolio
            elif i == len(df) - 1:
                # Final Step
                # Stock is just prev delta*curr stock price
                stock_portfolio = df.iloc[i - 1]['delta'] * row["underlying_price"]
                # Cash account is stock portfolio - option price (option buyer will take the payoff)
                cash_account = stock_portfolio - row["option_price"]
            else:
                # Adjust cash account by adding the new delta change in stock * stock price
                delta_change = df.iloc[i - 1]['delta'] - row['delta']
                cash_account = delta_change * row["underlying_price"]
                # stock portfolio is just previous stock portfolio - current cashflow
                stock_portfolio = stock_portfolios[-1] - cash_account
            stock_portfolios.append(stock_portfolio)
            cash_accounts.append(cash_account)

        df['Stock Portfolio'] = stock_portfolios
        df['Cash Account'] = cash_accounts
        df['Discounted Cash Account'] = df["Cash Account"] * (-self.discount**df["time"])

        print("Sum of Discounted Cash Accounts: {:,.2f} and the T0 option price: {:,.2f}".format(
            df["Discounted Cash Account"].sum(), df.loc[0]["option_price"]))
        return df

    def resetGrids(self):
        """It should be fine to not reset before you change a parameter and recalculate... as the calculation is done top down.
        but safer to reset for now

        first dimension is time steps, the second dimension the spawning
        so, self.underlying_price[self.steps][self.steps] should give me the highest value of the underlying
        self.underlying_price[self.steps][0] should give me the lowest value of the underlying
        from grid[x][y] you would go an upstep to arive at grid[x+1][y+1] and a downstep at grid[x+1][y]
        """
        self.option_prices = np.zeros((self.steps + 1, self.steps + 1))
        self.underlying_price = np.zeros((self.steps + 1, self.steps + 1))
        self.deltas = np.zeros((self.steps, self.steps))

    def _initUpDown(self):
        """
        when we scale up and down by the sigma and delta, we would get good convergence, else it'll be very dependent on the time steps
        """

        if self.sigma is not None:
            assert (self.up is None) and (self.down is None), "if sigma is provided, up and down must not be provided"
            self.up = np.exp(self.sigma * np.sqrt(self.delta_t))
            self.down = np.exp(-self.sigma * np.sqrt(self.delta_t))
            return

    def calcRiskNeutralProbab(self):
        """get risk neutral probability.. this will be used in calculation of option price with given future payoffs.
        C_t = e^{-rdt}[p c^u_{t+1} + (1-p) c^d_{t+1}]
        """
        self.risk_neutral_prob = (np.exp(self.rate * self.delta_t) - self.down) / (self.up - self.down)

    def fillUnderlyingGrid(self):
        for i in range(self.steps, -1, -1):
            for j in range(i + 1):
                try:
                    self.underlying_price[i, j] = self.S0 * (self.up ** j) * (self.down ** (i - j))
                except Exception as e:
                    print(e, i, j)
                    raise Exception

    def optionPriceAtNode(self, step, d_index, ):
        """Except for the terminal nodes"""
        T1Up = self.option_prices[step + 1, d_index + 1]
        T1Down = self.option_prices[step + 1, d_index]
        opt_price = self.discount * (self.risk_neutral_prob * T1Up + (1 - self.risk_neutral_prob) * T1Down)
        return opt_price

    def fillOptionPricesRemaining(self):
        for step in range(self.steps - 1, -1, -1):
            for i in range(step + 1):
                self.option_prices[step, i] = self.optionPriceAtNode(step, i)

    def fillDeltaGrid(self):
        """ delta is defined as  (up option price - down option price)/ (up underlying_prices- down underlying_prices)"""
        for step in range(self.steps - 1, -1, -1):
            for i in range(step + 1):
                self.deltas[step, i] = (self.option_prices[step + 1, i + 1] - self.option_prices[step + 1, i]) / (
                        self.underlying_price[step + 1, i + 1] - self.underlying_price[step + 1, i])

    def getT0Delta(self):
        return self.deltas[0, 0]

    def visualizeGrid(self, grid, label=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        len_ = len(grid) - 1  # use len_ instead of self.steps as delta grid could be smaller than underlying grid
        for x in range(len_ + 1):
            for k in range(x + 1):
                # self.yax_align == "index"  # Cartesian cooridnates. align y axis with index, symmetric graph
                # self.yax_align == "y":  # align y axis with price
                y = (2 * k - x) if self.yax_align == "index" else grid[x, k]
                ax.plot(x, y, 'bo')
                ax.text(x, y, f'{grid[x, k]:.2f}',
                        ha='center', va='bottom', color='blue', fontsize=8)
                if x < len_:
                    y_up = (y + 1) if self.yax_align == "index" else grid[x + 1, k + 1]
                    y_down = (y - 1) if self.yax_align == "index" else grid[x + 1, k]
                    ax.plot([x, x + 1], [y, y_up], 'g--', lw=0.5)
                    ax.plot([x, x + 1], [y, y_down], 'r--', lw=0.5)

        ax.set_title("Binomial Tree for " + label or "")
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
        self.fillDeltaGrid()
        self.visualizeDeltaGrid()
        plt.show()


if __name__ == "__main__":
    # a = AmericanOptionBinomial(45, 5, 5, 45, 0., 'call', 1.2, 1/1.2)
    a = BinomialOptionModel(45, 2, 2, 45, 0.,
                            'put', 1.5, 1 / 1.5, yax_align="index")
    a.calculateOptionPrice()
    print(a.underlying_price)
    print("Option Price: ", a.getOptionPrice())
    a.visualizeAllGrids()
    # b = TreeOptionDependenceOnFactor(a, 'strike', range(-10, 100, 5))
    # b.plot()
