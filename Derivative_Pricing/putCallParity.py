# aribtrage check
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # or 'TkAgg', 'Qt5Agg', depending on your installation
import matplotlib.pyplot as plt

# Parameters

put  =3
call =10
stock =70
bond = 64
# same K, T and underlying asset for put and call
# The face value is of the bond is the strike of the option.
# The maturity of the bond is the expiration of the option.
# Is there an arbitrage? If so, how many trades and how many exchanges are involved?

class PutCallParity:
    def __init__(self, call_price, put_price, stock_price, bond_price, interest_rate):
        self.call_price = call_price
        self.put_price = put_price
        self.stock_price = stock_price
        self.bond_price = bond_price
        self.interest_rate = interest_rate

    def check_parity(self):
        """
        Check if put-call parity holds
        C +B = S + P
        """
        lhs = self.call_price + self.bond_price
        rhs = self.stock_price + self.put_price

        print(f"Checking put-call parity:")
        print(f"LHS (Call + Bond) = {lhs}")
        print(f"RHS (Stock + Put) = {rhs}")

        if abs(lhs - rhs) < 1e-5:  # Small tolerance for floating point errors
            print("Put-call parity holds. No arbitrage opportunity.")
            return True
        else:
            print("Put-call parity does not hold. Arbitrage opportunity exists.")
            return False

    def arbitrage_strategy(self):
        """
        Suggest an arbitrage strategy if parity doesn't hold.
        The strategy follows:
        - If C + B > S + P: Sell call, sell bond, buy stock, buy put.
        - If C + B < S + P: Buy call, buy bond, sell stock, sell put.
        """
        if self.check_parity():
            print("No arbitrage strategy needed.")
            return

        # Determine the direction of the arbitrage
        lhs = self.call_price + self.bond_price
        rhs = self.stock_price + self.put_price

        if lhs > rhs:
            print("Arbitrage strategy: \n- Sell call, sell bond, buy stock, buy put.")
        else:
            print("Arbitrage strategy: \n- Buy call, buy bond, sell stock, sell put.")

    def payoffPutStock(self):
        strike_price = self.bond_price
        stock_prices = np.linspace(self.stock_price-strike_price, self.stock_price+strike_price, 100)  # Range of stock prices from 50 to 150

        # Payoff calculation
        put_payoff = np.maximum(strike_price - stock_prices, 0)  # Payoff from the put option
        total_payoff = stock_prices + put_payoff  # Total payoff from stock and put

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, total_payoff, label='Total Payoff (Stock + Put)', color='blue')
        plt.axhline(y=strike_price, color='red', linestyle='--', label='Payoff = K (Strike Price)')
        plt.title('Payoff of Stock + Put Option')
        plt.xlabel('Stock Price (S)')
        plt.ylabel('Total Payoff')
        plt.legend()
        plt.grid()
        plt.show()

    def payoffCallStrike(self):
        strike_price = self.bond_price
        stock_prices = np.linspace(self.stock_price-strike_price, self.stock_price+strike_price, 100)  # Range of stock prices from 50 to 150

        # Payoff calculation
        call_payoff = np.maximum(stock_prices - strike_price, 0)  # Payoff from the put option
        total_payoff = strike_price + call_payoff  # Total payoff from stock and put

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, total_payoff, label='Total Payoff (strike + Call)', color='blue')
        # plt.axhline(y=strike_price, color='red', linestyle='--', label='Payoff = K (Strike Price)')
        plt.title('Payoff of strike + Call Option')
        plt.xlabel('Stock Price (S)')
        plt.ylabel('Total Payoff')
        plt.legend()
        plt.grid()
        plt.show()


PutCallParity(call, put, stock, bond, 0.05).payoffCallStrike()
