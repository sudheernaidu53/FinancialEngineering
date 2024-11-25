import numpy as np

class OptionMixin(object):

    def _validateOptionType(self):
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

    def _payoff(self, stock_price, strike=None):
        if self.option_type == 'call':
            return np.maximum(0, stock_price - self.strike)
        elif self.option_type == 'put':
            return np.maximum(0, self.strike - stock_price)
