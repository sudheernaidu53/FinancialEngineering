"""American Option"""
from binomialPricing import (
    BinomialOptionPricing,
    plt,
    np,
)

class AmericanOptionBinomial(BinomialOptionPricing):
    def optionPriceAtNode(self, step, d_index, ):
        opt_price = super().optionPriceAtNode(step, d_index)
        return max(opt_price, self._payoff(self.underlying_price[step, d_index]))

if __name__ == "__main__":
    # a = AmericanOptionBinomial(45, 5, 5, 45, 0., 'call', 1.2, 1/1.2)
    a = AmericanOptionBinomial(45, 50, 50, 45, 0., 'put', 1.5, 1/1.5)
    a.calculateOptionPrice()
    print(a.underlying_price)
    print("Option Price: ", a.getOptionPrice())
    a.visualizeAllGrids()