"""European option"""
from  binomialPricing import BinomialOptionPricing, plt

class EuropeanOption(BinomialOptionPricing):
    pass

if __name__ == "__main__":
    a = EuropeanOption(100, 2, 2, 100, 0., 'call', 1.1, 0.9, yax_align="index")
    a.calculateOptionPrice()
    print(a.underlying_price)
    print("Option Price: ", a.getOptionPrice())
    a.visualizeAllGrids()
