"""European option"""
from binomialPricing import BinomialOptionModel
from trinomialPricing import TrinomialOptionModel
from closedForm import ClosedFormPricing
from MonteCarlo import MonteCarloPricing

class EuropeanOptionBinomial(BinomialOptionModel):
    pass


class EuropeanOptionTrinomial(TrinomialOptionModel):
    pass

class EuropeanOptionBS(ClosedFormPricing):
    pass

class EuropeanOptionMC(MonteCarloPricing):
    pass


if __name__ == "__main__":
    # a = EuropeanOptionBinomial(100, 2, 0.25, 100, 0.05, 'call', 1.1, 0.9, yax_align="index")
    a = EuropeanOptionBinomial(100, 2, 0.25, 100, 0.05, 'call', sigma=0.2, yax_align="index")
    a.calculateOptionPrice()
    print(a.underlying_price)
    print("Option Price: ", a.getOptionPrice())
    a.visualizeAllGrids()
