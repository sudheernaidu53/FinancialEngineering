import numpy.random as npr
import numpy as np


class IRModelBase(object):

    @classmethod
    def setSeed(cls, seed):
        npr.seed(seed)


class VasicekIRModel(IRModelBase):
    def __init__(self, r0, mean_rev_mult, theta, sigma, time, nb_steps=2000, nb_iters=100000):
        self.r0 = r0
        self.mean_rev_mult = mean_rev_mult
        self.theta = theta
        self.sigma = sigma
        self.time = time
        self.nb_steps = nb_steps
        self.nb_iters = nb_iters
        self._initDependencies()

    def _initDependencies(self):
        self.dt = self.time / self.nb_steps
        self.rates = None
        self.vol = self.sigma * np.sqrt(self.dt)

    def _dr(self, t):
        z = npr.randn(self.nb_iters)
        return self.mean_rev_mult * (self.theta - self.rates[t - 1]) * self.dt + self.vol * z

    def simulateRates(self):
        self.rates[0] = self.r0
        for t in range(1, self.nb_steps):
            self.rates[t] = self.rates[t - 1] + self._dr(t)

    def getRatesGrid(self):
        if self.rates is None:
            self.rates = np.zeros((self.nb_steps, self.nb_iters))
            self.simulateRates()
        return self.rates

    def getAvgIR(self):
        rates = self.getRatesGrid()
        return np.mean(rates[-1])

if __name__ == "__main__":
    vasicek = VasicekIRModel(
        r0=2.85,
        mean_rev_mult=0.12,
        theta=2.5,
        sigma=0.75,
        time=6/12.0,
        nb_steps=250,
        nb_iters=25

    )
    vasicek.setSeed(1)

    print(vasicek.getAvgIR())
    print(vasicek.getRatesGrid()[1])
