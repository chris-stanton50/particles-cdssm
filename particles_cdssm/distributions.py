from particles.distributions import DiscreteDist

from numpy import random
from scipy import stats

class NegativeBinomial(DiscreteDist):
    """Negative Binomialp distribution.

    Parameters
    ----------
    n:  int, or array of ints
        number of failures untilp the experiment is run
    p:  float, or array of floats
        probability of success

    Note:
        Returns the distribution of the number of successes: support is
        0, 1, ...

    """

    def __init__(self, n=1, p=0.5):
        self.n = n
        self.p = p

    def rvs(self, size=None):
        return random.negative_binomial(self.n, self.p, size=size)

    def logpdf(self, x):
        return stats.nbinom.logpmf(x, self.n, self.p)

    def ppf(self, u):
        return stats.nbinom.ppf(u, self.n, self.p)