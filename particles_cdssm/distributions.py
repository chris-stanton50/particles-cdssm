from particles.distributions import DiscreteDist

from numpy import random
from scipy import stats

class NegativeBinomial(DiscreteDist):
    """Negative Binomial distribution.

    Parameters
    ----------
    n : int, or array of ints
        Number of successes at which the experiment is stopped.
    p : float, or array of floats
        Probability of success on each trial.

    Notes
    -----
    Returns the distribution of the number of failures before the
    `n`-th success. The support is 0, 1, ...

    The mean and variance are

        mean = n * (1 - p) / p
        variance = n * (1 - p) / p**2
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