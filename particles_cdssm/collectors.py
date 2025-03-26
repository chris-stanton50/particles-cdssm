import numpy as np

from particles.collectors import Collector, OnlineSmootherMixin
from particles.resampling import wmean_and_var
import particles.resampling as rs


class MultiOnlineSmootherMixin:
    """Mix-in for on-line smoothing algorithms. Extended to 
    act on multiple additive functions, which are provided 
    as an attribute to the fk object (fk.add_funcs), 
    which is a dictionary of additive functions.
    """

    def func_names(self, smc):
        return smc.fk.add_funcs.keys()

    def fetch(self, smc):
        if smc.t == 0:
            self.Phi = {name: add_func(0, None, smc.X) for name, add_func in smc.fk.add_funcs.items()}
        else:
            self.update(smc)
        out = {name: np.average(self.Phi[name], axis=0, weights=smc.W) for name in self.func_names(smc)}
        self.save_for_later(smc)
        return out

    def update(self, smc):
        """The part that varies from one (on-line smoothing) algorithm to the
        next goes here.
        """
        raise NotImplementedError

    def save_for_later(self, smc):
        """Save certain quantities that are required in the next iteration."""
        pass

class MultiOnline_smooth_naive(Collector, MultiOnlineSmootherMixin):
    def update(self, smc):
        self.Phi = {name: self.Phi[name][smc.A] + smc.fk.add_funcs[name](smc.t, smc.Xp, smc.X) for name in self.func_names(smc)}

class MultiOnline_smooth_ON2(Collector, MultiOnlineSmootherMixin):
    def update(self, smc):
        prev_Phi = {name: self.Phi[name].copy() for name in self.func_names(smc)}
        for n in range(smc.N):
            lwXn = self.prev_logw + smc.fk.logpt(smc.t, self.prev_X, smc.X[n])
            WXn = rs.exp_and_normalise(lwXn)
            for name in self.func_names(smc):
                self.Phi[name][n] = np.average(
                    prev_Phi[name] + smc.fk.add_funcs[name](smc.t, self.prev_X, smc.X[n]),
                    axis=0,
                    weights=WXn,
                )

    def save_for_later(self, smc):
        self.prev_X = smc.X
        self.prev_logw = smc.wgts.lw

class MultiOnline_smooth_mcmc(Collector, MultiOnlineSmootherMixin):
    """
    Implementation of online smoothing using MCMC steps 
    (Bunch and Godsill, 2014)/(Dau & Chopin 2022).
    
    Does online smoothing in O(N) time, without the need to 
    define an upper bound on logpt.
    
    Compatible with multiple additive functions.
    """
    
    signature = {"nsteps": 1}
    
    @property
    def nparis(self):
        return self.nsteps + 1
    
    def update(self, smc):
        prev_Phi = {name: self.Phi[name].copy() for name in self.func_names(smc)}
        xn = smc.X; prev_idx = smc.A
        for name in self.func_names(smc):
            self.Phi[name] = prev_Phi[name][prev_idx] + smc.fk.add_funcs[name](smc.t, self.prev_X[prev_idx], smc.X)
        for _ in range(self.nsteps):
            # IID version, otherwise introduces a bias!
            prop = rs.multinomial_iid(self.prev_w, M=smc.N) # (M, )
            lpr_acc = (smc.fk.logpt(smc.t, self.prev_X[prop], xn) # (N, )  
                        - smc.fk.logpt(smc.t, self.prev_X[prev_idx], xn))
            lu = np.log(np.random.rand(smc.N))
            prev_idx = np.where(lu < lpr_acc, prop, prev_idx)
            for name in self.func_names(smc):
                self.Phi[name] += prev_Phi[name][prev_idx] + smc.fk.add_funcs[name](smc.t, self.prev_X[prev_idx], smc.X)
        for name in self.func_names(smc):
            self.Phi[name] = self.Phi[name]/self.nparis
        
    def save_for_later(self, smc):
        self.prev_X = smc.X
        self.prev_w = smc.wgts.W