import numpy as np
import functools

from particles.smoothing import ParticleHistory
import particles.smoothing as sm
import particles.resampling as rs

from particles_cdssm.feynman_kac import ReparameterisedDA

# method for a ParticleHistory object: generates samples by tracing back the ancestral path.
# We bind this to the ParticleHistory class, so that we can call it as a method.

def generate_hist_obj(option, smc):
    if option is True and hasattr(smc.fk, 'cdssm'):
        return CDSSM_ParticleHistory(smc.fk, smc.qmc)
    elif option is True:
        return ParticleHistory(smc.fk, smc.qmc)
    elif option is False:
        return None
    elif callable(option):
        return sm.PartialParticleHistory(option)
    elif isinstance(option, int) and option >= 0:
        return sm.RollingParticleHistory(option)
    else:
        raise ValueError("store_history: invalid option")
    
def post_transform(method):
    @functools.wraps(method)
    def post_transform_method(self, *args, **kwargs):
        out = method(self, *args, **kwargs)
        if isinstance(self.fk, ReparameterisedDA):
            out = self.fk.transform_W_to_X(out)
        return out
    return post_transform_method

class ParticleHistory(sm.ParticleHistory):
    """
    New version of `ParticleHistory` with additional methods for backward sampling.
    """
    def backward_sampling_genealogy(self, M):
        """
        Extract M full trajectories from the particle history.

        M final states are chosen randomly, then the corresponding trajectory
        is constructed backwards, until time t=0.
        """
        idx = self._init_backward_sampling(M)
        for t in reversed(range(self.T - 1)):
            idx[t, :] = self.A[t + 1][idx[t + 1, :]]
        return self._output_backward_sampling(idx)

    def backward_sampling_genealogy_idx(self, M, idx = None):
        """
        """
        idx = self._init_backward_sampling(M) if idx is None else idx.copy()
        for t in reversed(range(self.T - 1)):
            idx[t, :] = self.A[t + 1][idx[t + 1, :]]
        return idx

    def backward_sampling_ON2_idx(self, M, idx=None):
        """
        """
        idx = self._init_backward_sampling(M) if idx is None else idx.copy()
        for m in range(M):
            for t in reversed(range(self.T - 1)):
                lwm = self.wgts[t].lw + self.fk.logpt(
                    t + 1, self.X[t], self.X[t + 1][idx[t + 1, m]]
                )
                idx[t, m] = rs.multinomial_once(rs.exp_and_normalise(lwm))

        return idx

    def backward_sampling_mcmc_idx(self, M, nsteps=1, idx=None):
        """
        Extract M full trajectories from the particle history.

        M final states are chosen randomly, then the corresponding trajectory
        is constructed backwards, until time t=0.
        """
        idx = self._init_backward_sampling(M) if idx is None else idx.copy()
        for t in reversed(range(self.T - 1)):
            xn = self.X[t + 1][idx[t + 1, :]]
            idx[t, :] = self.A[t + 1][idx[t + 1, :]]
            for i in range(nsteps):
                # IID version, otherwise introduces a bias!
                prop = rs.multinomial_iid(self.wgts[t].W, M=M)
                lpr_acc = (self.fk.logpt(t + 1, self.X[t][prop], xn)
                            - self.fk.logpt(t + 1, self.X[t][idx[t, :]], xn))
                lu = np.log(np.random.rand(M))
                idx[t, :] = np.where(lu < lpr_acc, prop, idx[t, :])
        return idx

class CDSSM_ParticleHistory(ParticleHistory):
    """
    New version of ParticleHistory with additional methods for backward sampling.
    """

    @post_transform
    def extract_one_trajectory(self):
        return super().extract_one_trajectory()

    @post_transform
    def backward_sampling_genealogy(self, M):            
        return super().backward_sampling_genealogy(M)
    
    @post_transform
    def backward_sampling_ON2(self, M):
        return super().backward_sampling_ON2(M)
        
    @post_transform
    def backward_sampling_mcmc(self, M, nsteps=1):
        return super().backward_sampling_mcmc(M, nsteps)

    def _backward_sampling_ON2(self, M):
        """
        Generate trajectories without post-transform: used in MCMC algorithms (iCSMC-BS/PGBS)
        """
        return super().backward_sampling_ON2(M)
    
    def _extract_one_trajectory(self):
        """
        Generate trajectory without post-transform: used in MCMC algorithms (iCSMC-BS/PGBS)
        """
        return super().extract_one_trajectory()

    def backward_sampling_reject(self, M, max_trials=None):
        raise NotImplementedError("Method `backward_sampling_reject` not implemented for CDSSMs")
    
    def backward_sampling_qmc(self, M):
        raise NotImplementedError("Method `backward_sampling_qmc` not implemented for CDSSMs")
    
    def two_filter_smoothing(self, 
                             t,
                             info,
                             phi,
                             loggamma,
                             linear_cost=False,
                             return_ess=False,
                             modif_forward=None,
                             modif_info=None
                             ):
        raise NotImplementedError("Method `two_filter_smoothing` not implemented for CDSSMs")

# def add_hist_methods(hist):
#     hist.backward_sampling_geneaology = types.MethodType(backward_sampling_geneaology, hist)
#     hist.backward_sampling_geneaology_idx = types.MethodType(backward_sampling_geneaology_idx, hist)
#     hist.backward_sampling_ON2_idx = types.MethodType(backward_sampling_ON2_idx, hist)
#     hist.backward_sampling_mcmc_idx = types.MethodType(backward_sampling_mcmc_idx, hist)
