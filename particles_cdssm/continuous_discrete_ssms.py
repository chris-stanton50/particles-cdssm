"""
Module for Continuous-Discrete State Space Models (CD-SSMs). These are the key building blocks for the algorithms implemented in
this package, and represent the models on which we want to conduct statistical inference.

Implemented Classes:

- CDSSMBase: Abstract Base class for CD-SSMs

Univariate API (dimX=1, dimY=1)
--------------
- CDSSM:            For use when the model SDE is an instance of 'sdes.SDE'. This class uses a simplified API for when dimX=1 and dimY=1. 
                        Abstract class: Needs to be subclassed with 'PY' and optionally, 'LY' and 'SigmaY' and 'proposal0' methods defined.
- GaussianCDSSM:    A special case of the CDSSM where the observation density is additive Gaussian noise.

Multivariate API (dimX>1, dimY>1 and dimX = 1, dimY>1, dimX>1, dimY=1)
--------------
- MvCDSSM:          For use when the model SDE is an instance of 'sdes.MvSDE'.  
                    Abstract class: Needs to be subclassed with 'PY' and optionally, 'LY' and 'SigmaY' and 'proposal0' methods defined.
- MvGaussianCDSSM:  A special case of the CDSSM where the observation density is additive Gaussian noise.


A CDSSM is initialised (__init__) by:

- A Model SDE instance (sdes.SDE or sdes.MvSDE)
- A starting point/distribution: x0 (np.ndarray or dists.ProbDist)
- s_ts: The times at which the latent SDE is observed (float or array-like).
        If specified to be an array-like, the observations are assumed to be at non-equidistant times.
- The parameters of the observation density f_t(y_t|e_t): taken in as kwargs

For the univariate API (subclasses of CDSSM), x0 must be either a float or a dists.ProbDist instance of dim 1.
For the multivariate API (subclasses of MvCDSSM), x0 must be a np.ndarray of shape (1, dimX) or a dists.ProbDist instance of dimension dimX.

Building new CDSSMs: 

To create a CDSSM, subclass either CDSSM or MvCDSSM and define the following methods:

- 'PY' as a method that returns the observation density as particles.distributions.ProbDist object.
        currently assumes that this only depends on the end point of the latent process
        and not on the path.
- default observation density parameters in the class attribute/property 'default_params' 

And, optionally:
- Method 'LY' to define a proxy linear Gaussian 
- Method 'SigmaY' to define the square root of the covariance matrix of the observation density
- Method 'proposal0' to define the guided proposal distribution for the initial state x0 when it is unknown.
"""


"""
Additional feature: building Linear, Gaussian CD-SSMs: 

Feature for `GaussianCDSSM` and `MvGaussianCDSSM` classes:

The underlying discrete time state space model is a time homogeneous, Linear, Gaussian State Space Model (LGSSM)
When the following conditions hold: 

1. The model SDE is a Linear SDE
2. We have a Gaussian CDSSM (i.e Linear, Gaussian observation density)
3. We start with an inital point x0, or a Gaussian initial distribution for x0
4. Observation times are equidistant (ensures time homogeneity)

For CD-SSMs that satisfy these conditions, we can represent them as a LG-SSM, which 
enables one to use Kalman filtering and smoothing methods for exact inference.

The implementation of Kalman filtering/smoothing in the particles package requires
that the LG-SSM has no drift in the latent process. Thus, we must also check whether:

5. The drift term implied from the model SDE is zero. (sde._b(s, t) = 0)

The property `islgssm' checks conditions 1-5. 

Assuming that 1-5 hold, the `lgssm` method will construct the implied LG-SSM object 
that is implied by the CD-SSM. Standard particle filtering and smoothing methods can then be applied,
as-well as Kalman filtering and smoothing, to this state space model. 
"""

import numpy as np
import numpy.linalg as nla

from particles.state_space_models import StateSpaceModel
from particles.kalman import MeanAndCov, filter_step, MVLinearGauss, LinearGauss
import particles.distributions as dists

from sdes.sdes import OrnsteinUhlenbeck, MvOrnsteinUhlenbeck
from sdes.numerical_schemes import EulerMaruyama, MvEulerMaruyama

class CDSSMBase:

    def __init__(self, model_sde, x0=None, s_ts = 1., **kwargs):
        self.model_sde = model_sde
        self.x0 = self.default_x0 if x0 is None else x0
        self._set_s_ts(s_ts)
        StateSpaceModel.__init__(self, **{**kwargs, **self.sde_params})
        
    def _error_msg(self, method):
        return StateSpaceModel._error_msg(self, method)

    def _set_s_ts(self, s_ts):
        if type(s_ts) is float:
            assert s_ts > 0, 's_ts must be positive'
            self.s_ts = s_ts
        else:
            assert all([t >= 0. for t in s_ts]), 'All elements of s_ts must be positive'
            assert s_ts[0] == 0, 'First element of s_ts must be 0'
            self.s_ts = np.array(s_ts) # Covert to array if input is a list
        
    @property
    def dimX(self):
        return self.model_sde.dimX
    
    @property
    def dimY(self):
        return self.PY(0, None, self.default_x0).dim

    @property
    def isobservedat0(self):
        # If starting state is known, then we assume that we don't observe Y at time 0.
        # If the starting state is a distribution, then we assume that we observe Y at time 0.
        return True if isinstance(self.x0, dists.ProbDist) else False

    @property
    def isequidistant(self):
        return type(self.s_ts) is float
    
    @property
    def nobs(self):
        if self.isequidistant:
            return None
        else:
            return len(self.s_ts) if self.isobservedat0 else len(self.s_ts) - 1  
    
    @property
    def sde_params(self):
        return self.model_sde.params
    
    @property
    def obs_params(self):
        return {k: self.__dict__[k] for k in self.default_params.keys()}

    @property
    def params(self):
        return {**self.sde_params, **self.obs_params}

    @property
    def islgssm(self):
        return False
    
    def PY(self, t, xp, x):
        """
        Conditional distribution of Y_t, given the states at the end point E_t.
        """
        return self._error_msg("PY")
    
    def S(self, t):
        """
        Placeholder for evaluation of the observation times s_t: for now we assume that
        they are observed at equidistant times.
        """
        if self.isequidistant:
            return t * self.s_ts
        else:
            return self.s_ts[t]

    def _P0_sim(self, num=1000):
        if self.isobservedat0:
            P0_sim = self._init_dist_container(1)
            P0_sim[P0_sim.dtype.names[-1]] = self.x0.rvs()
        else:
            P0_sim = self.model_sde.simulate(1, self.x0, t_start=self.S(0), t_end=self.S(1), num=num)
        return P0_sim

    def _PX_sim(self, t, xp, num=1000):
        t = t-1 if self.isobservedat0 else t
        return self.model_sde.simulate(xp.shape[0], xp[xp.dtype.names[-1]], t_start=self.S(t), t_end=self.S(t+1), num=num)

    def simulate_given_x(self, x):
        lag_x = [None] + x[:-1]
        return [
            self.PY(t, xp, x[x.dtype.names[-1]]).rvs(size=1) for t, (xp, x) in enumerate(zip(lag_x, x))
        ]

    def simulate(self, T, num=1000):
        """Simulate state and observation processes.

        Parameters
        ----------
        T: int
            processes are simulated from time 0 to time T-1

        Returns
        -------
        x, y: lists
            lists of length T
        """
        x = [self._P0_sim(num=num)]
        for t in range(1, T):
            x.append(self._PX_sim(t, x[-1], num=num))
        y = self.simulate_given_x(x)
        return x, y
    
class CDSSM(CDSSMBase):
    """
    Subclass this CDSSM when we have a univariate SDE with a univariate observation density.
    """
    default_x0 = 0.

    def _init_dist_container(self, N):
        """
        Used to create initial state container in the following contexts:
            - SMC algorithms (so used in fk models)
            - Generating a simulation from a CDSSM.
        """
        dtype = [('0.0', 'float64')]
        return np.empty(N, dtype=dtype)

    def state_container(self, N, T, num, delta_s):
        """
        Used in initialising containers for MCMC algorithms.
        """
        shape = [N, T]
        numerical_scheme = EulerMaruyama(OrnsteinUhlenbeck()) # Dummy instance to access method
        state_container = numerical_scheme._create_state_container(delta_s, num, shape, dim=self.dimX)
        return state_container
        
    def LY(self, t):
        """
        Inputs
        ------------
        t: float

        Returns
        ------------
        LY: float
        """
        return self._error_msg(self, "LY")
    
    def SigmaY(self, t):
        """
        Inputs
        ------------
        t: float

        Returns
        ------------
        CovY: float        
        """
        return self._error_msg(self, "SigmaY")

    def proposal0(self):
        """
        Used as the proposal in guided filters when the initial state x0 is unknown
        (i.e defined through a non-degenerate distribution).
        
        Returns
        ------------
        Proposal: (Multi-dimensional) distribution object
        """
        return self._error_msg(self, "proposal0")

class GaussianCDSSM(CDSSM):

    default_params = {'sigmaY': 1.}
                
    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=self.sigmaY)

    def LY(self, t):
        return 1.

    def SigmaY(self, t):
        return self.sigmaY

    def proposal0(self, data):
        """
        Used as the proposal in guided filters when the initial state x0 is unknown
        (i.e defined through a non-degenerate distribution).
        
        Returns
        ------------
        Proposal: (One-dimensional) distribution object
        """
        if isinstance(self.x0, dists.Normal):
            sig2post = 1.0 / (1.0 / self.x0.scale ** 2 + 1.0 / self.sigmaY ** 2)
            mupost = sig2post * (data[0] / self.sigmaY ** 2)
            return dists.Normal(loc=mupost, scale=np.sqrt(sig2post))
        else:
            raise NotImplementedError("Proposal for x0 not implemented for this model when x0 is not instance of `dists.Normal`")

    @property
    def islgssm(self):
        if self.isobservedat0 and not isinstance(self.x0, dists.Normal):
            return False
        tol = 1e-7
        drift = self.model_sde._b(self.S(0), self.S(1))
        no_drift = drift >= -tol and drift <= tol
        islgssm = self.model_sde.isLinear and self.isequidistant and no_drift
        return islgssm
    
    def lgssm(self):
        if not self.islgssm:
            raise ValueError("This CD-SSM cannot be represented as a linear Gaussian state space model")
        s0, s1 = self.S(0), self.S(1)
        rho = self.model_sde._a(s0, s1); sigmaX = np.sqrt(self.model_sde._v(s0, s1))
        if self.isobservedat0:
            mu0 = self.x0.loc; sigma0 = self.x0.scale
        else:
            mu0 = rho * self.x0; sigma0 = sigmaX
        lg_params = {'rho': rho,
            'sigmaX': sigmaX,
            'mu0': mu0,
            'sigma0': sigma0, 
            'sigmaY': self.sigmaY
            }
        return LinearGauss(**lg_params)
        
class MvCDSSM(CDSSMBase):

    @property
    def default_x0(self):
        return np.zeros((1, self.dimX))
    
    def _init_dist_container(self, N):
        dtype = [('0.0', 'float64', self.dimX)]
        return np.empty(N, dtype=dtype)

    def state_container(self, N, T, num, delta_s):
        """
        Used in initialising containers for MCMC algorithms.
        """
        shape = [N, T]
        numerical_scheme = MvEulerMaruyama(MvOrnsteinUhlenbeck()) # Dummy instance to access method
        state_container = numerical_scheme._create_state_container(delta_s, num, shape, dim=self.dimX)
        return state_container
                
    def LY(self, t):
        """
        Inputs
        ------------
        t: float

        Returns
        ------------
        LY: (dimY, dimX) array       
        """
        return self._error_msg(self, "LY")

    def SigmaY(self, t):
        """
        Inputs
        ------------
        t: float

        Returns
        ------------
        CovY: (dimY, dimY) array        
        """
        return self._error_msg(self, "SigmaY")

    def proposal0(self):
        """
        Used as the proposal in guided filters when the initial state x0 is unknown
        (i.e defined through a non-degenerate distribution).
        
        Returns
        ------------
        Proposal: (Multi-dimensional) distribution object
        """
        return self._error_msg(self, "proposal0")
            
class MvGaussianCDSSM(MvCDSSM):
    """
    Gaussian CDSSM that involves obsersations with Gaussian distributed noise:
    Dimension of observations is by default set to match that of the latent states.
    
    Y_t | E_t=e_t \sim N_d(Ge_t, \Cov_Y).
        
    Takes the following parameters as input:

    G: (dimY, dimX) array
    covY: (dimY, dimY) array
    
    The dimension of Y is then inferred from the inputs. 
    If these parameters are not provided, default behaviour is to observe 
    each component of the latent process with independent, additive noise.
    """
    @property
    def default_params(self):
        return {'G': np.eye(self.dimX), 'covY': np.eye(self.dimX)}

    def __init__(self, model_sde, x0=None, s_ts = 1., **kwargs):
        CDSSMBase.__init__(self, model_sde, x0, s_ts, **kwargs)
        self._check_L_CovY_dims(self.G, self.covY)
        self.sigmaY = nla.cholesky(self.covY)

    def PY(self, t, xp, x):
        return dists.MvNormal(loc=x @ self.G.T, cov=self.covY)
    
    def LY(self, t):
        return self.G
        
    def SigmaY(self, t):
        return self.sigmaY
    
    # def gen_score_add_func(self, param_name):
    #     gplpx = self.model_sde.grad_param_log_px
    #     @use_end_point
    #     def add_func(t, x, xf):
    #         if t == 0:
    #             out = gplpx(self.S(0), self.S(1), self.x0, x, param_name)
    #             out += gplpx(self.S(1), self.S(2), x, xf, param_name)
    #         else:
    #             out = gplpx(self.S(t+1), self.S(t+2), x, xf, param_name)
    #             return out
    #     return add_func

    def _check_L_CovY_dims(self, G, covY):
        """
        TO DO: Re-write this - it is not very readable.
        """
        if covY.shape[0] != G.shape[0]:
            raise ValueError("Dimension mismatch between parameters G and CovY")
        if covY.shape[1] != covY.shape[0]:
            raise ValueError("CovY is not a square matrix")
        if G.shape[1] != self.dimX:
            raise ValueError("Second dimension of G must match dimension of model SDE")

    def proposal0(self, data):
        """
        Used as the proposal in guided filters when the initial state x0 is unknown
        (i.e defined through a non-degenerate distribution).
        
        Returns
        ------------
        Proposal: (Multi-dimensional) distribution object
        """
        if isinstance(self.x0, dists.MvNormal):
            mu0 = self.x0.loc; cov0 = self.x0.cov * (self.x0.scale ** 2)
            pred0 = MeanAndCov(mean=mu0, cov=cov0)
            f, _ = filter_step(self.G, self.covY, pred0, data[0])
            return dists.MvNormal(loc=f.mean, cov=f.cov)
        else:
            raise NotImplementedError("Proposal for x0 not implemented for this model when x0 is not instance of `dists.MvNormal`")

    @property
    def islgssm(self):
        if self.isobservedat0 and not isinstance(self.x0, dists.MvNormal):
            return False
        drift = self.model_sde._b(self.S(0), self.S(1)) # (1, dimX)
        no_drift = np.all(np.isclose(drift, np.zeros_like(drift), atol=1e-7))
        islgssm = self.model_sde.isLinear and self.isequidistant and no_drift
        return islgssm

    def lgssm(self):
        if not self.islgssm:
            raise ValueError("This CD-SSM cannot be represented as a linear Gaussian state space model")
        s0, s1 = self.S(0), self.S(1)
        F = self.model_sde._a(s0, s1)[0]; covX = self.model_sde._v(s0, s1)[0] # (dimX, dimX), (dimX, dimX)
        if self.isobservedat0:
            mu0 = self.x0.loc.ravel(); cov0 = self.x0.cov * (self.x0.scale ** 2)
        else:
            mu0 = np.dot(F, self.x0[0]); cov0 = covX
        mvlg_params = {'F': F,
            'covX': covX, # (dimX, dimX)
            'mu0': mu0, # (dimX,)
            'cov0': cov0, # (dimX, dimX)
            'G': self.G, # (dimY, dimX)
            'covY': self.covY # (dimY, dimY)
            }
        return MVLinearGauss(**mvlg_params)
            
            
"""
The following classes are now deprecated
"""

# class TimeSwitchingGaussianCDSSM(MvGaussianCDSSM):   
#     """
#     Not the most helpful CDSSM, but useful for testing purposes.
#     This CDSSM is linear Gaussian but time inhomogeneous, so we use 
#     'DisreteDiscreteSSM' as the discrete time proxy.
#     """    
#     @property
#     def default_params(self):
#         if self.dimX > 1:
#             def_params = {'G_1': np.eye(self.dimX), 'G_2': np.eye(self.dimX), 
#                           'covY_1': np.eye(self.dimX), 'covY_2': 0.1*np.eye(self.dimX), 't_switchY': 10}
#         else:
#             def_params = {'G_1': 1., 'covY_1': 1., 'G_2': 1., 'covY_2': 0.1*1., 't_switchY': 10}
#         return def_params
 
#     def __init__(self, model_sde, x0=None, delta_s = 1., **kwargs):
#         CDSSM.__init__(self, model_sde, x0, delta_s, **kwargs)
#         self._check_L_CovY_dims(self.G_1, self.covY_1)
#         self._check_L_CovY_dims(self.G_2, self.covY_2)
#         if type(self.G_1) is np.ndarray:
#             assert self.G_1.shape == self.G_2.shape, 'Shapes of G_1 and G_2 do not match'
#         if type(self.covY_1) is np.ndarray:
#             assert self.covY_1.shape == self.covY_2.shape, 'Shapes of CovY_1 and CovY_2 do not match'

#     @property
#     def dimY(self):
#         return 1 if type(self.covY_1) == float else self.covY_1.shape[0]

#     def LY(self, t):
#         t = t+1
#         return self.G_1 if t < self.t_switchY else self.G_2
    
#     def CovY(self, t):
#         t = t+1
#         return self.covY_1 if t < self.t_switchY else self.covY_2

#     def discrete_ssm(self):
#         """
#         Returns a state space model that is a proxy for the CDSSM.
#         Only possible when the latent SDE is Linear, so that the transition density is tractable.
#         """
#         if self.model_sde.isLinear:
#                 discrete_ssm = DiscreteDiscreteSSM(self)
#                 return discrete_ssm
#         else: # There are edge case where the SDE has non-Gaussian transition density to think about in the future.
#             return None
        
# class OU_CDSSM(GaussianCDSSM):

#     BenchmarkSSMCls = LinearGauss

#     @property
#     def benchmark_ssm_params(self):
#         """
#         If the model SDE of a standard CDSSM is a linear SDE that can be solved analytically, one has
#         access to the transition density of the diffusion, that is linear Gaussian. Thus, one can construct 
#         a LGSSM. 
#         """
#         linear_gauss_params = {'sigmaY': self.eta,
#                        'rho': self.model_sde._a(self.S(0), self.S(1)),
#                        'sigmaX': np.sqrt(self.model_sde._v(self.S(0), self.S(1))), # Assume equidistant observations for now
#                        'sigma0': np.sqrt(self.model_sde._v(self.S(0), self.S(1)))
#                       }
#         return linear_gauss_params

# class Reparam_OU_CDSSM(OU_CDSSM):

#     default_params = {'rho': 0.8187307530779818, 'sigmaX_2': 0.07417798964198115}
#     # Corresponds to rho=0.3, mu=0., phi=0.3, eta_sq=0.01**2 in OU_CDSSM
    
#     def __init__(self, x0=0., delta_s = 1., **kwargs):
#         StateSpaceModel.__init__(self, **kwargs)
#         self.x0 = x0
#         self.delta_s = delta_s
#         model_sde_params = {'rho': -np.log(self.rho), 'mu': 0., 'phi': self.phi()}
#         self.model_sde = self.ModelSDECls(**model_sde_params)
#         self.eta_sq = 0.01 ** 2
    
#     def phi(self):
#         return np.sqrt((-2.*np.log(self.rho) * self.sigmaX_2)/(1-self.rho**2))