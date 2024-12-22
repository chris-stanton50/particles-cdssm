"""
Auxiliary Bridges module:
--------------------------
Within this module, we implement abstractions of the different choices 
of auxiliary bridges that can be used as valid proposals for the true diffusion bridge

to construct an invertible mapping from the bridge 
of an SDE conditional on the starting point and the end point, to a sample path
which has distribution that is absolutely integrable with respect to the parameter
free Weiner measure.

The original bridge construction given in Yonekura and Beskos (2022) is the auxiliary
bridge process of Delyon and Hu (2006). Additional choices of auxiliary bridge process
proposed in Schauer, van der Meulen and Van Zanten (2017) along with proofs of their
equivalence to the true diffusion bridge. 


Univariate Case
-----------------

Forward Proposals: 4 Brownian Proposals, 2 OU Proposals
-----------------

NoDriftBasicBrownianProp: No Drift, diffusion 1.
NoDriftBrownianProp: No drift, diffusion \sigma(t_start, x_start)
DriftBasicBrownianProp: Drift b(t_start, x_start), diffusion 1.
DriftBrownianProp: Drift b(t_start, x_start), diffusion \sigma(t_start, x_start)

LocalLinearBasicOUProp: Drift local linearising b at t_start, x_start, diffusion coefficient as 1.
LocalLinearOUProp: Drift local linearising b at t_start, x_start, diffusion coefficient as \sigma(t_start, x_start)

Auxiliary Bridges: 2 Delyon Hu Bridges, 2 Brownian Aux Bridges, 1 OU Aux Bridge
-----------------
DelyonHuAuxBridge: The auxiliary bridge as proposed by Delyon and Hu (2006)
DriftDelyonHuAuxBridge: The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients. (Not implemented yet)

(subclasses of VanDerMeulenSchauerAuxBridge)

NoDriftBrownianAuxBridge: Drift: 0., diffusion \sigma(t_end, x_end) (matching condition)
DriftBrownianAuxBridge: Drift: b(t_end, x_end), diffusion \sigma(t_end, x_end) (matching condition)

LocalLinearOUAuxBridge: Drift: Local linearising b at t_end, x_end, diffusion \sigma(t_end, x_end) (matching condition)

Multivariate Case
-----------------

Forward Proposals: 6 Brownian Proposals, 2 OU Proposals
-----------------

MvNoDriftBasicBrownianProp: Drift: 0.*1_d, diffusion I_d 
MvDriftBasicBrownianProp: Drift: b(t_start, x_start), diffusion I_d
MvNoDriftIndepBrownianProp: Drift: b(t_start, x_start), diffusion diag(\sigma(t_start, x_start))
MvNoDriftBrownianProp: Drift: b(t_start, x_start), diffusion \sigma(t_start, x_start)
MvDriftIndepBrownianProp: Drift: b(t_start, x_start), diffusion diag(\sigma(t_start, x_start))
MvDriftBrownianProp: Drift: b(t_start, x_start), diffusion \sigma(t_start, x_start)

MvOUProposal: Drift: Local linearising b at t_start, x_start, taking only diagonal components of B(t) matrix, diffusion \sigma(t_start, x_start)
MvIndepOUProposal: Local linearising b at t_start, x_start, taking only diagonal components of B(t) matrix, diffusion \diag(\sigma(t_start, x_start))

Auxiliary Bridges: 2 Delyon Hu bridges, 2 Brownian Aux Bridges, 1 OU Aux Bridge
-----------------

MvDelyonHuAuxBridge: The auxiliary bridge as proposed by Delyon and Hu (2006)
MvDriftDelyonHuAuxBridge: The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients. (Not implemented yet)

(subclasses of MvVanDerMeulenSchauerAuxBridge)
MvNoDriftBrownianAuxBridge: Auxiliary bridge proposal that takes the Brownian motion with 0 drift as the linear SDE
MvDriftBrownianAuxBridge: Auxiliary bridge that takes the Brownian motion as the linear SDE

MvOUAuxBridge: Auxiliary bridge that takes the OU process as the linear SDE.

Hypoelliptic Case:
-----------------

Forward Proposals: Forward Proposals are possible for the hypoelliptic case, but only for filtering and smoothing algorithms that do not change ancestors.
-----------------

Currently not implemented.


Auxiliary Bridges: 4 Brownian Aux Bridges, 4 OU Aux Bridges
-----------------

IntegratedDriftBrownianAuxBridge: Auxiliary bridge that uses an integrated Brownian motion as the proxy.
IntegratedNoDriftBrownianAuxBridge: Auxiliary bridge that uses an integrated Brownian motion with 0 drift, as the proxy.
TwiceIntegratedDriftBrownianAuxBridge: Auxiliary bridge that uses an integrated Brownian motion as the proxy.
TwiceIntegratedNoDriftBrownianAuxBridge: Auxiliary bridge that uses an integrated Brownian motion with 0 drift, diagonal diffusion covariance as the proxy.  

IntegratedDriftLLOUAuxBridge: Auxiliary Bridge that uses an integrated O-U process as the proxy.
IntegratedNoDriftLLOUAuxBridge: Auxiliary Bridge that uses an integrated O-U process as the proxy.
TwiceIntegratedDriftLLOUAuxBridge: Auxiliary Bridge that uses a twice integrated O-U process as the proxy.
TwiceIntegratedNoDriftLLOUAuxBridge: Auxiliary Bridge that uses a twice integrated O-U process as the proxy.


End Point Proposals:
-----------------

MvNaiveEndPointProposal: E_t | E_{t-1} = e_{t-1} ~ N(0, (t-s)I_d)
MvEulerMaruyamaEndPointProposal: E_t | E_{t-1} = e_{t-1} ~ N(e_{t-1} + b(0, e_{t-1)), (t-s)\sigma(0, e_{t-1}))
MvOUEndPointProposal: Transition of an OU process with diagonal elements in B matrix.

IntegratedDriftBrownianEndPointProposal:
IntegratedOUEndPointProposal
    
IntegratedDriftBrownianEndPointProposal:
IntegratedOUEndPointProposal:

"""

import numpy as np
import numpy.linalg as nla
import scipy.stats as stats
from sdes.sdes import SDE, MvSDE, MvEllipticSDE, HypoellipticSDE, BrownianMotion, OrnsteinUhlenbeck, MvIndepBrownianMotion, MvBrownianMotion, MvIndepOrnsteinUhlenbeck, MvOrnsteinUhlenbeck, TimeSwitchingSDE
from sdes.sdes import HypoellipticSDE, IntegratedIndepBrownianMotion, IntegratedBrownianMotion, IntegratedIndepOrnsteinUhlenbeck, IntegratedOrnsteinUhlenbeck
from sdes.sdes import TwiceIntegratedIndepBrownianMotion, TwiceIntegratedBrownianMotion, TwiceIntegratedIndepOrnsteinUhlenbeck, TwiceIntegratedOrnsteinUhlenbeck
from sdes.path_integrals import log_girsanov, log_delyon_hu, log_drift_delyon_hu, log_van_der_meulen_schauer, mv_log_girsanov, mv_log_delyon_hu, mv_log_van_der_meulen_schauer
from sdes.tools import log_abs_det, filter_step_var_cov, MeanAndCov
from particles.distributions import ProbDist, Normal, VaryingCovNormal




# -----------------LinearSDE Proposal Classes -----------------

"""
Many of the auxiliary bridge (in particular, the VanDerMeulenSchauerAuxBridge) and forward proposals are based on the construction of a linear SDE that
approximates the true SDE. For forward proposals, this is done by approximating the drift and diffusion coefficients with a Taylor expansion around the starting point.

- For forward proposals, we locally linearlise around the starting point, and for auxiliary bridges, we locally linearlise around the end point.

"""

tol=1e-7

def add_tol(t_switch, func):
    def tol_func(*args, **kwargs):
        new_args = list(args)
        t = new_args[0] # Assume that t is the first argument
        if t == t_switch:
            new_args[0] -= tol
        return func(*new_args, **kwargs)
    return tol_func

class LinearSDEProposal(object):
    """
    Class that is used to add methods to the following classes:
    
    - ForwardProposal    
    - VanDerMeulenSchauerAuxBridge
    - MvEllipticForwardProposal
    - MvVanDerMeulenSchauerAuxBridge
    - LinearEndPointProposal    
        
    We need to specify the following attributes:

    any_cov (Forward proposals only)
    full_cov (multivariate Forward Proposals only)
    drift (Brownian proposals only)
    
    Due to the matching condition, the covariance matrix is predetermined by the diffusion of the signal at the end points
    in the case of auxiliary bridges.
    """   
    def _sde_tol_dec(self, func):
        isforwardprop = isinstance(self.SDE, ForwardProposal)
        sde = self.SDE.SDE if isforwardprop else self.SDE
        if isinstance(sde, TimeSwitchingSDE):
            if not isforwardprop:
                return add_tol(sde.t_switch, func)
            elif isforwardprop and self.SDE.t_end == sde.t_switch:
                return add_tol(self.t_end, func)
            else:
                return func 
        else:
            return func

    def _self_tol_dec(self, func):
        if isinstance(self.SDE, TimeSwitchingSDE) and self.t_end == self.SDE.t_switch:
            return add_tol(self.t_diff, func)
        else:
            return func

    def get_linearising_points(self):
        if isinstance(self, ForwardProposal) or isinstance(self, LinearEndPointProposal):
            return self.t_start, self.x_start
        # if isinstance(self, LinearSDEForwardProposal):
        #     return self.t_start, self.x_start
        # if isinstance(self, AdaptiveForwardProposal):
        #     return self.t_curr, self.x_curr
        if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            return self.t_end, self.x_end
        if isinstance(self, AuxiliaryBridge):
            return self.t_end - tol, self.x_end

    def get_linear_sde_params(self):
        t, x = self.get_linearising_points()
        linearising_functions = self.take_rough_component_dec(self.get_linearising_functions()) if isinstance(self, HypoellipticLinearSDEProposal) else self.get_linearising_functions()
        linear_sde_params = {param: func(t, x) for param, func in linearising_functions.items()}
        if isinstance(self.SDE, MvForwardProposal) or isinstance(self.SDE, MvVanDerMeulenSchauerAuxBridge):
            linear_sde_params.update({'N': self.N, 'dimX': self.dimX})
        linear_sde_params = self.check_linear_sde_params(linear_sde_params)
        self.check_matching_condition(linear_sde_params)
        return linear_sde_params

    def check_linear_sde_params(self, linear_sde_params):
        return self.check_drift_param(self.check_diffusion_param(linear_sde_params))

    def check_diffusion_param(self, linear_sde_params):
        par = self.diffusion_param_name
        diffusion = linear_sde_params[par]
        if isinstance(self, ForwardProposal) or isinstance(self, LinearEndPointProposal): 
            if not self.any_cov:
                del linear_sde_params[par]
                return linear_sde_params
            if isinstance(self, MvForwardProposal) or isinstance(self, MvLinearEndPointProposal) and not self.full_cov:
                linear_sde_params[par] = self._diffusion_to_diag(diffusion)
        if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            if self._diag_cov:
                linear_sde_params[par] = self._diffusion_to_diag(diffusion)
      
    def check_matching_condition(self, linear_sde_params):
        par = self.diffusion_param_name
        if isinstance(self, VanDerMeulenSchauerAuxBridge):
            matching_condition = np.all(np.isclose(self.sigma_x_end, linear_sde_params[par]))
            if not matching_condition:
                raise ValueError('Matching condition for bridge proposal not satisfied.')
            
    def _diffusion_to_diag(self, diffusion):
        N = self.N
        return np.stack([np.diag(diffusion[i]) for i in range(N)], axis=0) if N > 1 else np.diag(diffusion[0]).reshape(N, self.dimX)

    def build_linear_sde(self):
        self.LinearSDE = self.LinearSDECls(**self.get_linear_sde_params())

class EllipticLinearSDEProposal(LinearSDEProposal):

    @property
    def LinearSDECls(self):
        if not isinstance(self.SDE, MvSDE):
            return self.UnivLinearSDECls
        else:
            if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
                return self.MvIndepLinearSDECls if self._diag_cov else self.MvLinearSDECls
            else:
                return self.MvIndepLinearSDECls if not self.full_cov else self.MvLinearSDECls

class HypoellipticLinearSDEProposal(LinearSDEProposal):

    @property
    def LinearSDECls(self):
        if not isinstance(self.SDE, MvSDE):
            raise ValueError('Proposals based on hypoelliptic SDEs only exist multivariate SDEs.')
        else:
            if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
                return self.MvIndepLinearSDECls if self._diag_cov else self.MvLinearSDECls
            else:
                raise ValueError('Forward proposals for hypoelliptic SDEs not implemented.')

    def _rough_comp_dec(self, func):
        # Possible shapes for MV output: (N, dx, dx), (N, dx), (dx, dx)
        def rough_comp_func(t, x):
            dw = self.dimW; dx = self.dimX; N = self.N
            out = func(t, x)
            if out.shape == (dx, dw): # (dx, dw) -> # (N, dw)
                return out[-dw:, :]
            elif out.shape == (N, dx, dw): # (N, dx, dw) -> # (N, dw, dw)
                return out[:, -dw:, :]
            elif out.shape == (N, dx):
                return out[:, -dw:] # (N, dx) -> # (N, dw)
            else:
                return out
        return rough_comp_func

    def take_rough_component_dec(self, linearising_functions):
        return {key: self._rough_comp_dec(func) for key, func in linearising_functions.items()}

class BrownianProposal(LinearSDEProposal):
        
    @property
    def diffusion_param_name(self):
        return 's'

    def get_linearising_functions(self):
        # m_func = self._b_time_shifted if isinstance(self.SDE, AdaptiveForwardProposal) else self.SDE.b
        m_func = self.SDE.b
        s_func = self.SDE.sigma
        if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            m_func = self._sde_tol_dec(m_func)
            s_func = self._sde_tol_dec(s_func)
        return {'m': m_func, 's': s_func}

    def check_drift_param(self, linear_sde_params):
        if self.drift == False:
            linear_sde_params['m'] = np.zeros_like(linear_sde_params['m']) if isinstance(self.SDE, MvSDE) else 0.
        return linear_sde_params

class OUProposal(LinearSDEProposal):

    @property
    def diffusion_param_name(self):
        return 'phi'

    def get_linearising_functions(self):
        if not isinstance(self.SDE, MvSDE):
            return self.get_univ_linearising_functions()
        else:
            return self.get_mv_linearising_functions()
    
    def get_univ_linearising_functions(self):
        def rho(t, x):
            return -1.*self.SDE.db(t, x)
        def mu(t, x):
            A = self.SDE.b(t, x) - self.SDE.db(t, x)*x
            B = self.SDE.db(t, x)
            return -A/B
        def phi(t, x):
            return self.SDE.sigma(t, x)
        return {'rho': rho, 'mu': mu, 'phi': phi}
    
    def get_mv_linearising_functions(self):
        def rho(t, x):
            return -1.*self.SDE.db(t, x)
        def mu(t, x):
            return nla.solve(-1.*self.SDE.db(t, x), self.SDE.b(t, x) - np.einsum('ijk,ik->ij', self.SDE.db(t, x), x))
        def phi(t, x):
            return self.SDE.sigma(t, x)
        mv_linearising_functions = {'rho': rho, 'mu': mu, 'phi': phi}
        if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            mv_linearising_functions = {key: self._sde_tol_dec(func) for key, func in mv_linearising_functions.items()}
        return mv_linearising_functions
    
    def check_drift_param(self, linear_sde_params):
        pass
    
class EllipticBrownianProposal(EllipticLinearSDEProposal, BrownianProposal):

    @property
    def UnivLinearSDECls(self):
        return BrownianMotion
    
    @property
    def MvIndepLinearSDECls(self):
        return MvIndepBrownianMotion
    
    @property
    def MvLinearSDECls(self):
        return MvBrownianMotion

class EllipticOUProposal(EllipticLinearSDEProposal, OUProposal):

    @property
    def UnivLinearSDECls(self):
        return OrnsteinUhlenbeck
    
    @property
    def MvIndepLinearSDECls(self):
        return MvIndepOrnsteinUhlenbeck
    
    @property
    def MvLinearSDECls(self):
        return MvOrnsteinUhlenbeck
    

class IntegratedBrownianProposal(HypoellipticLinearSDEProposal, BrownianProposal):

    @property
    def MvLinearSDECls(self):
        return IntegratedBrownianMotion

    @property
    def MvIndepLinearSDECls(self):
        return IntegratedIndepBrownianMotion
    
class TwiceIntegratedBrownianProposal(HypoellipticLinearSDEProposal, BrownianProposal):

    @property
    def MvLinearSDECls(self):
        return TwiceIntegratedBrownianMotion

    @property
    def MvIndepLinearSDECls(self):
        return TwiceIntegratedIndepBrownianMotion

class IntegratedOUProposal(HypoellipticLinearSDEProposal, OUProposal):

    @property
    def MvLinearSDECls(self):
        return IntegratedOrnsteinUhlenbeck

    @property
    def MvIndepLinearSDECls(self):
        return IntegratedIndepOrnsteinUhlenbeck

class TwiceIntegratedOUProposal(HypoellipticLinearSDEProposal, OUProposal):

    @property
    def MvLinearSDECls(self):
        return TwiceIntegratedOrnsteinUhlenbeck
    
    @property
    def MvIndepLinearSDECls(self):
        return TwiceIntegratedIndepOrnsteinUhlenbeck


# -----------------Univariate Forward Proposals-----------------

class ForwardProposal(SDE):
    """
    Proposal SDE based on the Forward decomposition. Continuous-time likelihood between this proposal and 
    the signal is given by the Girsanov formula.

    The Forward Proposal SDE is only used for simulation, not transformation, and are in general not linear SDEs.
    Thus, for this class we are only interested in the 'simulate' method.

    To construct a ForwardProposal, one needs to subclass and define:

    'LinearSDECls' as a class attribute
    'build_linear_sde' method 
    """

    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        self.SDE = sde
        self.x_start = x_start
        self.t_start = t_start
        self.t_end = t_end
        self.y = y
        self.LY = LY
        self.sigmaY = sigmaY
        self.numerical_scheme = self.numerical_scheme_cls(self)
        self.build_linear_sde()
            
    @property
    def t_diff(self):
        return self.t_end - self.t_start

    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)

    def dsigma(self, t, x):
        return self.SDE.dsigma(self.t_start + t, x)
    
    def _b_time_shifted(self, t, x):
        return self.SDE.b(self.t_start + t, x)

    def simulate(self, size: int, num=5) -> np.ndarray:
        return super().simulate(size, self.x_start, 0., self.t_diff, num)

    def end_point_proposal(self):
        return self.LinearSDE.optimal_proposal_dist(self.t_start, self.t_end, self.x_start, self.y, self.LY, self.sigmaY)

    def b_vec(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self.LinearSDE._vec_grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
        return drift

    def log_girsanov(self, X: np.ndarray):
        names = X.dtype.names
        step = float(names[0])
        X_array = np.stack([self.x_start] + [X[name] for name in names], axis=1) # (N, num+1)
        b_1 = self._b_time_shifted; b_2 = self.b_vec; Cov = self.Cov
        log_girsanov_wgts = log_girsanov(X_array, b_1, b_2, Cov, step)
        return log_girsanov_wgts

# -----------------Brownian Univariate Forward Proposals - 4 Classes -----------------

class NoDriftBasicBrownianProp(ForwardProposal, EllipticBrownianProposal):
    """
    Forward proposal that always takes the standard Brownian motion as the linear SDE
    for evaulation of the proxy. Not adaptive to the end points of previous particles.
    May perform poorly depending of the diffusive regime.

    $$dX_t = dW_t$$
    """
    sname='NDBBrP'
    any_cov = False
    drift = False

class DriftBasicBrownianProp(ForwardProposal, EllipticBrownianProposal):
    """
    $$dX_t = b(t_start, x_start)dt + dW_t$$
    """
    sname='NDBBrP'
    any_cov = False
    drift = True
    
class NoDriftBrownianProp(ForwardProposal, EllipticBrownianProposal):
    """
    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    sname='NDBrP'
    any_cov = True
    drift = False
    
class DriftBrownianProp(ForwardProposal, EllipticBrownianProposal):
    """
    $$dX_t = b(t_start, x_start)dt + \sigma(t_start, x_start)dW_t$$
    """
    sname='DBrP'
    any_cov = True
    drift = True

# -----------------OU Univariate Forward Proposals - 2 Classes -----------------

class LocalLinearBasicOUProp(ForwardProposal, EllipticOUProposal):
    """
    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_start, x_start) - db(t_start, x_start) * x_start
    B = db(t_start, x_start)
    C = 1.
    """
    any_cov = False

class LocalLinearOUProp(ForwardProposal, EllipticOUProposal):
    """
    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_start, x_start) - db(t_start, x_start) * x_start
    B = db(t_start, x_start)
    C = \sigma(t_start, x_start)
    """
    sname='OUP'
    any_cov = True


# -----------------Univariate Diffusion Bridge Proposals-----------------

class AuxiliaryBridge(SDE):
    """
    Base class for auxiliary bridges.

    We use this class to construct any 1-D auxiliary bridge process.

    Given an SDE with certain drift and diffusion coefficient, a starting time, and end time, 
    a starting point and an ending point, this defines a diffusion bridge. It is not possible
    to simulate from this diffusion bridge, as the drift of the diffusion bridge involves the 
    transition density of the SDE, which is typically intractable. An auxiliary bridge process
    is a diffusion that starts and ends at the same points as the diffusion bridge, with known 
    drift, with law that dominates that of the diffusion bridge. Further, the continuous-time 
    likelihood can be evaluated up to discretisation (i.e all the terms inside the path integrals
    are tractable). 
    """
    def __init__(self, sde, t_start, t_end, x_end):
        self.SDE = sde
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.numerical_scheme = self.numerical_scheme_cls(self)
    
    @property
    def t_diff(self):
        return self.t_end - self.t_start

    @property
    def N(self):
        if isinstance(self.x_end, float):
            return 1
        else:
            return self.x_end.shape[0]
    
    def b(self, t, x):
        raise NotImplementedError(self._error_msg('b'))
    
    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)
    
    def _b_time_shifted(self, t, x):
        return self.SDE.b(self.t_start + t, x)

    def _b_vec_time_shifted(self, t, x):
        return self.SDE.b_vec(self.t_start + t, x)

    def bridge_log_likelihood(self, x_start, X):
        raise NotImplementedError(self._error_msg('bridge_log_likelihood'))
    
    def simulate(self, size, x_start, num=5):
        if size != self.N and self.N > 1:
            raise ValueError(f'Simulation size {size} should match dimension of end point vector ({self.N}), unless a single end point is specified.')
        simulation = super().simulate(size, t_start=0., t_end=self.t_diff, x_start=x_start, num=num)
        end_point = simulation.dtype.names[-1]
        simulation[end_point] = np.ones(self.N)*self.x_end if type(self.x_end) == float else self.x_end
        return simulation
    
    def transform_W_to_X(self, W, x_start):
        self._check_end_points_match(W)
        return self.numerical_scheme.transform_W_to_X(W, 0., x_start=x_start, transform_end_point=False)
    
    def transform_X_to_W(self, X, x_start):
        self._check_end_points_match(X)
        return self.numerical_scheme.transform_X_to_W(X, 0., x_start=x_start, transform_end_point=False)
    
    def _check_end_points_match(self, X):
        last_name = X.dtype.names[-1]
        if not np.all(np.isclose(X[last_name], self.x_end)):
            raise ValueError('End points of paths do not match end points of auxiliary bridge.')
        
class DelyonHuAuxBridge(AuxiliaryBridge):
    """
    The auxiliary bridge as proposed by Delyon and Hu (2006).
    """
    
    sname='DH'
    
    def b(self, t, x):
        return (self.x_end - x)/(self.t_diff - t)

    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        b = self._b_vec_time_shifted if hasattr(self.SDE, "b_vec") else self._b_time_shifted
        Cov = self.Cov
        t_end = X.dtype.names[-1]; x_end = X[t_end] # (N, )
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=1) # (N, num+1)
        step = float(X.dtype.names[0]); num = X_array.shape[1] - 1; Delta_s = step*num
        log_density = stats.norm.logpdf(x_end, loc=x_start, scale=np.sqrt(Delta_s * self.Cov(0, x_start)))
        log_det_covs = 0.5*(np.log(Cov(0, x_start)) - np.log(Cov(Delta_s, x_end))) # (N, )
        # The path integrals
        log_path_integral_wgts = log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts
        return log_wgts

class DriftDelyonHuAuxBridge(AuxiliaryBridge):
    """
    The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.
    """
    sname='DDH'
    def b(self, t, x):
        return self._b_time_shifted(t, x) + (self.x_end - x)/(self.t_diff - t)
    
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        b = self._b_vec_time_shifted if hasattr(self.SDE, "b_vec") else self._b_time_shifted
        Cov = self.Cov
        t_end = X.dtype.names[-1]; x_end = X[t_end] # (N, )
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=1) # (N, num+1)
        step = float(X.dtype.names[0]); num = X_array.shape[1] - 1; Delta_s = step*num
        log_density = stats.norm.logpdf(x_end, loc=x_start, scale=np.sqrt(Delta_s * self.Cov(0, x_start)))
        log_det_covs = 0.5*(np.log(Cov(0, x_start)) - np.log(Cov(Delta_s, x_end)))
        # The path integrals
        log_path_integral_wgts = log_drift_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts
        return log_wgts
    
class VanDerMeulenSchauerAuxBridge(AuxiliaryBridge):
    """
    The class of guided bridge proposals based on Linear SDEs:
    """
    def __init__(self, sde, t_start, t_end, x_end):
        super().__init__(sde, t_start, t_end, x_end)
        self.build_linear_sde()

    def b(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self.LinearSDE.grad_log_px(t, self.t_diff, x, self.x_end)
        return drift
    
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        b = self._b_vec_time_shifted if hasattr(self.SDE, "b_vec") else self._b_time_shifted
        Cov = self.Cov; linear_sde = self.LinearSDE
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=1) # (N, num+1)
        step = float(X.dtype.names[0])
        linear_sde_transition_dist = linear_sde.transition_dist(0., self.t_diff, x_start)
        log_linear_sde_density = linear_sde_transition_dist.logpdf(self.x_end)
        # The path integrals
        log_path_integral_wgts = log_van_der_meulen_schauer(X_array, b, Cov, linear_sde, step) # (N, )
        log_wgts = log_linear_sde_density + log_path_integral_wgts
        return log_wgts

    @property
    def sigma_x_end(self):
        return self.SDE.sigma(self.t_end - tol, self.x_end)

# -----------------Brownian Univariate Auxiliary Bridges - 2 Classes -----------------

class NoDriftBrownianAuxBridge(VanDerMeulenSchauerAuxBridge, EllipticBrownianProposal):
    """
    Auxiliary bridge proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. 

    $$dX_t = \sigma(t_end, x_end) dW_t$$
    """
    sname='NDBr'
    drift=False

class DriftBrownianAuxBridge(VanDerMeulenSchauerAuxBridge, EllipticBrownianProposal):
    """
    Auxiliary bridge that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
 
    $$dX_t = b(t_end, x_end)dt + \sigma(t_end, x_end) dW_t$$
    """
    sname='DBr'
    drift=True

# -----------------OU Univariate Auxiliary Bridges - 1 Class -----------------

class LocalLinearOUAuxBridge(VanDerMeulenSchauerAuxBridge, EllipticOUProposal):
    """
    Auxiliary bridge that takes the OU process as the linear SDE for evaluation of the proxy.
    Drift coefficient is obtained through local linearisation of the drift of the signal 
    about the end points of the previous particles. Diffusion coefficient is given by 
    the diffusion of the signal evaluated at hte end points of the previous particles.

    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_end, x_end) - db(t_end, x_end) * x_end
    B = db(t_end, x_end)
    C = \sigma(t_end, x_end)
    """
    sname='OU'

# ----------------- Multivariate Forward Proposals-----------------

class MvForwardProposal(MvSDE, ForwardProposal):
    """
    Proposal SDE based on the Forward decomposition. Continuous-time likelihood between this proposal and 
    the signal is given by the Girsanov formula.

    The Forward Proposal SDE is only used for simulation, not transformation, and are in general not linear SDEs.
    Thus, for this class we are only interested in the 'simulate' method.

    To construct a ForwardProposal, one needs to subclass and define:

    'LinearSDECls' as a class attribute
    'build_linear_sde' method 

    Inputs: x_start: (N, dimX)
    """    
    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        ForwardProposal.__init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY)
        self.check_sde()

    @property
    def dimX(self):
        return self.SDE.dimX

    @property
    def dimW(self):
        return self.SDE.dimW

    @property
    def N(self):
        return self.x_start.shape[0]
        
    @property
    def _diag_cov(self):
        return self.SDE._diag_cov

    def b(self, t, x):
        """
        Input: float, (N, dimX)
        Returns: (N, dimX)
        """
        drift = self._b_time_shifted(t, x) # Inherited from ForwardProposal
        drift += np.einsum('ijk,ik->ij', self.Cov(t, x), self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)) # (N, dimX)
        return drift

    def db(self, t, x):
        """
        To do: think about the special cases in which we can just take the derivative of the original diffusion process.
        """
    # Used to construct a linear SDE for an auxiliary bridge for a forward proposal. 
    # Come back to this an implement if you really need it.
        db = self.SDE.db(self.t_start + t, x) # (N, dimX, dimX)
        # For now, we omit the general case where the diffusion coefficient is state-dependent.
        # db += 2.*self.SDE.dsigma(self.t_start + t, x)*self.sigma(t, x)*self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
        # db += self.Cov(t, x) * self.LinearSDE.grad_grad_log_py(t, self.t_diff, self.y, self.LY, self.sigmaY)
        return db

    def sigma(self, t, x):
        """
        Input: (float, (N, dimX))
        Returns: (N, dimX, dimX)
        """
        return self.SDE.sigma(self.t_start + t, x)

    def dsigma(self, t, x):
        """
        Inputs: float, (N, dimX)
        Returns: float, (N, dimX, dimX, dimX)
        """
        return self.SDE.dsigma(self.t_start + t, x)
    
    def simulate(self, size: int, num=5) -> np.ndarray:
        return super().simulate(size, num)

    def log_girsanov(self, X: np.ndarray):
        """
        Inputs: 
        ------------
        x_start: (N, dimX) array
        X: structured array with fields '0.0', '0.1', ..., '1.0' (N, num)
        
        Returns:
        ------------
        (N, ) Array of weights
        """
        step = float(X.dtype.names[0])
        X_array = np.stack([self.x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        b_1 = self._b_time_shifted; b_2 = self.b; Cov = self.Cov
        log_girsanov_wgts = mv_log_girsanov(X_array, b_1, b_2, Cov, step)
        return log_girsanov_wgts

    def end_point_proposal(self):
        return ForwardProposal.end_point_proposal(self)

class MvEllipticForwardProposal(MvForwardProposal):
    
    def check_sde(self):
        if not isinstance(self.SDE, MvEllipticSDE):
            raise ValueError('The underlying SDE must be an elliptic SDE for elliptic forward proposals.')



#----------------- Brownian Multivariate Forward Proposals: 6 Classes-----------------    

class MvNoDriftBasicBrownianProp(MvEllipticForwardProposal, EllipticBrownianProposal):
    """
    Forward proposal that always takes the standard Brownian motion as the linear SDE
    for evaulation of the proxy. Not adaptive to the end points of previous particles.
    May perform poorly depending of the diffusive regime.

    $$dX_t = dW_t$$
    """
    sname = 'MvNDBBrP'
    any_cov = False
    drift = False

class MvDriftBasicBrownianProp(MvEllipticForwardProposal, EllipticBrownianProposal):
    """
    Forward proposal that takes the standard Brownian motion with a drift component as the linear SDE
    for evaulation of the proxy. 

    $$dX_t = m dt + dW_t$$
    """
    sname = 'MvDBBrP'
    any_cov = False
    drift = True
    
class MvNoDriftIndepBrownianProp(MvEllipticForwardProposal, EllipticBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
    
    Only the diagonal elements of the diffusion matrix are used.
 
    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    sname = 'MvNDIBrP'
    any_cov = True
    full_cov = False
    drift = False

class MvNoDriftBrownianProp(MvEllipticForwardProposal, EllipticBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
    
    Allows for a full diffusion matrix.
 
    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    sname = 'MvNDBrP'
    
    any_cov = True
    full_cov = True
    drift = False

class MvDriftIndepBrownianProp(MvEllipticForwardProposal, EllipticBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Drift/diffusion constants are given by the 
    drift/diffusion of the signal evaluated at the end points of the previous particles.

    Only the diagonal elements of the diffusion matrix are used.
    
    $$dX_t = b(t_start, x_start)dt + \sigma(t_start, x_start)dW_t$$
    """
    sname = 'MvDIBrP'
        
    any_cov = True
    full_cov = False
    drift = True
        
class MvDriftBrownianProp(MvEllipticForwardProposal, EllipticBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Drift/diffusion constants are given by the 
    drift/diffusion of the signal evaluated at the end points of the previous particles.

    The full diffusion matrix is used.
        
    $$dX_t = b(t_start, x_start)dt + \sigma(t_start, x_start)dW_t$$
    """
    
    sname = 'MvDBrP'

    any_cov = True
    full_cov = True
    drift = True

#----------------- OU Multivariate Forward Proposals 3 Classes-----------------    

class MvBasicOUProposal(MvEllipticForwardProposal, EllipticOUProposal):

    sname = 'MvBOUP'
    any_cov = False
    
class MvIndepOUProposal(MvEllipticForwardProposal, EllipticOUProposal):

    sname = 'MvIOUP'
    any_cov = True
    full_cov = False

class MvOUProposal(MvEllipticForwardProposal, EllipticOUProposal):    

    sname = 'MvOUP'
    any_cov = True
    full_cov = True    

# ----------------- Multivariate Diffusion Bridge Proposals-----------------

class MvAuxiliaryBridge(MvSDE, AuxiliaryBridge):

    def __init__(self, *args):
        AuxiliaryBridge.__init__(self, *args)
        if self.x_end.shape[1] != self.SDE.dimX:
            raise ValueError(f'Second dimension of end point array must match dimension of underlying SDE: {self.x_end.shape[1]}!={self.SDE.dimX}')

    @property
    def dimX(self):
        return self.SDE.dimX

    @property
    def dimW(self):
        return self.SDE.dimW

    @property
    def N(self):
        return self.x_end.shape[0]
    
    def b(self, t, x):
        raise NotImplementedError(self._error_msg('b'))
    
    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)
    
    # def _b_vec_time_shifted(self, t, x): # This inherits from AuxiliaryBridge, but you need to implement b_vec for the underlying
    #     return self.SDE.b_vec(self.t_start + t, x)

    def bridge_log_likelihood(self, x_start, X):
        raise NotImplementedError(self._error_msg('bridge_log_likelihood'))
    
    def simulate(self, size, x_start, num=5):
        N = self.N; x_end = self.x_end
        if N == 1 and size != 1:
            x_end = np.concatenate([self.x_end]*size, axis=0) # (size, dimX)
        if size != N and N > 1:
            raise ValueError(f'Simulation size {size} should match number of end point vectors ({self.N}).')
        if x_start.shape not in  [(self.N, self.dimX), (1, self.dimX)] and N>1:
            raise ValueError(f'Starting point array shape {x_start.shape} should be (N, dimX) ({self.N}, {self.dimX}) or (1, dimX) for N>1: N={self.N}.')
        if x_start.shape == (1, self.dimX) and N > 1:
            x_start = np.concatenate([x_start]*N)
        simulation = super().simulate(size, x_start=x_start, num=num)
        end_point = simulation.dtype.names[-1]
        simulation[end_point] = x_end
        return simulation
        
    def _check_end_points_match(self, X):
        last_name = X.dtype.names[-1]
        if not np.all(np.isclose(X[last_name], self.x_end)):
            raise ValueError('End points of paths do not match end points of auxiliary bridge.')
        
    def transform_X_to_W(self, X, x_start):
        if isinstance(self, HypoellipticAuxiliaryBridge):
            raise ValueError('Cannot transform X to W for a hypoelliptic auxiliary bridge.')
        return AuxiliaryBridge.transform_X_to_W(self, X, x_start)

class MvEllipticAuxiliaryBridge(MvAuxiliaryBridge):
    
    def __init__(self, *args):
        MvAuxiliaryBridge.__init__(self, *args)
        self.check_sde()
        
    def check_sde(self):
        if not isinstance(self.SDE, MvEllipticSDE):
            raise ValueError('For an elliptic auxiliary bridge, underlying SDE must be an elliptic SDE.')

class MvDelyonHuAuxBridge(MvEllipticAuxiliaryBridge, DelyonHuAuxBridge):
    """
    The auxiliary bridge as proposed by Delyon and Hu (2006).
    """
    sname = 'MvDH'
    def b(self, t, x):
        return (self.x_end - x)/(self.t_diff - t)

    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = self._self_tol_dec
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        step = float(X.dtype.names[0]); num = X_array.shape[0] - 1; Delta_s = step*num
        log_density = VaryingCovNormal(loc=x_start, cov=Delta_s*Cov(0, x_start)).logpdf(self.x_end) # (N, )
        log_det_covs = 0.5 * (log_abs_det(Cov(0, x_start)) - log_abs_det(Cov(Delta_s, self.x_end))) # (N, )
        # The path integrals
        log_path_integral_wgts = mv_log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts # (N, )
        return log_wgts
    
class MvDriftDelyonHuAuxBridge(MvEllipticAuxiliaryBridge, DriftDelyonHuAuxBridge):
    """
    The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.
    """
    
    sname = 'MvDDH'
    def b(self, t, x):
        return (self._b_time_shifted(t, x) + (self.x_end - x)/(self.t_diff - t))
    
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = self._self_tol_dec
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        step = float(X.dtype.names[0]); num = X_array.shape[1] - 1; Delta_s = step*num
        log_density = VaryingCovNormal(loc=x_start, scale=Delta_s*Cov(0, x_start)).logpdf(self.x_end)
        log_det_covs = 0.5 * (log_abs_det(Cov(0, x_start)) - log_abs_det(Cov(Delta_s, self.x_end)))
        # The path integrals
        log_path_integral_wgts = mv_log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts
        return log_wgts

class MvVanDerMeulenSchauerAuxBridge(MvAuxiliaryBridge, VanDerMeulenSchauerAuxBridge):
    """
    The class of guided bridge proposals based on Linear SDEs:
    """
    def __init__(self, sde, t_start, t_end, x_end):
        VanDerMeulenSchauerAuxBridge.__init__(self, sde, t_start, t_end, x_end)
        self.check_sde()

    def b(self, t, x):
        drift = self._b_time_shifted(t, x) # (N, dimX)
        drift += np.einsum('ijk,ik->ij', self.Cov(t, x), self.LinearSDE.grad_log_px(t, self.t_diff, x, self.x_end))
        return drift

    @property
    def sigma_x_end(self):
        """
        The value of \sigma(t, x_end). The 'matching condition' requires that the diffusion covariance of the underlying SDE at the end points
        must match the diffusion of the auxiliary bridge at the end points. For this to hold, it is sufficient for the diffusion coefficients to 
        match at the end points.
        """
        tol_dec = self._sde_tol_dec
        sigma = tol_dec(self.SDE.sigma)
        sigma_x_end = sigma(self.t_end, self.x_end) # (N, dimX, dimX)
        if self._diag_cov:
            sigma_x_end = np.stack([np.diag(sigma_x_end[i]) for i in range(self.N)], axis=0) if self.N > 1 else np.diag(sigma_x_end[0])
        return sigma_x_end
    
    @property
    def _diag_cov(self):
        if hasattr(self.SDE, '_diag_cov') and self.SDE._diag_cov:
            return True
        else:
            return False
        
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = self._self_tol_dec
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        linear_sde = self.LinearSDE; step = float(X.dtype.names[0])
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        linear_sde_transition_dist = linear_sde.transition_dist(0., self.t_diff, x_start)
        log_linear_sde_density = linear_sde_transition_dist.logpdf(self.x_end)
        # The path integrals
        log_path_integral_wgts = mv_log_van_der_meulen_schauer(X_array, b, Cov, linear_sde, step) # (N, )
        log_wgts = log_linear_sde_density + log_path_integral_wgts
        return log_wgts
 

# ---------------Multivariate Brownian Auxiliary Bridge Proposals: 2 Classes ----------------

class MvNoDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, MvEllipticAuxiliaryBridge, EllipticBrownianProposal):
    """
    Auxiliary bridge proposal that takes the Brownian motion as the linear SDE
    for evaulation of the proxy, without the drift. 

    $$dX_t = \sigma(t_end, x_end) dW_t$$
    """
    sname = 'MvNDBr'    
    drift = False

class MvDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, EllipticBrownianProposal):
    """
    Auxiliary bridge proposal that takes the Brownian motion as the linear SDE
    for evaulation of the proxy, without the drift. 

    $$dX_t = b(t_end, x_end)dt + \sigma(t_end, x_end) dW_t$$
    """
    sname = 'MvDBr'
    drift = True

# ---------------Multivariate OU Auxiliary Bridge Proposals: 1 Class ----------------


class MvLLOUAuxBridge(MvVanDerMeulenSchauerAuxBridge, EllipticOUProposal):
    """
    Auxiliary bridge that takes the OU process as the linear SDE for evaluation of the proxy.
    Drift coefficient is obtained through local linearisation of the drift of the signal 
    about the end points of the previous particles. Diffusion coefficient is given by 
    the diffusion of the signal evaluated at the end points of the previous particles.

    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    Constructed Linear SDE requires evaluation of the Matrix exponential to solve.
    
     $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_start, x_end) - db(t_start, x_end) * x_end
    B = db(t_start, x_end)
    C = \sigma(t_start, x_end)
    """
    sname = 'MvOU'


# ---------------Hypoelliptic Auxiliary Bridge Proposals:  ----------------

class HypoellipticAuxiliaryBridge(MvAuxiliaryBridge):

    def check_sde(self):
        if not isinstance(self.SDE, HypoellipticSDE):
            raise ValueError('For a hypoelliptic auxiliary bridge, underlying SDE must be a hypoelliptic SDE.')

# ---------------Hypoelliptic Brownian Auxiliary Bridge Proposals: 4 Classes ----------------

class IntegratedDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, HypoellipticAuxiliaryBridge, IntegratedBrownianProposal):
    sname = 'H1IDBr'
    drift = True

class IntegratedNoDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, HypoellipticAuxiliaryBridge, IntegratedBrownianProposal):
    sname = 'H1INDBr'
    drift = False
    
class TwiceIntegratedDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, HypoellipticAuxiliaryBridge, TwiceIntegratedBrownianProposal):
    sname = 'H2IDBr'
    drift = True

class TwiceIntegratedNoDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, HypoellipticAuxiliaryBridge, TwiceIntegratedBrownianProposal):
    sname = 'H2INDBr'
    drift = False

# ---------------Hypoelliptic OU Auxiliary Bridge Proposals: 2 Classes ----------------

class IntegratedLLOUAuxBridge(MvVanDerMeulenSchauerAuxBridge, HypoellipticAuxiliaryBridge, IntegratedOUProposal):
    sname = 'H1IDOU'
    
class TwiceIntegratedLLOUAuxBridge(MvVanDerMeulenSchauerAuxBridge, HypoellipticAuxiliaryBridge, TwiceIntegratedOUProposal):
    sname = 'H2INDOU'


#--------------------------------------------------------------------------------------------------------------------------------------


class EndPointProposal(ProbDist):
    """
    Base class for end point proposals used in Backward Guided/Backward Reparameterised Feynman-Kac models.
    """
    pass

class LinearEndPointProposal(Normal, EndPointProposal):
    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        ForwardProposal.__init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY)
        pred = MeanAndCov(self.pred_loc, self.pred_cov)
        opt_prop_loc, opt_prop_cov = filter_step_var_cov(LY, sigmaY ** 2, pred, y)
        Normal.__init__(self, loc=opt_prop_loc, cov=opt_prop_cov)
        
    @property
    def pred_loc(self):
        s = self.t_start; t = self.t_end; x_s = self.x_start
        A = self.LinearSDE._a(s, t); b = self.LinearSDE._b(s, t)
        return A*x_s + b
    
    @property
    def pred_cov(self):
        return self.LinearSDE._v(self.t_start, self.t_end)

class MvLinearEndPointProposal(VaryingCovNormal, LinearEndPointProposal):
    """
    End point proposals for backward guided/reparameterised Feynman Kac models 
    that for each input particle, construct a Linear SDE based on Taylor expansion
    of the drift and diffusion coefficients about the input particles.
    """

    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        MvForwardProposal.__init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY)
        pred = MeanAndCov(self.pred_loc, self.pred_cov)        
        opt_prop_loc, opt_prop_cov = filter_step_var_cov(LY, sigmaY @ sigmaY.T, pred, y) 
        VaryingCovNormal.__init__(self, loc=opt_prop_loc, cov=opt_prop_cov)
    
    @property
    def pred_loc(self):
        s = self.t_start; t = self.t_end; x_s = self.x_start
        A = self.LinearSDE._a(s, t); b = self._b(s, t)
        mu_x = np.einsum('ijk,ik->ij', A, x_s) + b # (N, dimX, dimX), (N, dimX) -> (N, dimX)
        return mu_x
    
    @property
    def pred_cov(self):
        return self.LinearSDE._v(self.t_start, self.t_end)
    
    def ppf(self, u):
        """Method would be inherited from `Normal' unless overridden."""
        raise NotImplementedError


# --------------- Elliptic End Point Proposals: 3 Classes ----------------

class NaiveEndPointProposal(LinearEndPointProposal, EllipticBrownianProposal):
    any_cov = False
    drift = False
    
class EulerMaruyamaEndPointProposal(LinearEndPointProposal, EllipticBrownianProposal):   
    any_cov = True
    drift = True

class OUEndPointProposal(LinearEndPointProposal, EllipticOUProposal):
    any_cov = True
    
# ---------------Multivariate Elliptic End Point Proposals: 3 Classes ----------------

class MvNaiveEndPointProposal(MvLinearEndPointProposal, EllipticBrownianProposal):
    any_cov = False
    drift = False
    
class MvEulerMaruyamaEndPointProposal(MvLinearEndPointProposal, EllipticBrownianProposal):   
    any_cov = True
    full_cov = True
    drift = True

class MvOUEndPointProposal(MvLinearEndPointProposal, EllipticOUProposal):
    any_cov = True
    full_cov = True
    
# ---------------Hypoelliptic Integrated End Point Proposals: 2 Classes ----------------
    
class IntegratedDriftBrownianEndPointProposal(MvLinearEndPointProposal, IntegratedBrownianProposal):
    any_cov = True
    full_cov = True
    drift = True

class IntegratedOUEndPointProposal(MvLinearEndPointProposal, IntegratedOUProposal):
    any_cov = True
    full_cov = True
    
# ---------------Hypoelliptic Twice Integrated End Point Proposals: 2 Classes ----------------
    
class TwiceIntegratedDriftBrownianEndPointProposal(MvLinearEndPointProposal, TwiceIntegratedBrownianProposal):
    any_cov = True
    full_cov = True
    drift = True

class TwiceIntegratedOUEndPointProposal(MvLinearEndPointProposal, TwiceIntegratedBrownianProposal):
    any_cov = True
    full_cov = True


# Forward Proposals
univ_forward_proposals = [NoDriftBasicBrownianProp, DriftBasicBrownianProp, NoDriftBrownianProp, DriftBrownianProp, LocalLinearBasicOUProp, LocalLinearOUProp]
mv_forward_proposals = [MvNoDriftBasicBrownianProp, MvDriftBasicBrownianProp, MvNoDriftIndepBrownianProp, MvNoDriftBrownianProp, MvDriftIndepBrownianProp, MvDriftBrownianProp, MvOUProposal, MvIndepOUProposal]
integrated_forward_proposals = [] # Not implemented yet.
twice_integrated_forward_proposals = [] # Not implemented yet.

#Auxiliary Bridges
univ_auxiliary_bridges = [DelyonHuAuxBridge,  NoDriftBrownianAuxBridge, DriftBrownianAuxBridge, LocalLinearOUAuxBridge]
mv_auxiliary_bridges = [MvDelyonHuAuxBridge, MvNoDriftBrownianAuxBridge, MvDriftBrownianAuxBridge, MvLLOUAuxBridge]
integrated_auxiliary_bridges = [IntegratedDriftBrownianAuxBridge, IntegratedNoDriftBrownianAuxBridge, IntegratedLLOUAuxBridge]
twice_integrated_auxiliary_bridges = [TwiceIntegratedDriftBrownianAuxBridge, TwiceIntegratedNoDriftBrownianAuxBridge, TwiceIntegratedLLOUAuxBridge]

# End Point Proposals
univ_end_point_proposals = [NaiveEndPointProposal, EulerMaruyamaEndPointProposal, OUEndPointProposal] 
mv_end_point_proposals = [MvNaiveEndPointProposal, MvEulerMaruyamaEndPointProposal, MvOUEndPointProposal]
integrated_end_point_proposals = [IntegratedDriftBrownianEndPointProposal, IntegratedOUEndPointProposal]
twice_integrated_end_point_proposals = [TwiceIntegratedDriftBrownianEndPointProposal, TwiceIntegratedOUEndPointProposal]
 
"""
Draft code for forward proposals that use the Euler-Maruyama scheme as the proxy for the transition density.
Thus for fixed t, the transition density is non-linear in x, which is not the case when the transition density
is set by considering the end points of the previous particles. 
"""

# class AdaptiveForwardProposal(ForwardProposal):
#     """
#     Forward proposal that uses a transition density that is a non-linear function of the current state.
#     Includes as a special case using the transition density from the Euler-Maruyama scheme.
#     """
#     def b(self, t, x):
#         self.build_linear_sde()
#         drift = self._b_time_shifted(t, x) 
#         drift += self.Cov(t, x) * self.LinearSDE.grad_log_py(0., self.t_diff - t, x, self.y, self.LY, self.sigmaY)
#         return drift

#     def db(self, t, x):
#         """
#         To do: Think about how to implement this: it is needed for the automatic construction of OU bridges
#         of forward proposals.
#         """
#         raise NotImplementedError('First derivative of adaptive proposal not implemented yet.')

# class EulerForwardProposal(AdaptiveForwardProposal):
#     """
#     Forward proposal that uses the Euler-Maruyama scheme as the proxy for the transition density.
#     """
#     LinearSDECls = BrownianMotion

#     def build_linear_sde(self, t_start, x_start):
#         self.LinearSDE = self.LinearSDECls(m=self._b_time_shifted(t_start, x_start), s=self.Cov(t_start, x_start))

# class OUAdaptiveForwardProposal(AdaptiveForwardProposal):
#     """
#     Forward proposal that uses the transition density from an OU process as the proxy for the true transition density.
#     """
#     LinearSDECls = OrnsteinUhlenbeck
    
#     def build_linear_sde(self, x_start):
#         return super().build_linear_sde(x_start)

# class LinearSDEForwardProposal(ForwardProposal):

#     def b(self, t, x):
#         drift = self._b_time_shifted(t, x)
#         drift += self.Cov(t, x) * self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
#         return drift

#     def db(self, t, x):
#         db = self.SDE.db(self.t_start + t, x) 
#         db += self.SDE.dCov(self.t_start + t, x)*self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
#         db += self.Cov(t, x) * self.LinearSDE.grad_grad_log_py(t, self.t_diff, self.y, self.LY, self.sigmaY)
#         return db

#     def end_point_proposal(self, x_start):
#         self.build_linear_sde(x_start)
#         return self.LinearSDE.optimal_proposal_dist(self.t_start, self.t_end, x_start, self.y, self.LY, self.sigmaY)
   


# class MvEulerForwardProposal(MvEllipticForwardProposal):

#     def b(self, t, x):
#         """
#         Input: float, (N, dimX)
#         Returns: (N, dimX)
#         """
#         drift = self._b_time_shifted(t, x) # Inherited from ForwardProposal
#         drift += np.einsum('ijk,ik->ij', self.Cov(t, x), self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)) # (N, dimX)
#         return drift

#     def db(self, t, x): 
#         """
#         To do: think about the special cases in which we can just take the derivative of the original diffusion process.
#         """
#         db = self.SDE.db(self.t_start + t, x) # (N, dimX, dimX)
#         # For now, we omit the general case where the diffusion coefficient is state-dependent.
#         # db += 2.*self.SDE.dsigma(self.t_start + t, x)*self.sigma(t, x)*self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
#         # db += self.Cov(t, x) * self.LinearSDE.grad_grad_log_py(t, self.t_diff, self.y, self.LY, self.sigmaY)
#         return db

# class MvLinearSDEForwardProposal(MvEllipticForwardProposal):

#     def b(self, t, x):
#         """
#         Input: float, (N, dimX)
#         Returns: (N, dimX)
#         """
#         drift = self._b_time_shifted(t, x) # Inherited from ForwardProposal
#         drift += np.einsum('ijk,ik->ij', self.Cov(t, x), self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)) # (N, dimX)
#         return drift

#     def db(self, t, x):
#         """
#         To do: think about the special cases in which we can just take the derivative of the original diffusion process.
#         """
#     # Used to construct a linear SDE for an auxiliary bridge for a forward proposal. 
#     # Come back to this an implement if you really need it.
#         db = self.SDE.db(self.t_start + t, x) # (N, dimX, dimX)
#         # For now, we omit the general case where the diffusion coefficient is state-dependent.
#         # db += 2.*self.SDE.dsigma(self.t_start + t, x)*self.sigma(t, x)*self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
#         # db += self.Cov(t, x) * self.LinearSDE.grad_grad_log_py(t, self.t_diff, self.y, self.LY, self.sigmaY)
#         return db

#     def end_point_proposal(self, x_start):
#         self.build_linear_sde(x_start)
#         return self.LinearSDE.optimal_proposal_dist(self.t_start, self.t_end, x_start, self.y, self.LY, self.sigmaY)
