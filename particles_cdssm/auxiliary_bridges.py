"""
Auxiliary Bridges module:
--------------------------
Within this module, we implement abstractions of the folllowing objects:

- Forward Proposals:                        Proposal SDEs for use in Forward Guided/Reparameterised Feynman-Kac models. Likelihood ratio between proposal and signal is given by the Girsanov formula.
(Forward Guided/Forward Reparameterised)    Evaulation of likelihood ratio requires an invertible covariance matrix, thus this contruction is only possible for elliptic model SDEs,

- Auxiliary Bridges:                        SDEs that are absolutely contiuous with respect to the true diffusion bridge of the signal process. The constuction of diffusion bridge proposals
(Backward Guided/Backward Reparameterised)  has been an active area of research over the past 20 years. We implement here the Delyon-Hu bridge (2006) and the Van Der Meulen-Schauer bridges (2017)
                                            that are based on selecting an appropriate choice of LinearSDE that is close to the signal process.

- End Point Proposals:                      Proposals that are used to propose the end points of the signal process conditional on the obervation. Used in the first step of the simulation of a
(Backward Guided/Backward Reparameterised)  backward proposal, within a backward guided or backward reparameterised Feynman-Kac model. Subclasses of the particles.distributions.ProbDist class.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Univariate Case
---------------------------------------------------
---------------------------------------------------


Forward Proposals: 4 Brownian Proposals, 2 OU Proposals
-----------------

(Abstract Base class: ForwardProposalBase -> ForwardProposal)

- NoDriftBasicBrownianProp (NDBBrP): No Drift, diffusion 1.
- NoDriftBrownianProp (NDBrP): No drift, diffusion \sigma(t_start, x_start)
- DriftBasicBrownianProp (DBBrP): Drift b(t_start, x_start), diffusion 1.
- DriftBrownianProp (DBrP): Drift b(t_start, x_start), diffusion \sigma(t_start, x_start)

- LocalLinearBasicOUProp (BOUP): Drift local linearising b at t_start, x_start, diffusion coefficient as 1.
- LocalLinearOUProp (OUP): Drift local linearising b at t_start, x_start, diffusion coefficient as \sigma(t_start, x_start)

Auxiliary Bridges: 2 Delyon Hu Bridges, 2 Brownian Aux Bridges, 1 OU Aux Bridge 
-----------------

(Abstract Base class: AuxiliaryBridgeBase -> AuxiliaryBridge)

- DelyonHuAuxBridge (DH): The auxiliary bridge as proposed by Delyon and Hu (2006)
- DriftDelyonHuAuxBridge (DDH): The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients. (Not implemented yet)

(Abstract Base Class: AuxiliaryBridgeBase -> AuxiliaryBridge -> VanDerMeulenSchauerAuxBridge)

- NoDriftBrownianAuxBridge (NDBr): Drift: 0., diffusion \sigma(t_end, x_end) (matching condition)
- DriftBrownianAuxBridge (DBr): Drift: b(t_end, x_end), diffusion \sigma(t_end, x_end) (matching condition)

- LocalLinearOUAuxBridge (OU): Drift: Local linearising b at t_end, x_end, diffusion \sigma(t_end, x_end) (matching condition)

End Point Proposals: 3 LinearSDE End Point Proposals (Can add more if desired)
--------------------

(Abstract Base Class: EndPointProposal -> LinearEndPointProposalBase -> LinearEndPointProposal)

- NaiveEndPointProposal (NDBBrP): E_t | E_{t-1} = e_{t-1} ~ N(0, (t-s))
- EulerMaruyamaEndPointProposal (DBrP): E_t | E_{t-1} = e_{t-1} ~ N(e_{t-1} + b(0, e_{t-1)), (t-s)\sigma(0, e_{t-1}))
- OUEndPointProposal (OUP): Transition of an OU process with diffusion coefficient \sigma(0, e_{t-1})

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Multivariate Elliptic Case
---------------------------

Forward Proposals: 6 Brownian Proposals, 3 OU Proposals
-----------------

(Abstract Base class: ForwardProposalBase -> MvForwardProposal -> MvEllipticForwardProposal)

- MvNoDriftBasicBrownianProp (MvNDBBrP): Drift: 0.*1_d, diffusion I_d 
- MvDriftBasicBrownianProp (MvDBBrP): Drift: b(t_start, x_start), diffusion I_d
- MvNoDriftIndepBrownianProp (MvNDIBrP): Drift: b(t_start, x_start), diffusion diag(\sigma(t_start, x_start))
- MvNoDriftBrownianProp (MvNDBrP): Drift: b(t_start, x_start), diffusion \sigma(t_start, x_start)
- MvDriftIndepBrownianProp (MvDIBrP): Drift: b(t_start, x_start), diffusion diag(\sigma(t_start, x_start))
- MvDriftBrownianProp (MvDBrP): Drift: b(t_start, x_start), diffusion \sigma(t_start, x_start)

- MvBasicOUProposal (MvBOUP): Local linearising b at t_start, x_start, diffusion I_d
- MvIndepOUProposal (MvIOUP): Local linearising b at t_start, x_start, taking only diagonal components of B(t) matrix, diffusion \diag(\sigma(t_start, x_start))
- MvOUProposal (MvOUP): Drift: Local linearising b at t_start, x_start, taking only diagonal components of B(t) matrix, diffusion \sigma(t_start, x_start)

Auxiliary Bridges: 2 Delyon Hu bridges, 2 Brownian Aux Bridges, 1 OU Aux Bridge
-----------------

(Abstract Base Class: AuxiliaryBridgeBase -> MvAuxiliaryBridge -> MvEllipticAuxiliaryBridge)

- MvDelyonHuAuxBridge (MvDH): The auxiliary bridge as proposed by Delyon and Hu (2006)
- MvDriftDelyonHuAuxBridge (MvDDH): The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients. (Not implemented yet)

(Abstract Base Class: AuxiliaryBridgeBase -> MvAuxiliaryBridge -> MvVanDerMeulenSchauerAuxBridge -> MvEllipticVanDerMeulenSchauerAuxBridge)
                           AuxiliaryBridgeBase -> MvAuxiliaryBridge -> MvEllipticAuxiliaryBridge -> MvEllipticVanDerMeulenSchauerAuxBridge

- MvNoDriftBrownianAuxBridge (MvNDBr): Auxiliary bridge proposal that takes the Brownian motion with 0 drift as the linear SDE
- MvDriftBrownianAuxBridge (MvDBr): Auxiliary bridge that takes the Brownian motion as the linear SDE
- MvOUAuxBridge (MvOU): Auxiliary bridge that takes the OU process as the linear SDE.


End Point Proposals: 4 LinearSDE End Point Proposals (Can add more if desired)
--------------------

(Abstract Base Class: EndPointProposal -> LinearEndPointProposalBase -> MvLinearEndPointProposal -> MvEllipticLinearEndPointProposal)

- MvNaiveEndPointProposal (MvNDBBrP): E_t | E_{t-1} = e_{t-1} ~ N(0, (t-s)I_d)
- MvEulerMaruyamaEndPointProposal (MvDBrP): E_t | E_{t-1} = e_{t-1} ~ N(e_{t-1} + b(0, e_{t-1)), (t-s)\sigma(0, e_{t-1}))
- MvIndepOUEndPointProposal (MvIOUP): Transition of an OU process with diagonal elements in B matrix, diagnoal diffusion matrix.
- MvOUEndPointProposal (MvOUP): Transition of an OU process with diagonal elements in B matrix.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Multivariate Hypoelliptic Case (Includes both integrated and twice integrated SDEs)
-----------------------------------------------------------------------------------

Forward Proposals: Forward Proposals are possible for the hypoelliptic case, but only for filtering and smoothing algorithms that do not change ancestors.
-----------------

Currently not implemented.


Auxiliary Bridges: - 4 Integrated SDEs, 4 Twice Integrated SDEs: All based on the Van Der Meulen-Schauer bridge
-----------------

(Abstract Base Class: AuxiliaryBridgeBase -> MvAuxiliaryBridge -> HypoellipticAuxiliaryBridge -> IntegratedAuxiliaryBridge -> IntegratedVanDerMeulenSchauerAuxBridge)
                                          AuxiliaryBridgeBase -> MvAuxiliaryBridge -> MvVanDerMeulenSchauerAuxiliaryBridge -> IntegratedVanDerMeulenSchauerAuxBridge


- IntegratedDriftBrownianAuxBridge (H1IDBr): Auxiliary bridge that uses an integrated Brownian motion as the proxy.
- IntegratedNoDriftBrownianAuxBridge (H1INDBr): Auxiliary bridge that uses an integrated Brownian motion with 0 drift, as the proxy.
- IntegratedLLOUAuxBridge (H1IOU): Auxiliary Bridge that uses an integrated O-U process as the proxy.
- IntegratedNoDriftLLOUAuxBridge (H1INDOU): Auxiliary Bridge that uses an integrated O-U process as the proxy. Removes the 0th order term from the Taylor expanison of the drift.


(Abstract Base Class: MvAuxiliaryBridge -> HypoellipticAuxiliaryBridge -> TwiceIntegratedAuxiliaryBridge -> TwiceIntegratedVanDerMeulenSchauerAuxBridge)
                                               MvAuxiliaryBridge -> MvVanDerMeulenSchauerAuxiliaryBridge -> TwiceIntegratedVanDerMeulenSchauerAuxBridge


- TwiceIntegratedDriftBrownianAuxBridge (H2IDBr): Auxiliary bridge that uses an integrated Brownian motion as the proxy.
- TwiceIntegratedNoDriftBrownianAuxBridge (H2INDBr): Auxiliary bridge that uses an integrated Brownian motion with 0 drift, diagonal diffusion covariance as the proxy.
- TwiceIntegratedLLOUAuxBridge (H2IOU): Auxiliary Bridge that uses a twice integrated O-U process as the proxy.
- TwiceIntegratedNoDriftLLOUAuxBridge (H2INDOU): Auxiliary Bridge that uses a twice integrated O-U process as the proxy. Removes the 0th order term from the Taylor expanison of the drift.


End Point Proposals: 4 for integrated SDEs, 4 for twice integrated SDEs
-----------------

(Abstract Base Class: EndPointProposal -> LinearEndPointProposalBase -> MvLinearEndPointProposal -> HypoellipticLinearEndPointProposal -> IntegratedLinearEndPointProposal)

- IntegratedNaiveEndPointProposal (H1INDBBrP): E_t | E_{t-1} = e_{t-1} ~ N(0, (t-s)I_d)
- IntegratedDriftBrownianEndPointProposal (H1IDBrP): E_t | E_{t-1} = e_{t-1} ~ N(e_{t-1} + b(0, e_{t-1}), (t-s)\sigma(0, e_{t-1}))
- IntegratedIndepOUEndPointProposal (H1IOUP): Transition of an integrated O-U process with diagonal elements in B matrix, diagnoal diffusion matrix.
- IntegratedOUEndPointProposal (H1OUP): Transition of an integrated O-U process with diagonal elements in B matrix.

(Abstract Base Class: EndPointProposal -> LinearEndPointProposalBase -> MvLinearEndPointProposal -> HypoellipticLinearEndPointProposal -> TwiceIntegratedLinearEndPointProposal)

- TwiceIntegratedNaiveEndPointProposal (H2INDBBrP): E_t | E_{t-1} = e_{t-1} ~ N(0, (t-s)I_d)    
- TwiceIntegratedDriftBrownianEndPointProposal (H2IDBrP): E_t | E_{t-1} = e_{t-1} ~ N(e_{t-1} + b(0, e_{t-1}), (t-s)\sigma(0, e_{t-1}))
- TwiceIntegratedIndepOUEndPointProposal (H2IOUP): Transition of a twice integrated O-U process with diagonal elements in B matrix, diagonal diffusion matrix.
- TwiceIntegratedOUEndPointProposal (H2OUP): Transition of a twice integrated O-U process with diagonal elements in B matrix.


Also implemmented at the start of the file, are utility classes that are used to add the following methods to the classes above:

`build_linear_sde`: `BuildLinearSDE` -> BuildBrownianLinearSDE, BuildOULinearSDE
`check_sde`: `CheckSDE` -> CheckUnivSDE, CheckEllipticSDE, CheckHypoellipticSDE, CheckIntegratedSDE, CheckTwiceIntegratedSDE
"""

import numpy as np
import numpy.linalg as nla
import scipy.stats as stats
from particles_cdssm.numerical_schemes import HypoellipticEulerMaruyama
from particles_cdssm.sdes import SDEBase, SDE, MvSDE, MvEllipticSDE, HypoellipticSDE, BrownianMotion, OrnsteinUhlenbeck, MvIndepBrownianMotion, MvBrownianMotion, MvIndepOrnsteinUhlenbeck, MvOrnsteinUhlenbeck
from particles_cdssm.sdes import HypoellipticSDE, IntegratedSDE, TwiceIntegratedSDE, IntegratedIndepBrownianMotion, IntegratedBrownianMotion, IntegratedIndepOrnsteinUhlenbeck, IntegratedOrnsteinUhlenbeck
from particles_cdssm.sdes import TwiceIntegratedIndepBrownianMotion, TwiceIntegratedBrownianMotion, TwiceIntegratedIndepOrnsteinUhlenbeck, TwiceIntegratedOrnsteinUhlenbeck, TimeSwitchingSDE
from particles_cdssm.path_integrals import log_girsanov, log_delyon_hu, log_van_der_meulen_schauer, mv_log_girsanov, mv_log_delyon_hu, mv_log_van_der_meulen_schauer
from particles_cdssm.tools import log_abs_det, filter_step_var_cov, mv_filter_step_var_cov, MeanAndCov
from particles.distributions import Normal, VaryingCovNormal

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

def _self_tol_dec(self, func):
    """
    Used to decorate functions within `bridge_log_likelihood' method of 
    the following classes: 
    - MvDelyonHuAuxBridge 
    - MvVanDerMeulenSchauerAuxBridge
    """
    if isinstance(self.SDE, TimeSwitchingSDE) and self.t_end == self.SDE.t_switch:
        return add_tol(self.t_diff, func)
    else:
        return func
        
class BuildLinearSDE(object):
    """
    Utility Class that is used to add the 'build_linear_sde' methods to the following classes:
    
    - ForwardProposal
    - VanDerMeulenSchauerAuxBridge
    - LinearEndPointProposal

    - MvEllipticForwardProposal
    - MvEllipticVanDerMeulenSchauerAuxBridge
    - MvEllipticLinearEndPointProposal
    
    - MvIntegratedVanDerMeulenSchauerAuxBridge
    - MvTwiceIntegratedVanDerMeulenSchauerAuxBridge
    
    - IntegratedLinearEndPointProposal
    - TwiceIntegratedLinearEndPointProposal
    
    We need to specify the following attributes:

    any_cov (ForwardProposalBase/LinearEndPointPropoosalBase only)
    full_cov (MvForwardProposal/MvLinearEndPointProposal only)
    drift (Brownian LinearSDEs only)
    
    Due to the matching condition, the covariance matrix is predetermined by the diffusion of the signal at the end points
    in the case of auxiliary bridges.
    
    We need to add 3 decorators:
    
    - Rough comp dec: when the underlying SDE is Hypoelliptic
    - SDE Tol Dec: When the underlying SDE is a TimeSwitchingSDE and/or is an instance of a multivariate diffusion bridge MvVanderMeulenSchauerAuxBridge
    - Trim output: when N=1, we need to change the output that comes from the linearing functions to be of shape (N, dimX, dimW) -> (dimX, dimW).
    """   

    def get_linearising_points(self):
        if isinstance(self, ForwardProposalBase) or isinstance(self, LinearEndPointProposalBase):
            return self.t_start, self.x_start
        if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            return self.t_end, self.x_end
        if isinstance(self, AuxiliaryBridge):
            return self.t_end - tol, self.x_end

    def _sde_tol_dec(self, func):
        if not isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            return func
        isforwardprop = isinstance(self.SDE, ForwardProposalBase)
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

    def _check_dim_dec(self, func):
        if not (self.N == 1 and isinstance(self.SDE, MvSDE)):
            return func
        dw = self.SDE.dimW; dx = self.SDE.dimX; N = self.N
        if isinstance(self.SDE, MvEllipticSDE):
            def trim_output(t, x):
                out = func(t, x)
                return out.reshape(dx, dw) if out.shape == (N, dx, dw) else out
        else:
            def trim_output(t, x):
                out = func(t, x)
                return out.reshape(dw, dw) if out.shape == (N, dw, dw) else out
        return trim_output
    
    def get_linearising_functions(self):
        linearising_functions = self._get_linearising_functions()
        # Add any decorators here
        decorators = [self._check_dim_dec, self._sde_tol_dec]
        for dec in decorators: # Add all decorators to linearising functions
            linearising_functions = {key: dec(func) for key, func in linearising_functions.items()}
        return linearising_functions

    def check_diffusion_param(self, linear_sde_params):
        par = self.diffusion_param_name
        diffusion = linear_sde_params[par]
        if isinstance(self, ForwardProposalBase) or isinstance(self, LinearEndPointProposalBase): 
            if not self.any_cov:
                del linear_sde_params[par]
                return linear_sde_params
            if (isinstance(self, MvForwardProposal) or isinstance(self, MvLinearEndPointProposal)) and not self.full_cov:
                linear_sde_params[par] = self._param_to_diag(diffusion)
        if isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            if self._diag_cov:
                linear_sde_params[par] = self._param_to_diag(diffusion)
        return linear_sde_params

    def _param_to_diag(self, param):
        N = self.N
        return np.diagonal(param, axis1=1, axis2=2) if N > 1 else np.diag(param).reshape(N, self.SDE.dimW)
    
    def check_linear_sde_params(self, linear_sde_params):
        return self.check_drift_param(self.check_diffusion_param(linear_sde_params))

    def check_matching_condition(self, linear_sde_params):
        """
        Checks whether the matching condition of the diffusion coefficient for the VanDerMeulen-Schauer aux bridge holds.
        
        linear_sde_params: Will extract the diffusion param, which is of shape (N, dimW, dimW) for N > 1 or (dimW, dimW) for N=1
                
        We need to compare to sigma_x_end, which is of shape (N, dimX, dimW) for any N.
        """
        par = self.diffusion_param_name
        if isinstance(self, VanDerMeulenSchauerAuxBridge) or isinstance(self, MvVanDerMeulenSchauerAuxBridge):
            if not isinstance(self, HypoellipticAuxiliaryBridge):
                diff_coef = linear_sde_params[par]
            else:
                Z = np.zeros((self.N, self.dimS, self.dimW)) if self.N > 1 else np.zeros((self.dimS, self.dimW))
                ax = 1 if self.N > 1 else 0
                diff_coef = np.concatenate([Z, linear_sde_params[par]], axis=ax)
            matching_condition = np.all(np.isclose(self.sigma_x_end, diff_coef)) # Broadcasts for N=1
            if not matching_condition:
                raise ValueError('Matching condition for bridge proposal not satisfied.')
                
    def get_linear_sde_params(self):
        t, x = self.get_linearising_points()
        linearising_functions = self.get_linearising_functions()
        linear_sde_params = {param: func(t, x) for param, func in linearising_functions.items()}
        self.check_matching_condition(linear_sde_params)
        if isinstance(self.SDE, MvSDE):
            linear_sde_params.update({'N': self.N, 'dimX': self.SDE.dimX})
            linear_sde_params = self.check_linear_sde_params(linear_sde_params)
        return linear_sde_params
    
    @property
    def LinearSDECls(self):
        if isinstance(self.SDE, SDE):
            return self.UnivLinearSDECls
        if isinstance(self.SDE, MvEllipticSDE):
            if isinstance(self, MvEllipticVanDerMeulenSchauerAuxBridge):
                return self.MvIndepLinearSDECls if self._diag_cov else self.MvLinearSDECls
            else:
                return self.MvIndepLinearSDECls if (not self.any_cov or not self.full_cov) else self.MvLinearSDECls
        if isinstance(self.SDE, IntegratedSDE):
            if isinstance(self, IntegratedVanDerMeulenSchauerAuxBridge):
                return self.IntegratedIndepLinearSDECls if self._diag_cov else self.IntegratedLinearSDECls
            else:
                return self.IntegratedIndepLinearSDECls if (not self.any_cov or not self.full_cov) else self.IntegratedLinearSDECls            
        if isinstance(self.SDE, TwiceIntegratedSDE):
            if isinstance(self, TwiceIntegratedVanDerMeulenSchauerAuxBridge):
                return self.TwiceIntegratedIndepLinearSDECls if self._diag_cov else self.TwiceIntegratedLinearSDECls
            else:
                return self.TwiceIntegratedIndepLinearSDECls if (not self.any_cov or not self.full_cov) else self.TwiceIntegratedLinearSDECls

    def build_linear_sde(self):
        self.LinearSDE = self.LinearSDECls(**self.get_linear_sde_params())

class BuildBrownianLinearSDE(BuildLinearSDE):
    
    UnivLinearSDECls = BrownianMotion
    MvLinearSDECls = MvBrownianMotion
    MvIndepLinearSDECls = MvIndepBrownianMotion
    IntegratedLinearSDECls = IntegratedBrownianMotion
    IntegratedIndepLinearSDECls = IntegratedIndepBrownianMotion
    TwiceIntegratedLinearSDECls = TwiceIntegratedBrownianMotion
    TwiceIntegratedIndepLinearSDECls = TwiceIntegratedIndepBrownianMotion
    
    diffusion_param_name = 's'

    def _get_linearising_functions(self):
        if not isinstance(self.SDE, HypoellipticSDE):
            m_func = self.SDE.b
            s_func = self.SDE.sigma
        else:
            m_func = self.SDE.b_rough
            s_func = self.SDE.sigma_rough
        return {'m': m_func, 's': s_func}

    def check_drift_param(self, linear_sde_params):
        if not self.drift:
            linear_sde_params['m'] = np.zeros_like(linear_sde_params['m']) if isinstance(self.SDE, MvSDE) else 0.
        return linear_sde_params

    
class BuildOULinearSDE(BuildLinearSDE):

    UnivLinearSDECls = OrnsteinUhlenbeck
    MvLinearSDECls = MvOrnsteinUhlenbeck
    MvIndepLinearSDECls = MvIndepOrnsteinUhlenbeck
    IntegratedLinearSDECls = IntegratedOrnsteinUhlenbeck
    IntegratedIndepLinearSDECls = IntegratedIndepOrnsteinUhlenbeck
    TwiceIntegratedLinearSDECls = TwiceIntegratedOrnsteinUhlenbeck
    TwiceIntegratedIndepLinearSDECls = TwiceIntegratedIndepOrnsteinUhlenbeck

    drift=True # We keep the 0th order element of the expansion of the 
                # drift term, in almost all cases.
    diffusion_param_name = 'phi'
    
    def _get_linearising_functions(self):
        if not isinstance(self.SDE, MvSDE):
            return self.get_univ_linearising_functions()
        elif not isinstance(self.SDE, HypoellipticSDE):
            return self.get_mv_linearising_functions()
        else:
            return self.get_hypo_linearising_functions()
    
    def get_univ_linearising_functions(self):
        def rho(t, x):
            db = self.SDE.db(t, x)
            db = self._subst_vals(db)
            return -1.*db
        def mu(t, x):
            A = self.SDE.b(t, x) - self.SDE.db(t, x)*x
            B = self.SDE.db(t, x)
            B = self._subst_vals(B)
            return -A/B
        def phi(t, x):
            return self.SDE.sigma(t, x)
        return {'rho': rho, 'mu': mu, 'phi': phi}
    
    def get_mv_linearising_functions(self):
        def rho(t, x):
            db = self.SDE.db(t, x)
            db = self._subst_vals(db)
            return -1.*db
        def mu(t, x):
            db = self.SDE.db(t, x)
            A = self.SDE.b(t, x) - np.einsum('ijk,ik->ij', self.SDE.db(t, x), x) # (N, dw)
            B = self._subst_vals_matrix(db)
            return nla.solve(-1.*B, A)
        def phi(t, x):
            return self.SDE.sigma(t, x)
        return {'rho': rho, 'mu': mu, 'phi': phi}
    
    def get_hypo_linearising_functions(self):
        def rho(t, x):
            self.SDE.db_rough(t, x)
            db_rough = self._subst_vals(self.SDE.db_rough(t, x))
            return -1.*db_rough
        def mu(t, x):
            db_rough = self.SDE.db_rough(t, x)
            A = self.SDE.b_rough(t, x) - np.einsum('ijk,ik->ij', self.SDE.db_rough(t, x), x[:, -self.SDE.dimW:]) # (N, dw)
            B = self._subst_vals_matrix(db_rough)
            return nla.solve(-1.*B, A)
        def phi(t, x):
            return self.SDE.sigma_rough(t, x)
        return {'rho': rho, 'mu': mu, 'phi': phi}
        
    def check_drift_param(self, linear_sde_params):     
        # If a multivariate SDE, take only the diagonal elements (to avoid matrix exponential)
        if isinstance(self.SDE, MvSDE):
            linear_sde_params['rho'] = self._param_to_diag(linear_sde_params['rho'])
        # Set the rho parameter to a small value if it is close to 0.
        # linear_sde_params['rho'] = self._subst_vals(linear_sde_params['rho'])
        if not self.drift: # Remove 0th order term in Taylor expansion of the drift term (currently only used in Hypoelliptic case).
            linear_sde_params['mu'] = np.zeros_like(linear_sde_params['mu']) if isinstance(self.SDE, MvSDE) else 0.
        return linear_sde_params

    def _subst_vals(self, param, subst_val=0.1, subst_val_tol=1e-5):
        if isinstance(param, float) and np.abs(param) < subst_val_tol:
            return subst_val
        else:
            return np.where(np.abs(param) < subst_val_tol, subst_val, param)
    
    def _subst_vals_matrix(self, db, subst_val=0.1, subst_val_tol=1e-5):
        N = db.shape[0]
        det_db = np.linalg.det(db)
        low_det = np.abs(det_db) < subst_val_tol
        if not np.any(low_det):
            return db
        elif np.all(low_det):
            return np.array([np.eye(self.SDE.dimW)]*N)
        else:
            for i, det in enumerate(det_db):
                if low_det[i]:
                    db[i] = subst_val * np.sign(det) * np.eye(self.SDE.dimX)

class CheckSDE(object):
    """
    Utility class to add `check_sde method to ForwardProposal, AuxiliaryBridge and EndPointProposal classes.
    """
    def check_sde(self):
        if not isinstance(self.SDE, SDEBase):
            raise ValueError(f'Input SDE is not an instance of an SDE Base.')
        if not isinstance(self.SDE, self.CheckSDECls):
            raise ValueError(f'The underlying SDE must be in class {self.CheckSDECls.__name__} for class {self.__class__.__name__}')

class CheckUnivSDE(CheckSDE):
    CheckSDECls = SDE
    
class CheckEllipticSDE(CheckSDE):
    CheckSDECls = MvEllipticSDE

class CheckHypoellipticSDE(CheckSDE):
    CheckSDECls = HypoellipticSDE

class CheckIntegratedSDE(CheckSDE):
    CheckSDECls = IntegratedSDE
        
class CheckTwiceIntegratedSDE(CheckSDE):
    CheckSDECls = TwiceIntegratedSDE
    

# -----------------Univariate Forward Proposals-----------------

class ForwardProposalBase(SDEBase):
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
        self.check_sde()
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

    def log_girsanov(self, X: np.ndarray):
        raise NotImplementedError(self._error_msg('log_girsanov'))        

class ForwardProposal(ForwardProposalBase, SDE, CheckUnivSDE):
    """
    Only univariate forward proposals (with dimX=dimW=1) will be instances of this class.
    """
    
    @property
    def N(self):
        if isinstance(self.x_start, float):
            return 1
        else:
            return self.x_start.shape[0]
        
    def b(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self._grad_log_py(t, x)
        return drift

    def db(self, t, x):
        """
        NOT IMPLEMENTED CORRECTLY.
        This is needed for ForwardGuided/ForwardReparametrised DA, when using a VanDerMeulen and Schauer OU bridge proposal.
        This bridge construction is only used in the smoothing, and numerical experiments show that the choice of bridge for the 
        reparameterisation does not affect the performance of the smoothing algorithms, so this is not a priority. 
        """
        # raise NotImplementedError(self._error_msg('db'))
        db = self.SDE.db(t, x)
        return db

    def b_vec(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self.LinearSDE._vec_grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
        return drift
    
    def _grad_log_py(self, t, x):
        return self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)

    def _b_2(self, t, x):
        return self.b_vec(t, x)
    
    def log_girsanov(self, X):        
        step = float(X.dtype.names[0])
        X_array = np.stack([self.x_start] + [X[name] for name in X.dtype.names], axis=1) # (N, num+1)
        b_1 = self._b_time_shifted; b_2 = self._b_2; Cov = self.Cov
        log_girsanov_wgts = log_girsanov(X_array, b_1, b_2, Cov, step)
        return log_girsanov_wgts    

# -----------------Brownian Univariate Forward Proposals - 4 Classes -----------------

class NoDriftBasicBrownianProp(ForwardProposal, BuildBrownianLinearSDE):
    """
    Forward proposal that always takes the standard Brownian motion as the linear SDE
    for evaulation of the proxy. Not adaptive to the end points of previous particles.
    May perform poorly depending of the diffusive regime.

    $$dX_t = dW_t$$
    """
    sname='NDBBrP'
    any_cov = False
    drift = False

class DriftBasicBrownianProp(ForwardProposal, BuildBrownianLinearSDE):
    """
    $$dX_t = b(t_start, x_start)dt + dW_t$$
    """
    sname='DBBrP'
    any_cov = False
    drift = True
    
class NoDriftBrownianProp(ForwardProposal, BuildBrownianLinearSDE):
    """
    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    sname='NDBrP'
    any_cov = True
    drift = False
    
class DriftBrownianProp(ForwardProposal, BuildBrownianLinearSDE):
    """
    $$dX_t = b(t_start, x_start)dt + \sigma(t_start, x_start)dW_t$$
    """
    sname='DBrP'
    any_cov = True
    drift = True

# -----------------OU Univariate Forward Proposals - 2 Classes -----------------

class LocalLinearBasicOUProp(ForwardProposal, BuildOULinearSDE):
    """
    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_start, x_start) - db(t_start, x_start) * x_start
    B = db(t_start, x_start)
    C = 1.
    """
    sname='BOUP'
    any_cov = False

class LocalLinearOUProp(ForwardProposal, BuildOULinearSDE):
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

class AuxiliaryBridgeBase(object):
    """
    Base class for auxiliary bridges.

    Any auxiliary bridge of any type (univariate, mv elliptic, mv hypoelliptic) will be an instance of this class.
    """
    def __init__(self, sde, t_start, t_end, x_end):
        self.SDE = sde
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.check_sde()
        self.numerical_scheme = self.numerical_scheme_cls(self)
    
    @property
    def t_diff(self):
        return self.t_end - self.t_start
        
    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)
    
    def _b_time_shifted(self, t, x):
        return self.SDE.b(self.t_start + t, x)

    def _b_vec_time_shifted(self, t, x):
        return self.SDE.b_vec(self.t_start + t, x)

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

class AuxiliaryBridge(AuxiliaryBridgeBase, SDE, CheckUnivSDE):
    """
    Only univarite auxiliary bridges (with dimX=dimW=1) will be instances of this class.
    """
    @property
    def N(self):
        if isinstance(self.x_end, float):
            return 1
        else:
            return self.x_end.shape[0]

    def simulate(self, size, x_start, num=5):
        if size != self.N and self.N > 1:
            raise ValueError(f'Simulation size {size} should match dimension of end point vector ({self.N}), unless a single end point is specified.')
        simulation = SDE.simulate(self, size, t_start=0., t_end=self.t_diff, x_start=x_start, num=num)
        end_point = simulation.dtype.names[-1]
        simulation[end_point] = np.ones(self.N)*self.x_end if type(self.x_end) == float else self.x_end
        return simulation
    
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
        log_path_integral_wgts = self.log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts
        return log_wgts
    
    def log_delyon_hu(self, X_array, b, Cov, step):
        return log_delyon_hu(X_array, b, Cov, step)

class DriftDelyonHuAuxBridge(DelyonHuAuxBridge):
    """
    The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.
    """
    sname='DDH'
    def b(self, t, x):
        return self._b_time_shifted(t, x) + (self.x_end - x)/(self.t_diff - t)
    
    def log_delyon_hu(self, X_array, b, Cov, step):
        # Note: this function has not been tested yet.
        raise NotImplementedError('Continuous-time likelihood for Delyon-Hu bridge with drift not implemented/tested yet.')
        #return log_drift_delyon_hu(X_array, b, Cov, step)
    
class VanDerMeulenSchauerAuxBridge(AuxiliaryBridge):
    """
    The class of guided bridge proposals based on Linear SDEs:
    """
    def __init__(self, sde, t_start, t_end, x_end):
        super().__init__(sde, t_start, t_end, x_end)
        self.build_linear_sde()

    def b(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self._grad_log_px(t, x)
        return drift
    
    def _grad_log_px(self, t, x):
        return self.LinearSDE.grad_log_px(t, self.t_diff, x, self.x_end)

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

class NoDriftBrownianAuxBridge(VanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    """
    Auxiliary bridge proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. 

    $$dX_t = \sigma(t_end, x_end) dW_t$$
    """
    sname='NDBr'
    drift=False

class DriftBrownianAuxBridge(VanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    """
    Auxiliary bridge that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
 
    $$dX_t = b(t_end, x_end)dt + \sigma(t_end, x_end) dW_t$$
    """
    sname='DBr'
    drift=True

# -----------------OU Univariate Auxiliary Bridges - 1 Class -----------------

class LocalLinearOUAuxBridge(VanDerMeulenSchauerAuxBridge, BuildOULinearSDE):
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


# -----------------Univariate End Point Proposals-----------------

class EndPointProposal(object):
    """
    Base class for end point proposals used in Backward Guided/Backward Reparameterised Feynman-Kac models.
    
    Other strategies could be developed for proposing the end point: for example, one could use a locally 
    Gaussian numerical scheme.    
    """
    pass

class LinearEndPointProposalBase(EndPointProposal):
    """
    All end point proposals based on the construction of a Linear SDE and Gaussian conditioning on the
    observation y_t are subclassess of this base class.
    """
    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        self.SDE = sde
        self.x_start = x_start
        self.t_start = t_start
        self.t_end = t_end
        self.y = y
        self.LY = LY
        self.sigmaY = sigmaY
        self.check_sde()
        self.build_linear_sde()
        pred = MeanAndCov(self.pred_loc, self.pred_cov)
        self.opt_prop_loc, self.opt_prop_cov = self.filter_step_var_cov(LY, sigmaY, pred, y)

    @property
    def dimX(self):
        return self.SDE.dimX

class LinearEndPointProposal(Normal, LinearEndPointProposalBase, CheckUnivSDE):
    """
    Univariate linear SDE end point proposals are subclasses of this class.
    """
    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        LinearEndPointProposalBase.__init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY)
        Normal.__init__(self, loc=self.opt_prop_loc, scale=np.sqrt(self.opt_prop_cov))

    @property
    def N(self):
        if isinstance(self.x_start, float):
            return 1
        else:
            return self.x_start.shape[0]

    @property
    def pred_loc(self):
        s = self.t_start; t = self.t_end; x_s = self.x_start
        A = self.LinearSDE._a(s, t); b = self.LinearSDE._b(s, t)
        return A*x_s + b

    @property
    def pred_cov(self):
        return self.LinearSDE._v(self.t_start, self.t_end)

    def filter_step_var_cov(self, LY, sigmaY, pred, y):
        return filter_step_var_cov(LY, sigmaY ** 2, pred, y)

# --------------- Univariate End Point Proposals: 3 Classes ----------------

class NaiveEndPointProposal(LinearEndPointProposal, BuildBrownianLinearSDE):
    sname = 'NDBBrP'
    any_cov = False
    drift = False
    
class EulerMaruyamaEndPointProposal(LinearEndPointProposal, BuildBrownianLinearSDE):   
    sname = 'DBrP'
    any_cov = True
    drift = True

class OUEndPointProposal(LinearEndPointProposal, BuildOULinearSDE):
    sname = 'OUP'
    any_cov = True
    
# ----------------- Multivariate Forward Proposals-----------------

class MvForwardProposal(ForwardProposalBase, MvSDE):
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

    @property
    def dimX(self):
        return self.SDE.dimX

    @property
    def dimW(self):
        return self.SDE.dimW

    @property
    def N(self):
        return self.x_start.shape[0]

    def b(self, t, x):
        """
        Input: float, (N, dimX)
        Returns: (N, dimX)
        """
        drift = self._b_time_shifted(t, x) # Inherited from ForwardProposalBase
        drift += np.einsum('ijk,ik->ij', self.Cov(t, x), self._grad_log_py(t, x)) # (N, dimX)
        return drift

    def db(self, t, x):
        """
        NOT IMPLEMENTED CORRECTLY
        This is needed for ForwardGuided/ForwardReparametrised DA, when using a VanDerMeulen and Schauer OU bridge proposal.
        Thr brige construction is only used in the smoothing, and numerical experiments show that the choice of bridge for the 
        reparameterisation does not affect the performance of the smoothing algorithms, so this is not a priority. 
        """
        db = self.SDE.db(self.t_start + t, x) # (N, dimX, dimX)
        # raise NotImplementedError('Need to fix this implementation.')
        return db

    def _grad_log_py(self, t, x):
        return self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)

    def _b_2(self, t, x):
        return self.b(t, x)
    
    def log_girsanov(self, X):
        step = float(X.dtype.names[0])
        X_array = np.stack([self.x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        b_1 = self._b_time_shifted; b_2 = self.b; Cov = self.Cov
        log_girsanov_wgts = mv_log_girsanov(X_array, b_1, b_2, Cov, step)
        return log_girsanov_wgts
        
class MvEllipticForwardProposal(MvForwardProposal, MvEllipticSDE, CheckEllipticSDE):
    pass

#----------------- Brownian Multivariate Forward Proposals: 6 Classes-----------------    

class MvNoDriftBasicBrownianProp(MvEllipticForwardProposal, BuildBrownianLinearSDE):
    """
    Forward proposal that always takes the standard Brownian motion as the linear SDE
    for evaulation of the proxy. Not adaptive to the end points of previous particles.
    May perform poorly depending of the diffusive regime.

    $$dX_t = dW_t$$
    """
    sname = 'MvNDBBrP'
    any_cov = False
    drift = False

class MvDriftBasicBrownianProp(MvEllipticForwardProposal, BuildBrownianLinearSDE):
    """
    Forward proposal that takes the standard Brownian motion with a drift component as the linear SDE
    for evaulation of the proxy. 

    $$dX_t = m dt + dW_t$$
    """
    sname = 'MvDBBrP'
    any_cov = False
    drift = True
    
class MvNoDriftIndepBrownianProp(MvEllipticForwardProposal, BuildBrownianLinearSDE):
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

class MvNoDriftBrownianProp(MvEllipticForwardProposal, BuildBrownianLinearSDE):
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

class MvDriftIndepBrownianProp(MvEllipticForwardProposal, BuildBrownianLinearSDE):
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
        
class MvDriftBrownianProp(MvEllipticForwardProposal, BuildBrownianLinearSDE):
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

class MvBasicOUProposal(MvEllipticForwardProposal, BuildOULinearSDE):
    
    sname = 'MvBOUP'
    any_cov = False
    
class MvIndepOUProposal(MvEllipticForwardProposal, BuildOULinearSDE):
    
    sname = 'MvIOUP'
    any_cov = True
    full_cov = False

class MvOUProposal(MvEllipticForwardProposal, BuildOULinearSDE):    

    sname = 'MvOUP'
    any_cov = True
    full_cov = True    

# ----------------- Multivariate Diffusion Bridge Proposals-----------------


class MvAuxiliaryBridge(AuxiliaryBridgeBase, MvSDE):

    def __init__(self, *args):
        AuxiliaryBridgeBase.__init__(self, *args)
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
        simulation = AuxiliaryBridge.simulate(self, size, x_start=x_start, num=num)
        end_point = simulation.dtype.names[-1]
        simulation[end_point] = x_end
        return simulation
        
class MvEllipticAuxiliaryBridge(MvAuxiliaryBridge, CheckEllipticSDE):
    pass

class MvDelyonHuAuxBridge(MvEllipticAuxiliaryBridge):
    """
    The auxiliary bridge as proposed by Delyon and Hu (2006).
    """
    sname = 'MvDH'
    def b(self, t, x):
        return DelyonHuAuxBridge.b(self, t, x)

    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = lambda func: _self_tol_dec(self, func)
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        step = float(X.dtype.names[0]); num = X_array.shape[0] - 1; Delta_s = step*num
        log_density = VaryingCovNormal(loc=x_start, cov=Delta_s*Cov(0, x_start)).logpdf(self.x_end) # (N, )
        log_det_covs = 0.5 * (log_abs_det(Cov(0, x_start)) - log_abs_det(Cov(Delta_s, self.x_end))) # (N, )
        # The path integrals
        log_path_integral_wgts = self.mv_log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts # (N, )
        return log_wgts

    def mv_log_delyon_hu(self, X_array, b, Cov, step):
        return mv_log_delyon_hu(X_array, b, Cov, step)
    
class MvDriftDelyonHuAuxBridge(MvEllipticAuxiliaryBridge):
    """
    The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.
    """
    
    sname = 'MvDDH'
    def b(self, t, x):
        return DriftDelyonHuAuxBridge.b(self, t, x)
    
    def mv_log_delyon_hu(self, X_array, b, Cov, step):
        raise NotImplementedError('Continuous-time likelihood for Delyon-Hu bridge with drift not implemented/tested yet.')
    
class MvVanDerMeulenSchauerAuxBridge(MvAuxiliaryBridge):
    """
    The class of guided bridge proposals based on Linear SDEs:
    """
    def __init__(self, sde, t_start, t_end, x_end):
        super().__init__(sde, t_start, t_end, x_end)
        self.build_linear_sde()

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
        sigma_x_end = sigma(self.t_end, self.x_end) # (N, dimX, dimW)
        return sigma_x_end
    
    @property
    def _diag_cov(self):
        if hasattr(self.SDE, '_diag_cov') and self.SDE._diag_cov:
            return True
        else:
            return False
        
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = lambda func: _self_tol_dec(self, func)
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        linear_sde = self.LinearSDE; step = float(X.dtype.names[0])
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        linear_sde_transition_dist = linear_sde.transition_dist(0., self.t_diff, x_start)
        log_linear_sde_density = linear_sde_transition_dist.logpdf(self.x_end)
        # The path integrals
        log_path_integral_wgts = mv_log_van_der_meulen_schauer(X_array, b, Cov, linear_sde, step) # (N, )
        log_wgts = log_linear_sde_density + log_path_integral_wgts
        return log_wgts

class MvEllipticVanDerMeulenSchauerAuxBridge(MvVanDerMeulenSchauerAuxBridge, MvEllipticAuxiliaryBridge):
    pass    

# ---------------Multivariate Brownian Auxiliary Bridge Proposals: 2 Classes ----------------

class MvNoDriftBrownianAuxBridge(MvEllipticVanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    """
    Auxiliary bridge proposal that takes the Brownian motion as the linear SDE
    for evaulation of the proxy, without the drift. 

    $$dX_t = \sigma(t_end, x_end) dW_t$$
    """
    sname = 'MvNDBr'    
    drift = False

class MvDriftBrownianAuxBridge(MvEllipticVanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    """
    Auxiliary bridge proposal that takes the Brownian motion as the linear SDE
    for evaulation of the proxy, without the drift. 

    $$dX_t = b(t_end, x_end)dt + \sigma(t_end, x_end) dW_t$$
    """
    sname = 'MvDBr'
    drift = True

# ---------------Multivariate OU Auxiliary Bridge Proposals: 1 Class ----------------


class MvLLOUAuxBridge(MvEllipticVanDerMeulenSchauerAuxBridge, BuildOULinearSDE):
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

# ---------------Multivariate Elliptic End Point Proposals ----------------

class MvLinearEndPointProposal(VaryingCovNormal, LinearEndPointProposalBase):
    """
    End point proposals for backward guided/reparameterised Feynman Kac models 
    that for each input particle, construct a Linear SDE based on Taylor expansion
    of the drift and diffusion coefficients about the input particles.
    """
    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        LinearEndPointProposalBase.__init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY)
        VaryingCovNormal.__init__(self, loc=self.opt_prop_loc, cov=self.opt_prop_cov)

    @property
    def N(self):
        return self.x_start.shape[0]

    @N.setter
    def N(self, value):
        pass

    @property
    def pred_loc(self):
        s = self.t_start; t = self.t_end; x_s = self.x_start
        A = self.LinearSDE._a(s, t); b = self.LinearSDE._b(s, t)
        mu_x = np.einsum('ijk,ik->ij', A, x_s) + b # (N, dimX, dimX), (N, dimX) -> (N, dimX)
        return mu_x

    @property
    def pred_cov(self):
        return self.LinearSDE._v(self.t_start, self.t_end)

    def filter_step_var_cov(self, LY, sigmaY, pred, y):
        return mv_filter_step_var_cov(LY, sigmaY @ sigmaY.T, pred, y)

class MvEllipticLinearEndPointProposal(MvLinearEndPointProposal, CheckEllipticSDE):
    pass    

# ---------------Multivariate Elliptic End Point Proposals: 4 Classes ----------------

class MvNaiveEndPointProposal(MvEllipticLinearEndPointProposal, BuildBrownianLinearSDE):
    sname='MvNDBBrP'
    any_cov = False
    drift = False
    
class MvEulerMaruyamaEndPointProposal(MvEllipticLinearEndPointProposal, BuildBrownianLinearSDE):   
    sname='MvDBrP'
    any_cov = True
    full_cov = True
    drift = True

class MvIndepOUEndPointProposal(MvEllipticLinearEndPointProposal, BuildOULinearSDE):
    sname='MvIOUP'
    any_cov = True
    full_cov = False
    
class MvOUEndPointProposal(MvEllipticLinearEndPointProposal, BuildOULinearSDE):
    sname='MvOUP'
    any_cov = True
    full_cov = True

# ---------------Hypoelliptic Auxiliary Bridge Proposals:  ----------------

class HypoellipticAuxiliaryBridge(MvAuxiliaryBridge, CheckHypoellipticSDE, HypoellipticSDE):

    numerical_scheme_cls = HypoellipticEulerMaruyama
    
    @property
    def dimS(self):
        return self.SDE.dimS

    def transform_X_to_W(self, X, x_start):
        return ValueError('Transformation from X to W not possible for a hypoelliptic auxiliary bridge.')

class IntegratedAuxiliaryBridge(CheckIntegratedSDE, HypoellipticAuxiliaryBridge):
    pass

class TwiceIntegratedAuxiliaryBridge(CheckTwiceIntegratedSDE, HypoellipticAuxiliaryBridge):
    pass
                
class IntegratedVanDerMeulenSchauerAuxBridge(IntegratedAuxiliaryBridge, MvVanDerMeulenSchauerAuxBridge):
    pass

class TwiceIntegratedVanDerMeulenSchauerAuxBridge(TwiceIntegratedAuxiliaryBridge, MvVanDerMeulenSchauerAuxBridge):
    pass
       

# ---------------Hypoelliptic Brownian Auxiliary Bridge Proposals: 4 Classes ----------------

class IntegratedDriftBrownianAuxBridge(IntegratedVanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    sname = 'H1IDBr'
    drift = True

class IntegratedNoDriftBrownianAuxBridge(IntegratedVanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    sname = 'H1INDBr'
    drift = False
    
class TwiceIntegratedDriftBrownianAuxBridge(TwiceIntegratedVanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    sname = 'H2IDBr'
    drift = True

class TwiceIntegratedNoDriftBrownianAuxBridge(TwiceIntegratedVanDerMeulenSchauerAuxBridge, BuildBrownianLinearSDE):
    sname = 'H2INDBr'
    drift = False

# ---------------Hypoelliptic OU Auxiliary Bridge Proposals: 4 Classes ----------------

class IntegratedLLOUAuxBridge(IntegratedVanDerMeulenSchauerAuxBridge, BuildOULinearSDE):
    sname = 'H1IOU'

class IntegratedNoDriftLLOUAuxBridge(IntegratedVanDerMeulenSchauerAuxBridge, BuildOULinearSDE):
    sname = 'H1INDOU'
    drift=False # Exclude the 0th order term in the Taylor expansion of the drift.
    
class TwiceIntegratedLLOUAuxBridge(TwiceIntegratedVanDerMeulenSchauerAuxBridge, BuildOULinearSDE):
    sname = 'H2IOU'

class TwiceIntegratedNoDriftLLOUAuxBridge(TwiceIntegratedVanDerMeulenSchauerAuxBridge, BuildOULinearSDE):
    sname = 'H2INDOU'
    drift=False # Exclude the 0th order term in the Taylor expansion of the drift.
    

# ---------------Hypoelliptic End Point Proposals:  ----------------

class HypoellipticLinearEndPointProposal(MvLinearEndPointProposal, CheckHypoellipticSDE):
    pass

class IntegratedLinearEndPointProposal(CheckIntegratedSDE, HypoellipticLinearEndPointProposal):
    pass

class TwiceIntegratedLinearEndPointProposal(CheckTwiceIntegratedSDE, HypoellipticLinearEndPointProposal):
    pass

# ---------------Hypoelliptic Integrated End Point Proposals: 4 Classes ----------------

class IntegratedNaiveEndPointProposal(IntegratedLinearEndPointProposal, BuildBrownianLinearSDE):
    sname='H1INDBBrP'
    any_cov = False
    drift = False

class IntegratedNoDriftEndPointProposal(IntegratedLinearEndPointProposal, BuildBrownianLinearSDE):
    sname='H1INDBrP'
    any_cov = True
    full_cov = True
    drift = False
        
class IntegratedDriftBrownianEndPointProposal(IntegratedLinearEndPointProposal, BuildBrownianLinearSDE):
    sname='H1IDBrP'
    any_cov = True
    full_cov = True
    drift = True

class IntegratedIndepOUEndPointProposal(IntegratedLinearEndPointProposal, BuildOULinearSDE):
    sname='H1IIOUP'
    any_cov = True
    full_cov = False
    
class IntegratedOUEndPointProposal(IntegratedLinearEndPointProposal, BuildOULinearSDE):
    sname='H1IOUP'
    any_cov = True
    full_cov = True
    
# ---------------Hypoelliptic Twice Integrated End Point Proposals: 4 Classes ----------------

class TwiceIntegratedNaiveEndPointProposal(TwiceIntegratedLinearEndPointProposal, BuildBrownianLinearSDE):
    sname='H2INDBBrP'
    any_cov = False
    drift = True
        
class TwiceIntegratedDriftBrownianEndPointProposal(TwiceIntegratedLinearEndPointProposal, BuildBrownianLinearSDE):
    sname='H2IDBrP'
    any_cov = True
    full_cov = True
    drift = True

class TwiceIntegratedIndepOUEndPointProposal(TwiceIntegratedLinearEndPointProposal, BuildOULinearSDE):
    sname='H2IIOUP'
    any_cov = True
    full_cov = False

class TwiceIntegratedOUEndPointProposal(TwiceIntegratedLinearEndPointProposal, BuildOULinearSDE):
    sname='H2IOUP'
    any_cov = True
    full_cov = True


# Forward Proposals
univ_forward_proposals = [NoDriftBasicBrownianProp, DriftBasicBrownianProp, NoDriftBrownianProp, DriftBrownianProp, LocalLinearBasicOUProp, LocalLinearOUProp]
mv_forward_proposals = [MvNoDriftBasicBrownianProp, MvDriftBasicBrownianProp, MvNoDriftIndepBrownianProp, MvNoDriftBrownianProp, MvDriftIndepBrownianProp, MvDriftBrownianProp, MvBasicOUProposal, MvIndepOUProposal, MvOUProposal]
integrated_forward_proposals = [] # Not implemented yet.
twice_integrated_forward_proposals = [] # Not implemented yet.

# Auxiliary Bridges
univ_auxiliary_bridges = [DelyonHuAuxBridge,  NoDriftBrownianAuxBridge, DriftBrownianAuxBridge, LocalLinearOUAuxBridge]
mv_auxiliary_bridges = [MvDelyonHuAuxBridge, MvNoDriftBrownianAuxBridge, MvDriftBrownianAuxBridge, MvLLOUAuxBridge]
integrated_auxiliary_bridges = [IntegratedDriftBrownianAuxBridge, IntegratedNoDriftBrownianAuxBridge, IntegratedLLOUAuxBridge, IntegratedNoDriftLLOUAuxBridge]
twice_integrated_auxiliary_bridges = [TwiceIntegratedDriftBrownianAuxBridge, TwiceIntegratedNoDriftBrownianAuxBridge, TwiceIntegratedLLOUAuxBridge, TwiceIntegratedNoDriftLLOUAuxBridge]

# End Point Proposals
univ_end_point_proposals = [NaiveEndPointProposal, EulerMaruyamaEndPointProposal, OUEndPointProposal]
mv_end_point_proposals = [MvNaiveEndPointProposal, MvEulerMaruyamaEndPointProposal, MvIndepOUEndPointProposal, MvOUEndPointProposal]
integrated_end_point_proposals = [IntegratedNaiveEndPointProposal, IntegratedDriftBrownianEndPointProposal, IntegratedIndepOUEndPointProposal, IntegratedOUEndPointProposal]
twice_integrated_end_point_proposals = [TwiceIntegratedNaiveEndPointProposal, TwiceIntegratedDriftBrownianEndPointProposal, TwiceIntegratedIndepOUEndPointProposal, TwiceIntegratedOUEndPointProposal]
 
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
