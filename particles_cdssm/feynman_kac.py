"""
Feynman Kac module: 

In this module, we construct the Feynman Kac models for continuous-discrete state space models, which include reparameterisations.

How to interact with the API:

We have implemented the CDSSM_SMC class: 

This is a subclass of SMC. We use this to run particle filters and smoothers for continuous-discrete state space models.
The only difference between this class and the standard SMC class, is that we need to pass a CDSSM_FeynmanKac object to the __init__ method,
and we need to provide the number of imputed points for simulation of SDEs.

We have the following Feynman-Kac formalisms of continuous-discrete state space models: These are instances of CDSSM_FeynmanKac:

- Bootstrap DA
- BootstrapReparameterisedDA - provide an AuxiliaryBridge class
- ForwardGuidedDA - provide a ForwardProposal class
- BackwardGuidedDA - provide an EndPointProposal class, AuxiliaryBridge class
- ForwardGuidedReparameterisedDA - provide a ForwardProposal class, AuxiliaryBridge class
- BackwardGuidedReparameterisedDA - provide an EndPointProposal class, AuxiliaryBridge class # Only option for hypoelliptic case.

We provide to the __init__ method of these classes:

- cdssm: an instance of CDSSM
- data: the observed data

And any proposal/auxiliary bridge classes that are required. Some default choices are specified if 
they are not provided, but the user can specify one manually.

Each of these classes have the class methods:

'auxiliary_bridge_cls_options'
'univ_auxiliary_bridge_cls_options'
'mv_auxiliary_bridge_cls_options'
'integrated_auxiliary_bridge_cls_options'
'twice_integrated_auxiliary_bridge_cls_options'

'proposal_sde_cls_options'
'univ_proposal_sde_cls_options'
'mv_proposal_sde_cls_options'

'end_pt_proposal_cls_options'
'univ_end_pt_proposal_cls_options'
'mv_end_pt_proposal_cls_options'
'integrated_end_pt_proposal_cls_options'
'twice_integrated_end_pt_proposal_cls_options'

These methods return dictionaries of the available classes that can be used for the respective roles.

Currently, the default choices are:

- Forward Proposals: LocalLinearOUProp/ MvOUProposal
- end_pt_proposal_cls: LocalLinearOUProp/ MvOUProposal / IntegratedDriftBrownianEndPointProposal/ TwiceIntegratedDriftBrownianEndPointProposal
- auxiliary_bridge_cls: DelyonHuAuxBridge/ MvDelyonHuAuxBridge / IntegratedDriftBrownianAuxBridge/ TwiceIntegratedDriftBrownianAuxBridge
"""

import numpy as np

from particles import FeynmanKac
from particles.resampling import wmean_and_var

import sdes.sdes as sdes 
import sdes.auxiliary_bridges as axb
import sdes.continuous_discrete_ssms as cdssms

err_msg_missing_trans_pathspace = """
    Feynman-Kac class %s is missing method logpt, which provides the log-pdf
    of Markov transition X_t | X_{t-1}. This is required by most smoothing
    algorithms, and is only possible to implement for CDSSMs with a 
    reparameterisation of the pathspace.

    To do smoothing on CDSSM, use a 'reparameterised' CDSSM_FeynmanKac class.
    """

class CDSSM_FeynmanKac(FeynmanKac):
    
    num = 10 # Defualt number of steps for the numerical scheme if not passed new value by SMC.

    def __init__(self, cdssm=None, data=None):
        self.cdssm = cdssm
        self.data = data
        self.du = self.cdssm.model_sde.dimX # Defined in __init__ of ssms.Bootstrap
        self.model_sde = self.cdssm.model_sde
        self.dimX = self.cdssm.model_sde.dimX
        self.dimW = self.cdssm.model_sde.dimW
        self._check_CDSSM_FK(cdssm)
        if not self.cdssm.isequidistant:
            assert self.T == self.cdssm.nobs, 'Data length must be of length equal to expected number of observations from the CDSSM'
    
    def _check_CDSSM_FK(self, cdssm):
        if not isinstance(cdssm, cdssms.CDSSMBase):
            raise ValueError(f'cdssm {cdssm} must be an instance of CDSSM')

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)
            
    def M0(self, N): # Called by `SMC`
        return self.M0_observedat0(N) if self.cdssm.isobservedat0 else self.M0_else(N)

    def M0_observedat0(self, N):
        raise NotImplementedError(self._error_msg('M0_observedat0'))
    
    def M0_else(self, N):
        raise NotImplementedError(self._error_msg('M0_else'))
    
    def M(self, t, xp): # Called by `SMC`
        raise NotImplementedError(self._err_msg('M'))

    def logG(self, t, xp, x): # Called by `SMC`
        raise NotImplementedError(self._err_msg('M'))
    
    def logpt(self, t, xp, x):
        """Log-density of X_t given X_{t-1}."""
        raise NotImplementedError(err_msg_missing_trans_pathspace % self.__class__.__name__)
    
    def upper_bound_trans(self, t):
        """
        Method needs to be defined on the underlying cdssm to use the
        collector `Paris` and the offline smoothing algorithm 
        `backward_sampling_reject`.
        
        For CDSSMs, logpt is the product of the potential G_t and the transition density of proposal kernel M_t.
        It may not be possible to find an upper bound on the pathspace.
        """
        raise NotImplementedError('Upper bound on logpt is not available for CDSSMs')

    def add_func(self, t, xp, x):
        """
        Method needs to be defined on the underlying cdssm to use the collectors
        'Online_smooth_naive'/'Online_smooth_ON2'/'PaRIS'. 
        """
        return self.cdssm.add_func(t, xp, x)

    @property
    def is1d(self):
        return isinstance(self.cdssm, cdssms.CDSSM)

    def default_moments(self, W, X):
        """
        Defines the default moments function for the collector 'Moments' 
        (see the particles module 'collectors').

        In the future, we could use the function resampling.wmean_and_var_str_array to store 
        summaries of the entire SDE path. We will need to think about how to implement this a 
        bit, so that the paths at each time step have a common ending point.
        """
        end_pts = X[X.dtype.names[-1]] 
        return wmean_and_var(W, end_pts)

    def _preprocess_logpt(self, xp, x):
        x_start = xp[xp.dtype.names[-1]]; N = x_start.shape[0]
        x = np.stack([x]*N) if type(x) is np.void else x
        x_end = x[x.dtype.names[-1]]
        return x_start, x, x_end

    def _init_x0(self, N):
        return np.ones(N) * self.cdssm.x0 if self.is1d else np.concatenate([self.cdssm.x0]*N)

class ForwardProposalMixin:

    @classmethod
    def univ_proposal_sde_cls_options(cls):
        return {cls.__name__: cls for cls in axb.univ_forward_proposals}    

    @classmethod
    def mv_proposal_sde_cls_options(cls):
        return {cls.__name__: cls for cls in axb.mv_forward_proposals}

    @property
    def default_proposal_sde_cls(self):
        return axb.LocalLinearOUProp if self.is1d else axb.MvOUProposal

    def _build_forward_proposal(self, t, x_start):
        if self.cdssm.isobservedat0:
            return self.proposal_sde_cls(self.model_sde, x_start, self.cdssm.S(t-1), self.cdssm.S(t), self.data[t], self.cdssm.LY(t), self.cdssm.SigmaY(t))
        else:
            return self.proposal_sde_cls(self.model_sde, x_start, self.cdssm.S(t), self.cdssm.S(t+1), self.data[t], self.cdssm.LY(t), self.cdssm.SigmaY(t))
        
class AuxiliaryBridgeMixin:

    @classmethod
    def univ_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in axb.univ_auxiliary_bridges}

    @classmethod
    def mv_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in axb.mv_auxiliary_bridges}

    @classmethod
    def integrated_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in axb.integrated_auxiliary_bridges}

    @classmethod
    def twice_integrated_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in axb.twice_integrated_auxiliary_bridges}

    @property
    def default_auxiliary_bridge_cls(self):
        if isinstance(self.cdssm.model_sde, sdes.IntegratedSDE):
            return axb.IntegratedDriftBrownianAuxBridge
        if isinstance(self.cdssm.model_sde, sdes.TwiceIntegratedSDE):
            return axb.TwiceIntegratedDriftBrownianAuxBridge
        if isinstance(self.cdssm.model_sde, sdes.MvEllipticSDE):
            return axb.MvDelyonHuAuxBridge
        else:
            return axb.DelyonHuAuxBridge

    def _build_aux_bridge(self, t, x_start, x_end):
        if self.cdssm.isobservedat0:
            return self.auxiliary_bridge_cls(self.model_sde, self.cdssm.S(t-1), self.cdssm.S(t), x_end)
        else:
            return self.auxiliary_bridge_cls(self.model_sde, self.cdssm.S(t), self.cdssm.S(t+1), x_end)

class EndPointProposalMixin:

    @classmethod
    def univ_end_pt_proposal_cls_options(cls):
        return {cls.__name__: cls for cls in axb.univ_end_point_proposals}

    @classmethod
    def mv_end_pt_proposal_cls_options(cls):
        return {cls.__name__: cls for cls in axb.mv_end_point_proposals}

    @classmethod
    def integrated_end_pt_proposal_cls_options(cls):
        return {cls.__name__: cls for cls in axb.integrated_end_point_proposals}
    
    @classmethod
    def twice_integrated_end_pt_proposal_cls_options(cls):
        return {cls.__name__: cls for cls in axb.twice_integrated_end_point_proposals}
        
    @property
    def default_end_pt_proposal_cls(self):
        if isinstance(self.cdssm.model_sde, sdes.IntegratedSDE):
            return axb.IntegratedDriftBrownianEndPointProposal
        if isinstance(self.cdssm.model_sde, sdes.TwiceIntegratedSDE):
            return axb.TwiceIntegratedDriftBrownianEndPointProposal
        if isinstance(self.cdssm.model_sde, sdes.MvEllipticSDE):
            return axb.MvEulerMaruyamaEndPointProposal
        else:
            return axb.EulerMaruyamaEndPointProposal

    def _build_end_point_proposal(self, t, x_start):
        if self.cdssm.isobservedat0:
            return self.end_pt_proposal_cls(self.model_sde, x_start, self.cdssm.S(t-1), self.cdssm.S(t), self.data[t], self.cdssm.LY(t), self.cdssm.SigmaY(t))
        else:
            return self.end_pt_proposal_cls(self.model_sde, x_start, self.cdssm.S(t), self.cdssm.S(t+1), self.data[t], self.cdssm.LY(t), self.cdssm.SigmaY(t))
 
class GuidedDA(CDSSM_FeynmanKac):
    """
    Abstract Base Class for Guided Feynman-Kac Models for CD-SSMs.
    """
    
    def M0_observedat0(self, N):
        x = self.cdssm._init_dist_container(N)
        x['0.0'] = self.cdssm.proposal0(self.data).rvs(size=N)
        return x

    def logG(self, t, xp, x): # Called by `SMC`
        return self.logG0_observedat0(x) if (t == 0 and self.cdssm.isobservedat0) else self.logGt(t, xp, x)

    def logG0_observedat0(self, x):
        return (self.cdssm.x0.logpdf(x['0.0'])
        + self.cdssm.PY(0, None, x['0.0']).logpdf(self.data[0])
        - self.cdssm.proposal0(self.data).logpdf(x['0.0'])
        )
    
    def logGt(self, t, xp, x):
        raise NotImplementedError(self._err_msg('logGt'))

class ReparameterisedDA(CDSSM_FeynmanKac):

    def transform_W_to_X(self, W):
        if type(W[0]) is np.void: # Single path
            return self._sample_transform_W_to_X(W)
        else: # M>1 paths
            return self._samples_transform_W_to_X(W)

    def _sample_transform_W_to_X(self, W):
        W = [np.array(path).reshape(1) for path in W] # List of paths to list of arrays of paths
        X = self._samples_transform_W_to_X(W)
        return [path[0] for path in X] # List of arrays of paths to list of paths

    def _samples_transform_W_to_X(self, W):
        """
        Inputs     
        --------
        W: list of length T of struct_arrays. Each struct_array is of shape (M, ).   
            Contains the samples from the reparameterised pathspace smoothing distribution.

        Returns
        ---------
        X: list of length T of struct_arrays. Each struct_array is of shape (M, ).
            Samples represent the distribution of the model sde given the observed data.
        """
        X = [None] * len(W)
        M = W[0].shape[0]
        if self.cdssm.isobservedat0:
            X[0] = W[0]
        else:
            x_start = self._init_x0(M)
            x_end = W[0][W[0].dtype.names[-1]]
            aux_bridge = self._build_aux_bridge(0, x_start, x_end)
            X[0] = aux_bridge.transform_W_to_X(W[0], x_start)
        for t in range(1, len(W)):
            x_start = X[t-1][X[t-1].dtype.names[-1]]
            x_end = W[t][W[t].dtype.names[-1]]
            aux_bridge = self._build_aux_bridge(t, x_start, x_end)
            X[t] = aux_bridge.transform_W_to_X(W[t], x_start)
        return X
    
class BootstrapDA(CDSSM_FeynmanKac):
    """
    Basically the same as the standard Bootstrap PF. Only difference is that instead of 
    simulating from the transition density of the SDE, an approximate sample is generated by using a 
    numerical scheme.
    
    Observation density is univariate Gaussian: Y_t |E_t \sim N(e_t, \eta^2).

    Subclass this object and specifiy the 'ModelSDECls' to fully define a valid fk_model.

    Example:
    -----------

    class BootstrapDA_OU(BootstrapDA):
        ModelSDECls = OrnsteinUhlenbeck
    """    
    
    cls_sname = 'BootstrapDA'
    
    @property
    def sname(self):
        return self.cls_sname
    
    def M0_observedat0(self, N):
        x = self.cdssm._init_dist_container(N)
        x['0.0'] = self.cdssm.x0.rvs(size=N)
        return x
    
    def M0_else(self, N):
        return self.model_sde.simulate(N, self.cdssm.x0, t_start=self.cdssm.S(0), t_end=self.cdssm.S(1), num=self.num)
    
    def M(self, t, xp):
        t = t-1 if self.cdssm.isobservedat0 else t
        return self.model_sde.simulate(xp.shape[0], xp[xp.dtype.names[-1]], t_start=self.cdssm.S(t), t_end=self.cdssm.S(t+1), num=self.num)

    def logG(self, t, xp, x):
        return self.cdssm.PY(t, xp, x[x.dtype.names[-1]]).logpdf(self.data[t])

class BootstrapReparameterisedDA(BootstrapDA, ReparameterisedDA, AuxiliaryBridgeMixin):
    """    
    Bootstrap Feynman-Kac formalism, with reparameterised paths to enable ancestor updates.
    Can only be implemented on cdssms with model sdes that are instances of `MvEllipticSDE` or `SDE`
    """
    cls_sname = 'BsR'
    
    def __init__(self, cdssm=None, data=None, auxiliary_bridge_cls = None):
        CDSSM_FeynmanKac.__init__(self, cdssm=cdssm, data=data)
        if isinstance(self.cdssm.model_sde, sdes.HypoellipticSDE):
            raise ValueError('Bootstrap Reparametered Feynman Kac models cannot be constructed for hypoelliptic SDEs')
        self.auxiliary_bridge_cls = self.default_auxiliary_bridge_cls if auxiliary_bridge_cls is None else auxiliary_bridge_cls

    @property
    def sname(self):
        name = self.cls_sname
        bridge_ext = self.auxiliary_bridge_cls.sname if self.is1d else self.auxiliary_bridge_cls.sname[2:]
        return name + '_' + bridge_ext

    @classmethod
    def auxiliary_bridge_cls_options(cls):
        return {**cls.univ_auxiliary_bridge_cls_options(), **cls.mv_auxiliary_bridge_cls_options()}
            
    def M0_observedat0(self, N):
        return BootstrapDA.M0_observedat0(self, N)
    
    def M0_else(self, N):
        x_start = np.ones(N)*self.cdssm.x0 if self.is1d else np.concatenate([self.cdssm.x0]*N)
        x = BootstrapDA.M0_else(self, N)
        x_end = x[x.dtype.names[-1]]
        aux_bridge = self._build_aux_bridge(0, x_start, x_end)
        z = aux_bridge.transform_X_to_W(x, x_start)
        return z

    def M(self, t, xp):        
        x_start = xp[xp.dtype.names[-1]]
        x = BootstrapDA.M(self, t, xp)
        x_end = x[x.dtype.names[-1]]
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        z = aux_bridge.transform_X_to_W(x, x_start)
        return z
    
    def logpt(self, t, xp, x):
        x_start, x, x_end = self._preprocess_logpt(xp, x)
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        x = aux_bridge.transform_W_to_X(x, x_start)
        bridge_log_likelihood = aux_bridge.bridge_log_likelihood(x_start, x)
        return bridge_log_likelihood

class ForwardGuidedDA(GuidedDA, ForwardProposalMixin):
    """
    Feynman-Kac Model for Guided Forward Proposals.
    To create a Feynman-Kac class, this needs to be subclassed with:

    'ModelSDECls' and 'ProposalSDECls' defined as class attributes.
    Conditional observation distribution is set as additive noise: $Y_t | E_t=e_t \sim N(L(t)*e_t, sigmaY(t)^2)$

    Subclass this object and specifiy the 'ModelSDECls' and 'ProposalSDECls to fully define a valid fk_model.

    Example:
    -----------
    
    class ForwardGuidedDA_OU_Optimal(ForwardGuidedDA):
        ModelSDECls = OrnsteinUhlenbeck
        ProposalSDECls = LocalLinearOUProp

    """
    cls_sname = 'FwG'
    
    def __init__(self, cdssm=None, data=None, proposal_sde_cls=None):
        super().__init__(cdssm=cdssm, data=data)
        if isinstance(self.cdssm.model_sde, sdes.HypoellipticSDE):
            raise ValueError('Forward Guided Feynman Kac model is not yet implemented for hypoelliptic SDEs')
        self.proposal_sde_cls = self.default_proposal_sde_cls if proposal_sde_cls is None else proposal_sde_cls

    @property
    def sname(self):
        name = self.cls_sname
        fw_prop_ext = self.proposal_sde_cls.sname if self.dimX == 1 else self.proposal_sde_cls.sname[2:]
        return name + '_' + fw_prop_ext
    
    @classmethod
    def proposal_sde_cls_options(cls):
        return {**cls.univ_proposal_sde_cls_options(), **cls.mv_proposal_sde_cls_options()}
    
    def M0_else(self, N):
        x_start = self.cdssm.x0
        forward_proposal = self._build_forward_proposal(0, x_start)                            
        return forward_proposal.simulate(N, num=self.num)
        
    def M(self, t, xp):
        x_start = xp[xp.dtype.names[-1]]
        forward_proposal = self._build_forward_proposal(t, x_start)
        return forward_proposal.simulate(xp.shape[0], num=self.num)
    
    def logGt(self, t, xp, x):
        x_start = xp[xp.dtype.names[-1]] if t > 0 else self._init_x0(x.shape[0])
        x_end = x[x.dtype.names[-1]]
        forward_proposal = self._build_forward_proposal(t, x_start)
        obs_density_logpdf = self.cdssm.PY(t, xp, x_end).logpdf(self.data[t])
        log_girsanov_likelihood = forward_proposal.log_girsanov(x)
        return obs_density_logpdf + log_girsanov_likelihood
        
class ForwardReparameterisedDA(ForwardGuidedDA, ReparameterisedDA, AuxiliaryBridgeMixin):
    """
        Data Augmentation of forward proposals
    
        To use, one need to subclass and specify the following class methods:

        ProposalSDECls
        AuxiliaryBridgeCls

        It will be interesting to consider the impact of different auxiliary bridge constructions on algorithm performance.
    """    
    cls_sname = 'FwR'
    
    def __init__(self, cdssm=None, data=None, proposal_sde_cls=None, auxiliary_bridge_cls=None):
        ForwardGuidedDA.__init__(self, cdssm=cdssm, data=data, proposal_sde_cls=proposal_sde_cls)
        if isinstance(self.cdssm.model_sde, sdes.HypoellipticSDE):
            raise ValueError('Forward Reparametered Feynman Kac models cannot be constructed for hypoelliptic SDEs')
        self.auxiliary_bridge_cls = self.default_auxiliary_bridge_cls if auxiliary_bridge_cls is None else auxiliary_bridge_cls

    @property
    def sname(self):
        name = self.cls_sname
        bridge_ext = self.auxiliary_bridge_cls.sname if self.dimX == 1 else self.auxiliary_bridge_cls.sname[2:]
        fw_prop_ext = self.proposal_sde_cls.sname if self.dimX == 1 else self.proposal_sde_cls.sname[2:]
        return name + '_' + bridge_ext + '_' + fw_prop_ext

    @classmethod
    def auxiliary_bridge_cls_options(cls):
        return {**cls.univ_auxiliary_bridge_cls_options(),
                **cls.mv_auxiliary_bridge_cls_options()
                }

    def M0_else(self, N):
        x_start = self._init_x0(N)
        x = ForwardGuidedDA.M0_else(self, N)
        x_end = x[x.dtype.names[-1]]
        aux_bridge = self._build_aux_bridge(0, x_start, x_end)
        return aux_bridge.transform_X_to_W(x, x_start)
        
    def M(self, t, xp):
        x_start = xp[xp.dtype.names[-1]]
        x = ForwardGuidedDA.M(self, t, xp)
        x_end = x[x.dtype.names[-1]]
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        return aux_bridge.transform_X_to_W(x, x_start)

    def logGt(self, t, xp, x):
        x_start = xp[xp.dtype.names[-1]] if t > 0 else self._init_x0(x.shape[0])
        x_end = x[x.dtype.names[-1]]
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        x = aux_bridge.transform_W_to_X(x, x_start)
        return ForwardGuidedDA.logGt(self, t, xp, x)
        
    def logpt(self, t, xp, x):
        """
        The role of logpt in the smoothing algorithms. Test whether changing the transform improves performance.
        """
        x_start, x, x_end = self._preprocess_logpt(xp, x)
        forward_proposal = self._build_forward_proposal(t, x_start)
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        x = aux_bridge.transform_W_to_X(x, x_start)
        bridge_log_likelihood = aux_bridge.bridge_log_likelihood(x_start, x) # Think about whether this calculation works/ M_t(z_t|z_{t-1})
        log_girsanov_likelihood = forward_proposal.log_girsanov(x) # G_t(z_{t-1}, z_t))
        return bridge_log_likelihood + log_girsanov_likelihood

    def _build_aux_bridge(self, t, x_start, x_end):
        forward_proposal = self._build_forward_proposal(t, x_start)
        if self.cdssm.isobservedat0:
            return self.auxiliary_bridge_cls(forward_proposal, 0., self.cdssm.S(t) - self.cdssm.S(t-1), x_end)
        else:
            return self.auxiliary_bridge_cls(forward_proposal, 0., self.cdssm.S(t+1) - self.cdssm.S(t), x_end)

class BackwardGuidedDA(GuidedDA, AuxiliaryBridgeMixin, EndPointProposalMixin):
    """
    Guided Backward Proposals
    """
    cls_sname = 'BwG'
    
    def __init__(self, cdssm=None, data=None, end_pt_proposal_cls=None, auxiliary_bridge_cls=None):
        BootstrapDA.__init__(self, cdssm=cdssm, data=data)
        self.end_pt_proposal_cls = self.default_end_pt_proposal_cls if end_pt_proposal_cls is None else end_pt_proposal_cls
        self.auxiliary_bridge_cls = self.default_auxiliary_bridge_cls if auxiliary_bridge_cls is None else auxiliary_bridge_cls

    @property
    def sname(self):
        name = self.cls_sname
        bridge_ext = self.auxiliary_bridge_cls.sname if self.dimX == 1 else self.auxiliary_bridge_cls.sname[2:]
        bridge_ext = bridge_ext[1:] if isinstance(self.cdssm.model_sde, sdes.HypoellipticSDE) else bridge_ext
        end_pt_prop_ext = self.end_pt_proposal_cls.sname if self.dimX == 1 else self.end_pt_proposal_cls.sname[2:]
        end_pt_prop_ext = end_pt_prop_ext[1:] if isinstance(self.cdssm.model_sde, sdes.HypoellipticSDE) else end_pt_prop_ext
        return name + '_' + bridge_ext + '_' + end_pt_prop_ext

    @classmethod
    def auxiliary_bridge_cls_options(cls):
        return {**cls.univ_auxiliary_bridge_cls_options(),
                **cls.mv_auxiliary_bridge_cls_options(),
                **cls.integrated_auxiliary_bridge_cls_options(),
                **cls.twice_integrated_auxiliary_bridge_cls_options()
                }

    @classmethod
    def end_pt_proposal_cls_options(cls):
        return {**cls.univ_end_pt_proposal_cls_options(),
                **cls.mv_end_pt_proposal_cls_options(),
                **cls.integrated_end_pt_proposal_cls_options(),
                **cls.twice_integrated_end_pt_proposal_cls_options()
                }

    def M0_else(self, N):
        x_start = self.cdssm.x0
        end_point_proposal = self._build_end_point_proposal(0, x_start)
        x_end = end_point_proposal.rvs(N)
        aux_bridge = self._build_aux_bridge(0, x_start, x_end)
        return aux_bridge.simulate(N, self.cdssm.x0, num=self.num)
        
    def M(self, t, xp):
        N = xp.shape[0]; x_start = xp[xp.dtype.names[-1]]
        end_point_proposal = self._build_end_point_proposal(t, x_start)
        x_end = end_point_proposal.rvs(N)
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        return aux_bridge.simulate(N, x_start, num=self.num)
            
    def logGt(self, t, xp, x):
        x_start = xp[xp.dtype.names[-1]] if t > 0 else self._init_x0(x.shape[0])
        x_end = x[x.dtype.names[-1]]
        end_point_proposal = self._build_end_point_proposal(t, x_start)
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        end_pt_prop_logpdf = end_point_proposal.logpdf(x_end)
        bridge_log_likelihood = aux_bridge.bridge_log_likelihood(x_start, x)
        obs_density_logpdf = self.cdssm.PY(t, xp, x_end).logpdf(self.data[t])
        return obs_density_logpdf + bridge_log_likelihood - end_pt_prop_logpdf
            
class BackwardReparameterisedDA(BackwardGuidedDA, ReparameterisedDA):
    """For backward, reparameterised DA, the same auxiliary bridge process is used for the transform as in the proposal"""

    cls_sname = 'BwR'
    
    def __init__(self, cdssm=None, data=None, end_pt_proposal_cls=None, auxiliary_bridge_cls=None):
        BackwardGuidedDA.__init__(self, cdssm=cdssm, data=data, end_pt_proposal_cls=end_pt_proposal_cls, auxiliary_bridge_cls=auxiliary_bridge_cls)
        self.brownian_motion = sdes.BrownianMotion() if self.is1d else sdes.MvIndepBrownianMotion(dimX=self.cdssm.model_sde.dimX)  
        
    def M0_else(self, N):
        x_start = self.cdssm.x0
        end_point_proposal = self._build_end_point_proposal(0, x_start)
        x_end = end_point_proposal.rvs(N)
        W = self._simulate_W(0, N)
        W[W.dtype.names[-1]] = x_end
        return W

    def M(self, t, xp):
        N = xp.shape[0]; x_start = xp[xp.dtype.names[-1]]
        end_point_proposal = self._build_end_point_proposal(t, x_start)
        x_end = end_point_proposal.rvs(N)
        W = self._simulate_W(t, N)
        W[W.dtype.names[-1]] = x_end
        return W

    def logGt(self, t, xp, x):
        x_start = xp[xp.dtype.names[-1]] if t > 0 else self._init_x0(x.shape[0])
        x_end = x[x.dtype.names[-1]]
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        x = aux_bridge.transform_W_to_X(x, x_start)
        end_point_proposal = self._build_end_point_proposal(t, x_start)
        obs_density_logpdf = self.cdssm.PY(t, xp, x_end).logpdf(self.data[t])
        bridge_log_likelihood = aux_bridge.bridge_log_likelihood(x_start, x)
        end_pt_prop_logpdf = end_point_proposal.logpdf(x_end)
        return obs_density_logpdf + bridge_log_likelihood - end_pt_prop_logpdf
        
    def logpt(self, t, xp, x):
        """
        In ON^2 smoothing function, x is a single path, xp is all the possible previous paths
        In Particle Gibbs, x is a single path, xp is a single path

        xp: struct_array of size N
        x: struct_array of size 1
        """
        x_start, x, x_end = self._preprocess_logpt(xp, x)
        aux_bridge = self._build_aux_bridge(t, x_start, x_end)
        x = aux_bridge.transform_W_to_X(x, x_start)
        bridge_log_likelihood = aux_bridge.bridge_log_likelihood(x_start, x)
        return bridge_log_likelihood

    def _simulate_W(self, t, N):
        x_start = 0. if self.is1d else np.zeros((N, self.dimX)) 
        if self.cdssm.isobservedat0:
            W = self.brownian_motion.simulate(N, x_start, 0., self.cdssm.S(t) - self.cdssm.S(t-1), num=self.num)
        else:
            W = self.brownian_motion.simulate(N, x_start, 0., self.cdssm.S(t+1) - self.cdssm.S(t), num=self.num)
        return W

FK_FILTERING_CLASSES = set([BootstrapDA, ForwardGuidedDA, BackwardGuidedDA])
FK_SMOOTHING_CLASSES = set([BootstrapReparameterisedDA, ForwardReparameterisedDA, BackwardReparameterisedDA])
FK_HYPOELLIPTIC_CLASSES = set([BootstrapDA, BackwardGuidedDA, BackwardReparameterisedDA])
FK_CLASSES = FK_FILTERING_CLASSES.union(FK_SMOOTHING_CLASSES)

def gen_all_fk_models(cdssm, data, smoothing=False):
    """
    Constructs all possible Feynman-kac models, with the constructions built in the package.
    Stores in a dictionary that can be fed as an input to the MultiSMC funciton.
    """
    all_fk_models = {}
    proposal_kwarg_names = ['auxiliary_bridge_cls', 'proposal_sde_cls', 'end_pt_proposal_cls']
    fk_classes = FK_SMOOTHING_CLASSES if smoothing else FK_CLASSES
    if isinstance(cdssm.model_sde, sdes.HypoellipticSDE):
        fk_classes = fk_classes.intersection(FK_HYPOELLIPTIC_CLASSES) 
    # Construct all possible Feynman-kac models with the constructions built in the package:
    sde_type_str = sde_type(cdssm.model_sde) + '_'
    for fk_cls in fk_classes:
        fk_models = {}
        options_dicts =  {name: getattr(fk_cls, sde_type_str + name + '_options')()
                        for name in proposal_kwarg_names if hasattr(fk_cls, name + '_options')}
        fk_kwargs = {'cdssm': cdssm, 'data': data}
        fk_models = _gen_fk_models_rec(fk_models, options_dicts, fk_cls, fk_cls.cls_sname, fk_kwargs)
        all_fk_models.update(fk_models) 
    return all_fk_models
        
def _gen_fk_models_rec(fk_models, options_dicts, curr_fk_cls, curr_fk_name, curr_fk_kwargs):
    if not options_dicts:
        fk_models[curr_fk_name] = curr_fk_cls(**curr_fk_kwargs)
        return fk_models
    model_sde = curr_fk_kwargs['cdssm'].model_sde
    name = list(options_dicts.keys())[0]
    options_dict = options_dicts.pop(name)
    for proposal_cls_name, option in options_dict.items():
        curr_fk_kwargs[name] = option
        opt_name = option.sname if not isinstance(model_sde, sdes.MvSDE) else option.sname[2:]
        opt_name = opt_name[1:] if isinstance(model_sde, sdes.HypoellipticSDE) else opt_name
        curr_fk_name_ext = curr_fk_name + '_' + opt_name
        fk_models = _gen_fk_models_rec(fk_models, options_dicts, curr_fk_cls, curr_fk_name_ext, curr_fk_kwargs)
    options_dicts[name] = options_dict
    return fk_models

def gen_fk_models(cdssm, data, smoothing=False, fk_names=None):
    """
    Constructs a selection of fk models given an instance of 
    CDSSM and a dataset.
    """
    all_fk_models = gen_all_fk_models(cdssm, data, smoothing=smoothing)
    if fk_names is None:
        return all_fk_models
    else:
        return {name: all_fk_models[name] for name in fk_names}

def sde_type(model_sde):
    if isinstance(model_sde, sdes.IntegratedSDE):
        return 'integrated'
    if isinstance(model_sde, sdes.TwiceIntegratedSDE):
        return 'twice_integrated'
    if isinstance(model_sde, sdes.MvEllipticSDE):
        return 'mv'
    else:
        return 'univ'