"""
Module for standard state space models that are constructed from CD-SSMs. The idea is that:
when the transition density of the SDE that defines the CD-SSM is tractable, one does not 
need to use CDSSM SMC algorithms that use data augmentation. Instead, one can use standard 
particle filtering methods on discrete state and observation spaces. 

Implemented class:
---------------------------------------------------------------------------------------------

DiscreteDiscreteSSM: A class that constructs a State Space model from an instance of a CDSSM.


To use a DiscreteDiscreteSSM, the the model sde within the CD-SSM must have a tractable transition
density: this must be defined on the SDE through the method `transition_dist'. One may optionally
also define the method `optimal_proposal_dist' for the model sde, to define guided proposals for the 
discrete-discrete SSM.

- transition_dist
- optimal_proposal_dist (optional)
---------------------------------------------------------------------------------------------
"""
import particles.state_space_models as ssms

class DiscreteDiscreteSSM(ssms.StateSpaceModel):
    """
    Constructs a State Space model from an instance of a CDSSM.
    Requires that the following methods are defined for the model 
    sde within the CDSSM:
    
    - transition_dist
    - optimal_proposal_dist (optional)
    
    The observation density is taken from the input CDSSM.
    
    Kalman filtering/smoothing is not compatible with with class, however
    it can be applied to time inhomogeneous CDSSMs.
    """    
    def __init__(self, cdssm):
        self.cdssm = cdssm

    def PX0(self):
        if self.cdssm.isobservedat0:
            return self.cdssm.x0 # ProbDist object
        else:
            return self.cdssm.model_sde.transition_dist(self.cdssm.S(0), self.cdssm.S(1), self.cdssm.x0)

    def PX(self, t, xp):
        if self.cdssm.isobservedat0:
            return self.cdssm.model_sde.transition_dist(self.cdssm.S(t-1), self.cdssm.S(t), xp)
        else:
            return self.cdssm.model_sde.transition_dist(self.cdssm.S(t), self.cdssm.S(t+1), xp)

    def PY(self, t, xp, x):
        return self.cdssm.PY(t, xp, x)

    def proposal0(self, data):
        if self.cdssm.isobservedat0:
           return self.cdssm.proposal0(data)
        else:
            return self.cdssm.model_sde.optimal_proposal_dist(self.cdssm.S(0), self.cdssm.S(1), self.cdssm.x0, data[0], self.cdssm.LY(0), self.cdssm.SigmaY(0))

    def proposal(self, t, xp, data):
        if self.cdssm.isobservedat0:
            return self.cdssm.model_sde.optimal_proposal_dist(self.cdssm.S(t-1), self.cdssm.S(t), xp, data[t], self.cdssm.LY(t), self.cdssm.SigmaY(t))
        else:
             return self.cdssm.model_sde.optimal_proposal_dist(self.cdssm.S(t), self.cdssm.S(t+1), xp, data[t], self.cdssm.LY(t), self.cdssm.SigmaY(t))        
        

# ------------------------------------------------------------ Deprecated -------------------------------------------------
# --------------- Standard State Space Model representations of LGSSMs for numerical experiment benchmarks ----------------

# class ReparamLinearGauss(LinearGauss):
    
#     default_params = {'sigmaX_2': 0.2,
#                       'rho': 0.9,
#                       }
#     """
#     A LGSSM that has been reparameterised to enable conjugacy results
#     to be used in joint inference. This is a nice example of how to
#     reparameterise a state space model.
#     """

#     def __init__(self, **kwargs):
#         StateSpaceModel.__init__(self, **kwargs)
#         orig_params = self.params_map(**{**self.default_params, **kwargs})
#         super().__init__(**orig_params)

#     def params_map(self, sigmaX_2, rho):
#         """
#         """
#         orig_params = {'sigma0': np.sqrt(sigmaX_2), 'sigmaX': np.sqrt(sigmaX_2), 'rho': rho, 'sigmaY': 0.01}
#         return orig_params
    
#     @classmethod
#     def prior(cls, alpha_X=3., beta_X=1., lmda=1., mu=0.): #alpha_Y=3., beta_Y=1.):
#         """
#         Constructs a prior distribution for this state space model.
#         This prior distribution is conjugate for this model.         
#         """
#         local_vars = locals()
#         hyperparams = {k: local_vars[k] for k in ["alpha_X", "beta_X", "lmda", "mu"]}
#         lgssm_prior_dict = OrderedDict()
#         lgssm_prior_dict['sigmaX_2'] = dists.InvGamma(a=alpha_X, b=beta_X)
#         lgssm_prior_dict['rho'] = dists.Cond(lambda theta: dists.Normal(loc=mu, scale=np.sqrt(theta['sigmaX_2']/lmda)))
#         # lgssm_prior_dict['sigmaY_2'] = dists.Cond(lambda theta: dists.InvGamma(a=alpha_Y, b=beta_Y))
#         prior = dists.StructDist(lgssm_prior_dict)
#         prior.hyperparams = hyperparams
#         return prior

#     @classmethod
#     def posterior(cls, x, y, alpha_X=3., beta_X=1., lmda=1., mu=0.): # , alpha_Y=3., beta_Y=3.):
#         """
#         Constructs a posterior distribution for the parameters of this 
#         state space model, given prior parameters and data x, y.
#         """
#         T = len(x)
#         a = np.sum(x[:-1]*x[:-1]) + lmda
#         b = lmda*mu + np.sum(x[:-1]*x[1:])
#         c = np.sum(x*x) + 2*beta_X + lmda * (mu ** 2)
#         post_params = {}
#         post_params['alpha_X'] = 0.5*T + alpha_X
#         post_params['beta_X'] = 0.5*(c - b*b/a)
#         post_params['mu'] = b/a
#         post_params['lmda'] = a
#         # post_params['alpha_Y'] = 0.5*T + alpha_Y
#         # post_params['beta_Y'] = beta_Y + 0.5*np.sum((x-y)*(x-y))
#         return cls.prior(**post_params)