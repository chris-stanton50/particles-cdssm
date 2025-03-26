# -*- coding: utf-8 -*-
"""
SDEs module

Define objects that represent the solutions of stochastic diffeeential equations (SDEs).

To build an SDE, subclass one of the following:

- SDE: For SDEs such that dimX=dimW=1
- MvEllipticSDE: For any other SDE with dimX=dimW>1 and an invertible diffusion coefficient
- MvIntegratedSDE: For SDEs with dimX // 2 = dimW. The smooth component is an integral of hte rough component.
- MvTwiceIntegratedSDE: For SDEs with dimX // 2 = dimW. The smooth components are twice integrals of the rough component. 

In further development, one could 
Examples of Univariate SDEs:

1. SinDiffusion: SDE with a sinusoidal drift and constant diffusion
2. ArctanDiffusion: SDE with an arctan drift and constant diffusion
3. LogisticDiffusion: SDE with logistic growth and constant diffusion

Examples of LinearSDEs:

1. BrownianMotion: SDE of the Brownian Motion
2. OrnsteinUhlenbeck: The Ornstein-Uhlenbeck SDE
3. TVOrnsteinUhlenbeck: Time varying Ornstein Uhlenbeck process
4. BrownianBridge: Brownian Bridge SDE

Examples of Multivariate LinearSDEs:

1. MvIndependentBrownianMotion: Multivariate Independent Brownian Motion SDE
2. MvBrownianMotion: Multivariate Brownian Motion SDE
3. MvOrnsteinUhlenbeck: Multivariate Ornstein-Uhlenbeck SDE
4. MvIndependentOrnsteinUhlenbeck: Multivariate Independent Ornstein-Uhlenbeck SDE
5. MvFullOrnsteinUhlenbeck: Multivariate Full Ornstein-Uhlenbeck SDE. Evaluates Matrix exponential.
"""

import math
import numpy as np
import scipy.linalg as sla
from particles.distributions import Normal, VaryingCovNormal, MvNormal
from sdes.numerical_schemes import EulerMaruyama, LinearExact, MvEulerMaruyama, HypoellipticEulerMaruyama, MvLinearExact
from sdes.tools import MeanAndCov, filter_step_var_cov, mv_filter_step_var_cov, vectorise_param, grad_log_linear_gaussian, grad_grad_log_linear_gaussian, mv_grad_log_linear_gaussian, mv_grad_grad_log_linear_gaussian
from sdes.tools import get_methods, vec_grad_log_linear_gaussian


#---------------------------------- Abstract Base Class for Univariate SDEs ----------------------------------


class SDEBase(object):
    """
    Base class for all SDEs. To create an SDE, the following methods need to be defined:

    b: The drift coefficient
    sigma: The diffusion coefficient

    Optionally, one can also define: 

    db: First derivative of the drift coefficient
    dsigma: First derivative of the diffusion coefficient

    Given an SDE, there two primary functionalities: simulation, and transformation.

    To simulate a sample path from an SDE, one can use the method `simulate', to 
    generate samples from a start time to an end time, for given starting value.

    A 1D SDE implies existence of a unique function from the driving noise to the 
    SDE solution, a discretisation of this map, which is derived from the natural 
    approach to discretising integrals, is implemented as the method 
    `transform_W_to_X'. 

    For elliptic SDEs, there also exists a function from the SDE solution to the
    driving noise. This function is implemented, again through a natural 
    discretisation, using the method `transform_X_to_W'.

    All 3 of these methods are implemented in a vectorised form, (see the functions
    `X_to_W_step', `W_to_X_step' and 'univ_simulation_step'). In practice, this 
    means that one can say generate N simulations from N different starting points
    from a start time t_start to an end time t_end, using a single call to the method.
    """
    def __init__(self, **kwargs):
        if hasattr(self, 'default_params'):
            self.__dict__.update(self.default_params)
        self.__dict__.update(kwargs)

    @property
    def isLinear(self):
        return isinstance(self, LinearSDE) or isinstance(self, MvLinearSDE)

    @property
    def params(self):
        return {k: self.__dict__[k] for k in self.default_params.keys()}

    def _error_msg(self, method):
        return ('method ' + method + ' not implemented in SDE class %s' %
                self.__class__.__name__)

    def simulate(self, size: int, x_start=None, t_start: float =0., t_end: float =1., num=5) -> np.ndarray:
        """
        Method to generate sample paths from a 1D SDE: implementation is vectorised, so simulation 
        using multiple start points is possible.

        Inputs
        ------------
        size (int):           The number of simulations to generate. If x_start is a vector, must match dimension.
        x_start (array-like): The starting point(s) for the simulation. For a single start point, can be 
                              set to a float, otherwise set to a vector. If set to a vector, size must 
                              match the vector dimension.
        t_start (float):      The starting time of the simulation. 
        t_end (float):        The ending time of the simulation.
        num (int):            The number of imputed points used in the simulation. Low values will have 
                              greater bias but reduce time cost.

        Returns
        ------------
        simulations (np.ndarray): A structured array (structured by timestamps) containing simulation outputs. 
        """
        self.numerical_scheme = self.numerical_scheme_cls(self)
        simulations = self.numerical_scheme.simulate(size=size, t_start=t_start, t_end=t_end, x_start=x_start, num=num)
        return simulations
    
    def transform_X_to_W(self, X: np.ndarray, t_start: float, x_start) -> np.ndarray:
        """
        Method to apply the map from the solution of the SDE to the driving noise, using discretisation.
        Transform is uniquely determined by a drift and diffusion coefficient, a start time, end time and
         a start point. End time is implied by the input X, so is not an input to the function.
        Implementation is vectorised in the start point, so a different transform can be applied to a 
        each of the simulations in the input X.

        Inputs
        ------------
        X (np.ndarray):         Structured array of simulations to which to apply the transform
        t_start (float):        Start time of the SDE
        x_start (array-like)    Start value for the transform. Can either be a float, in which case the 
                                same transform is applied to every simulation

        Returns
        ------------
        W (np.ndarray):         Structured array of the same shape and dtype of 'X', that contains the 
                                transform outputs.
        """
        W =self.numerical_scheme_cls.transform_X_to_W(X, t_start=t_start, x_start=x_start, transform_end_point=True)
        return W
    
    def transform_W_to_X(self, W: np.ndarray, t_start: float, x_start) -> np.ndarray:
        """
        Method to apply the map from the driving noise to the solution of the SDE, using discretisation.
        Transform is uniquely determined by a drift and diffusion coefficient, a start time, end time and
         a start point. End time is implied by the input W, so is not an input to the function.
        Implementation is vectorised in the start point, so a different transform can be applied to each
        of the simulations in the input W.

        Inputs
        ------------
        W (np.ndarray):         Structured array of simulations to which to apply the transform
        t_start (float):        Start time of the SDE
        x_start (array-like):   Start value for the transform. Can either be a float, in which case the 
                                same transform is applied to every simulation

        Returns
        ------------
        X (np.ndarray):         Structured array of the same shape and dtype of 'W', that contains the 
                                transform outputs.
        """
        X = self.numerical_scheme.transform_W_to_X(W, t_start=t_start, x_start=x_start, transform_end_point=True)
        return X


class SDE(SDEBase):
    dimX = 1
    dimW = 1
    numerical_scheme_cls = EulerMaruyama

    def b(self, t: float, x: np.ndarray):
        """
        **Mandatory Function**
        Placeholder for drift coeffciient of an SDE

        Inputs
        ------------
        t (float): time
        x (float/np.ndarray): float or (N,) array of current states
    
        Returns
        ------------
        b (float/np.ndarray): float or (N,) array of drift coefficient values
        """
        raise NotImplementedError(self._error_msg('b'))

    def sigma(self, t: float, x):
        """
        **Mandatory Function**
        Placeholder for drift coefficient of an SDE

        Inputs
        ------------
        t (float): time
        x (float/np.ndarray): float or (N,) array of current states
    
        Returns
        ------------
        sigma (float/np.ndarray): float or (N,) array of diffusion coefficient values
        """
        raise NotImplementedError(self._error_msg('sigma'))

    def db(self, t: float, x):
        """
        **Optional Function**
        Placeholder for derivative of the drift coefficient of an SDE.
        Used in higher order numerical schemes, and in finding coefficients 
        of 1st order linear SDE approximations.

        Inputs
        ------------
        t (float): time
        x (float/np.ndarray): float / (N,) array of current states
    
        Returns
        ------------
        db (float/np.ndarray): float / (N,) array of db values
        """
        return NotImplementedError(self._error_msg('db'))

    def dsigma(self, t: float, x):
        """
        **Optional Function**
        Placeholder for derivative of the diffusion coefficient of an SDE.
        Used in higher order numerical schemes.

        Inputs
        ------------
        t (float): time
        x (float/np.ndarray): float / (N,) array of current states
    
        Returns
        ------------
        dsigma (float/np.ndarray): float / (N,) array of db values
        """
        raise NotImplementedError(self._error_msg('dsigma'))

    def Cov(self, t: float, x: float):
        return self.sigma(t, x) ** 2

    def dCov(self, t: float, x: float):
        return 2 * self.sigma(t, x) * self.dsigma(t, x)

    def transition_dist(self, s: float, t: float, x_s: np.ndarray):
        """
        **Optional Function**
        Placeholder for transition density of the SDE. In general, this is 
        unknown, but for some SDEs, it can be computed exactly.
        Used to run SMC algorithms on SDEs with Exact Proposals 
        see the class (sdes.state_space_models.DiscreteDiscreteSSM)

        Inputs
        ------------
        s (float): start time
        t (float): end time
        x_s (float/np.ndarray): float / (N,) array of current states
    
        Returns
        ------------
        transition_dist (particles.distributions.ProbDist): ProbDist object representing the transition density of x_t|x_s.
        """
        raise NotImplementedError(self._error_msg('transition_dist'))
    
    def optimal_proposal_dist(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        **Optional Function**
        Placeholder for derivative of the diffusion coefficient of an SDE.
        Used in higher order numerical schemes.

        Inputs
        ------------
        s (float): start time
        t (float): end time
        x_s (float/np.ndarray): float / (N,) array of current states
        y_t (float/np.ndarray): float / (1,) array for the observation
    
        Returns
        ------------
        optimal_dist (particles.distributions.ProbDist): ProbDist object representing the optimal proposal for x_t|x_s.
        """
        raise NotImplementedError(self._error_msg('optimal_proposal_dist'))

#---------------------------------- Examples of non-linear, Univariate SDEs ----------------------------------

class SinDiffusion(SDE):
    """
    The sin diffusion SDE, given by:
    $dX_t = sin(x -\theta_1) dt + \theta_2 dW_t$
    """
    default_params = {'theta_1': 0., 
                      'theta_2': 1.,
                      'sigma_0': 1.}
    
    def b(self, t, x):
        return np.sin(x - self.theta_1)

    def sigma(self, t, x):
        return self.theta_2

    def db(self, t, x):
        return np.cos(x - self.theta_1)
    
    def dsigma(self, t, x):
        return 0.

class ArctanDiffusion(SDE):
    """
    The arctan diffusion, as used in VanDerMeulen & Schauer (2017), Example 4.3:

    $dX_t = \alpha \arctan(X_t) + \beta dt + \phi dW_t$

    Under certain conditions on the parameters, the process mean reverts to -tan(\beta/\alpha)
    """
    default_params = {'alpha': -2.,
                      'beta': 0.,
                      'phi': 0.75}
    
    def b(self, t, x):
        return self.alpha * np.arctan(x) + self.beta
    
    def sigma(self, t, x):
        return self.phi
    
    def db(self, t, x):
        return self.alpha/(1 + (x ** 2))
    
    def dsigma(self, t, x):
        return 0.

class LogisticDiffusion(SDE):
    """
    The logistic (Verhulst) growth population model, under stochastic shocks:

    $dX_t = r X_t (1-X_t/K) dt + sX_t dW_t$
    
    r: Intrinsic growth rate
    K: Carrying capacity
    s: Noise intensity
    
    A reasonable initial condition is X_0 = K/10
    """
    
    default_params = {'r': 0.5,
                      'K': 100,
                      's': 0.1}
    
    def b(self, t, x):
        return self.r * x * (1 - x/self.K)

    def db(self, t, x):
        return self.r * (1 - 2*x/self.K)
        
    def sigma(self, t, x):
        return self.s * x
    
    def dsigma(self, t, x):
        return self.s

#---------------------------------- Abstract base classes for Linear, Univariate SDEs ----------------------------------

class LinearSDE(SDE):
    """
    Linear SDE, of the form:

    $$dX_t = [A(t) + B(t)X_t] dt + C(t)dW_t$$


    The following methods from `SDE` are automatically defined:
    
    'b'
    'sigma'
    'db'
    'dsigma'    
    'transition_dist' 
    'optimal_proposal_dist'

    Includes the additional methods:
    'log_px' -  Log transition density of X(t) | X(s).
    'grad_log_px' - Grad of log transition density of X(t) | X(s).
    'grad_log_py' - Grad of log transition density of Y_t | X(s). Assumes a linear Gaussian observation density. 
    'exact_simulate' - Exact simulation from the Linear SDE

    Many of the above new methods depend on the transition density of the linear SDE, which is 
    linear, Gaussian, given by:

    p_{s,t}(x_t| x_s) = \phi(x_t, a x_s + b, v)

    a, b and v correspond to _a, _b and _v methods.

    To subclass a linear SDE, we need to define the following methods:
    
    'A(t)': Additive constant in the drift coefficient
    'B(t)': Coefficient of x in the Drift coefficient
    'C(t)': Diffusion coefficient
    
    '_a(s, t)': Coefficient of x_s in transition density mean function.
    '_b(s, t)': Additive constant in transition density mean function.
    '_v(s, t)': Transition density variance
    """
    exact_scheme = LinearExact

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vec_params = False

    def A(self, t):
        raise NotImplementedError(self._error_msg('A'))
    
    def B(self, t):
        raise NotImplementedError(self._error_msg('B'))

    def C(self, t):
        raise NotImplementedError(self._error_msg('C'))

    def _a(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_a'))

    def _b(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_b')) 
    
    def _v(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_v'))

    def _a_vec(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_a_vec'))

    def _b_vec(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_b_vec')) 
    
    def _v_vec(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_v_vec'))
    
    def b(self, t, x):
        return self.A(t) + self.B(t) * x
    
    def sigma(self, t, x):
        return self.C(t)

    def b_vec(self, t, x):
        return self.A_vec(t) + self.B_vec(t) * x
    
    def sigma_vec(self, t, x):
        return self.C_vec(t)

    def Cov_vec(self, t, x):
        return self.sigma_vec(t, x) ** 2

    def db(self, t, x):
        return self.B(t)
    
    def dsigma(self, t, x):
        return 0.
    
    def grad_log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        Gradient of the log transition density
        """

        # When simulating:

        # s: float
        # t: float
        # x_s: (N, )
        # x_t: (N, )

        return grad_log_linear_gaussian(x_s, x_t, self._a(s, t), self._b(s, t), self._v(s, t))

    def _vec_grad_log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        Vectorised implementation of the gradient of the log transition density, for use in 
        the evaluation of path integrals.
        s: (num+1, )
        t: float    
        x_s (N, num+1)
        x_t (N, )
        """
        return vec_grad_log_linear_gaussian(x_s, x_t, self._a_vec(s, t), self._b_vec(s, t), self._v_vec(s, t))
    
    def _vec_grad_grad_log_px(self, s: float, t: float):
        """
        s: (num+1, )
        t: float
        """
        return grad_grad_log_linear_gaussian(self._a_vec(s, t), self._b_vec(s, t), self._v_vec(s, t))
    
    def grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        Gradient of the log transition density of Y_t | X(s), where Y_t | E_t \sim N(Le_t, \sigma_Y^2)

        # When simulating:

        # s: float
        # t: float
        # x_s: (N, ) 
        # y_t: (1, ) / (1, dimY)
        # eta_sq: float / (dimY, dimY)
        """
        v = (LY**2)*self._v(s, t) + (sigmaY ** 2)
        return grad_log_linear_gaussian(x_s, y_t, LY*self._a(s, t), LY*self._b(s, t), v)

    def grad_grad_log_py(self, s: float, t: float, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        Second derivative of the gradient of the log transition density of Y_t | X(s), where Y_t | E_t \sim N(Le_t, \sigma_Y^2)
        s: float
        t: float
        y_t: (1, ) / (1, dimY)
        LY: float / (dimY, 1)
        sigmaY: float / (dimY, dimY)
        """
        v = (LY**2)*self._v(s, t) + sigmaY ** 2
        return grad_grad_log_linear_gaussian(self._a(s, t), self._b(s, t), v)
    
    def _vec_grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        When implementing a Girsanov integral:

        s (num+1, )
        t: float ()
        x_s: (N, num+1)
        y_1: float 
        eta_eq: float        
        """
        if not self.vec_params:
            self.generate_vec_params(len(s))
        v = (LY**2)*self._v_vec(s, t) + sigmaY ** 2
        return grad_log_linear_gaussian(x_s, y_t, LY*self._a_vec(s, t), LY*self._b_vec(s, t), v)
    
    def exact_simulate(self, size: int, x_start, t_start: float = 0., t_end: float = 1., num=5) -> np.ndarray:
        """
        Exact simulation of the linear SDE
        """
        return self.exact_scheme.simulate(size=size, t_start=t_start, t_end=t_end, x_start=x_start, num=num)

    def transition_dist(self, s: float, t: float, x_s: np.ndarray):
        """
        Transition density of the linear SDE, output as a distribution object:
        """
        a = self._a(s, t); b = self._b(s, t); v = self._v(s,t)
        return Normal(loc=a*x_s + b, scale=np.sqrt(v))

    def optimal_proposal_dist(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        Proposal for the end point using the exact distribution $E_t | E_{t-1}=e_{t-1}, Y_t = y_t$ from a Linear SDE.
        No longer used in the CDSSM SMC algorithms, but kept for reference.
        x_s: (N, )
        y_t: (1, )
        LY: float
        sigmaY: float
        """    
        a = self._a(s, t); b = self._b(s, t); v = self._v(s,t)
        pred = MeanAndCov(a*x_s + b, v)
        opt_prop_mean, opt_prop_var = filter_step_var_cov(LY, sigmaY ** 2, pred, y_t)
        return Normal(loc=opt_prop_mean, scale=np.sqrt(opt_prop_var))
    
    def generate_vec_params(self, num_plus_1):
        vec_params = {}
        for k in self.default_params.keys():
            param = self.__dict__[k]
            vec_params['vec_' + k] = vectorise_param(param, num_plus_1)
        self.__dict__.update(vec_params)
        self.vec_params = True
    
    def grad_param_log_px(self, s, t, x, xf, param_name):
        v=self._v;  k = self._k
        dv = self.df_dp('v', param_name); dk = self.df_dp('k', param_name)
        grad_param_log_px = -0.5*dv(s, t)
        grad_param_log_px +=-0.5*(v(s, t)*dk(s, t, x, param_name) - k(s, t, x, xf)*dv(s, t))/(v(s, t) ** 2)
        return grad_param_log_px

    def _k(self, s, t, x, xf):
        return (xf - self._a(s, t) * x - self._b(s, t)) ** 2
    
    def df_dp(self, f: str, param_name: str):
        if f == 'k':
            da_dp = self.df_dp('a', param_name=param_name); db_dp = self.df_dp('b', param_name=param_name)
            def dk_dp(s, t, x):
                return -2. * (da_dp(s, t) * x + db_dp(s, t))
            return dk_dp
        elif f in ['a', 'b', 'v']:
            return getattr(self, f'_d{f}_d{param_name}')
        else: 
            raise ValueError("Input 'f' must be in [a, b, v, k]")

class TimeLinearSDE(LinearSDE):
    """
    Linear SDE that is not state dependent.
    $$dX_t = A(t)dt + C(t)dW_t$$.

    To use this class, one needs subclass, and to define methods: 
    'A', 'int_A', 'C', 'int_C_sq'.

    In this class of Linear SDEs, the transition density does not depend on start point 'x_0'
    """
    def B(self, t):
        return 0.

    def B_vec(self, t):
        return 0.
       
    def int_A(self, t):
        raise NotImplementedError(self._error_msg('int_A'))
    
    def int_A_vec(self, t):
        raise NotImplementedError(self._error_msg('int_A_vec'))

    def int_C_sq(self, t):
        raise NotImplementedError(self._error_msg('int_C_sq'))

    def int_C_sq_vec(self, t):
        raise NotImplementedError(self._error_msg('int_C_sq_vec'))
    
    def _a(self, s, t):
        return 1.

    def _a_vec(self, s, t):
        return 1.
    
    def _b(self, s, t):
        return self.int_A(t) - self.int_A(s)

    def _b_vec(self, s, t):
        return self.int_A_vec(t) - self.int_A_vec(s)
        
    def _v(self, s, t):
        return self.int_C_sq(t) - self.int_C_sq(s)
    
    def _v_vec(self, s, t):
        return self.int_C_sq_vec(t) - self.int_C_sq_vec(s)

#---------------------------------- Examples of Linear, Univariate SDEs ----------------------------------

class BrownianMotion(TimeLinearSDE):
    """
    SDE of the Brownian Motion:
    $$dX_t = m dt + s dW_t$$
    """
    default_params = {'m': 0.,
                      's': 1.
                      }

    def A(self, t):
        return self.m

    def A_vec(self, t):
        if not self.vec_params:
            self.generate_vec_params(len(t))
        return self.vec_m
    
    def int_A(self, t):
        return self.m * t

    def int_A_vec(self, t):
        return self.vec_m * t
    
    def C(self, t):
        return self.s

    def C_vec(self, t):
        return self.vec_s
    
    def int_C_sq(self, t):
        return (self.s ** 2) * t
    
    def int_C_sq_vec(self, t):
        return (self.vec_s ** 2) * t

class OrnsteinUhlenbeck(LinearSDE):
    """
    The Ornstein-Uhlenbeck SDE, given by:
    $dX_t = \rho(\mu - X_t)dt + \phi dW_t$
    """
    default_params = {'rho': 0.2,
                      'mu': 0.,
                      'phi': 1.
                      }
    
    def A(self, t):
        return self.rho * self.mu
    
    def B(self, t):
        return -1. * self.rho

    def C(self, t):
        return self.phi

    def A_vec(self, t):
        if not self.vec_params:
            self.generate_vec_params(len(t))
        return self.vec_rho * self.vec_mu

    def B_vec(self, t):
        return -1 * self.vec_rho

    def C_vec(self, t):
        return self.vec_phi

    def _a(self, s, t):
        return np.exp(-self.rho * (t-s))

    def _a_vec(self, s, t):
        return np.exp(-self.vec_rho * (t-s))

    def _b(self, s, t):
        return self.mu * (1. - self._a(s, t))

    def _b_vec(self, s, t):
        return self.vec_mu *(1. - self._a_vec(s, t))
    
    def _v(self, s, t):
        return ((self.phi ** 2)/(2*self.rho)) * (1 - np.exp(-2*self.rho*(t-s)))

    def _v_vec(self, s, t):
        return ((self.vec_phi ** 2)/(2*self.vec_rho)) * (1 - np.exp(-2*self.vec_rho*(t-s)))

    def _da_drho(self, s, t):
        return -(t-s) * self._a(s, t)
    
    def _da_dphi(self, s, t):
        return 0.
    
    def _da_dmu(self, s, t):
        return 0.
    
    def _db_drho(self, s, t):
        return -self.mu * (t-s) * self._a(s, t)

    def _db_dphi(self, s, t):
        return 0.
    
    def _db_dmu(self, s, t):
        return (1. - self._a(s, t))
        
    def _dv_drho(self, s, t):
        return ((self.phi ** 2)/(2*self.rho)) * ((2.*((t-s) + (1./(2.*self.rho)))*np.exp(-2.*self.rho*(t-s)))- (1./self.rho))

    def _dv_dphi(self, s, t):
        return 2./self.phi * self._v(s, t)

    def _dv_dmu(self, s, t):
        return 0.
    
class TVOrnsteinUhlenbeck(OrnsteinUhlenbeck):
    """
    Time varying Ornstein Uhlenbeck process to use to test the smoothing algorithms:
    
    $dX_t = \rho(\mu - X_t)dt + C(t) dW_t$

    Where C(t) = \phi_1 for t \in [0,1]
               = \phi_2 for t > 1 
    """
    default_params = {'rho': 0.2,
                      'mu': 0.,
                      'phi_1': 0.3,
                      'phi_2': 0.1
                      }
    
    def C(self, t):
        return self.phi_1 * (t < 1) + self.phi_2 *(t >= 1) 
        
    def C_vec(self, t):
        return self.vec_phi_1 * (t < 1) + self.vec_phi_2 * (t >= 1)

    def _v(self, s, t):
        return (s>=1)*self._v_s_geq1(s, t) + (t <= 1)*self._v_t_leq1(s, t) + (s<1) * (t>1) * self._v_t_else(s, t)
    
    def _v_s_geq1(self, s, t):
        return ((self.phi_2 ** 2)/(2*self.rho)) * (1 - np.exp(-2.*self.rho*(t-s)))
    
    def _v_t_leq1(self, s, t):
        return ((self.phi_1 ** 2)/(2*self.rho)) * (1 - np.exp(-2.*self.rho*(t-s)))
    
    def _v_t_else(self, s, t):
            v_t  = (self.phi_2 ** 2)* (1 - np.exp(-2.*self.rho*(t-1.)))
            v_t += (self.phi_1 ** 2)* (np.exp(-2.*self.rho*(t-1.)) - np.exp(-2.*self.rho*(t-s)))
            v_t = v_t/(2.*self.rho)
            return v_t
    
    def _v_vec(self, s, t):
        return (s>=1)*self._v_s_geq1_vec(s, t) + (t <= 1)*self._v_t_leq1_vec(s, t) + (s<1) * (t>1) * self._v_t_else_vec(s, t)
    
    def _v_s_geq1_vec(self, s, t):
        return ((self.vec_phi_2 ** 2)/(2*self.vec_rho)) * (1 - np.exp(-2.*self.vec_rho*(t-s)))
    
    def _v_t_leq1_vec(self, s, t):
        return ((self.vec_phi_1 ** 2)/(2*self.vec_rho)) * (1 - np.exp(-2.*self.vec_rho*(t-s)))
    
    def _v_t_else_vec(self, s, t):
            v_t  = (self.vec_phi_2 ** 2)* (1 - np.exp(-2.*self.vec_rho*(t-1.)))
            v_t += (self.vec_phi_1 ** 2)* (np.exp(-2.*self.vec_rho*(t-1.)) - np.exp(-2.*self.vec_rho*(t-s)))
            v_t = v_t/(2.*self.vec_rho)
            return v_t

class BrownianBridge(LinearSDE):
    """
    dX_t = \frac{x^* - X_t}{T-t}dt + dW_t

    The Brownian Bridge SDE.

    Parameters:
    End point: x^*
    End time: T (This bit could be reformulated so that this is a Bridge construction)

    This definition could be extended to add a general constant in the diffusion coefficient:
    """
    default_params = {'x_end': 0.,
                      'T': 1.}
        
    def A(self, t):
        self.x_end/(self.T - t)

    def B(self, t):
        return -1./(self.T - t)
    
    def C(self, t):
        return 1.

    def _a(self, s, t):
        return (self.T - t)/(self.T-s)

    def _b(self, s, t):
        return (t-s)/(self.T - s) * self.x_end
        
    def _v(self, s, t):
        return (t-s)*(self.T-t)/(self.T - s)
    
#---------------------------------- Abstract Base Classes of Multivariate SDEs ----------------------------------
#---------------------- Includes Base Classes for both the Elliptic and Hypoelliptic Case -----------------------

class MvSDE(SDEBase):
    """
    Subclass this class to create multivariate SDEs.

    You will need to define:

    dimX: the dimension of the state X
    dimW: The dimension of the driving Brownian noise

    _diag_cov: Whether the covariance matrix of the diffusion is diagonal

    b: The Drift function: Should map from (N, dimX) to (N, dimX).
    sigma: The Diffusion function: Should map from (1, dimX) to (dimX, dimW)
    
    Optionally:
    
    db: First derivative of the drift coefficient: (N, dimX) -> (N, dimX, dimX)
    dsigma: First derivative of the diffusion coefficient: (N, dimX) -> (N, dimX, dimX, dimW)
    """
    
    # Simulation still possible for hypoelliptic diffusions.
    numerical_scheme_cls = MvEulerMaruyama

    def transition_dist(self, s: float, t: float, x_s: np.ndarray):
        """
        **Optional Function**
        Placeholder for transition density of the SDE. In general, this is 
        unknown, but for some SDEs, it can be computed exactly.
        Used to run SMC algorithms on SDEs with Exact Proposals 
        see the class (sdes.state_space_models.DiscreteDiscreteSSM)

        Inputs
        ------------
        s (float): start time
        t (float): end time
        x_s (np.ndarray): (N, dimX) array of current states
    
        Returns
        ------------
        transition_dist (particles.distributions.ProbDist): Mv ProbDist object representing the transition density of x_t|x_s.
        """
        raise NotImplementedError(self._error_msg('transition_dist'))
    
    def optimal_proposal_dist(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        **Optional Function**
        Placeholder for derivative of the diffusion coefficient of an SDE.
        Used in higher order numerical schemes.

        Inputs
        ------------
        s (float): start time
        t (float): end time
        x_s (np.ndarray): (N, dimX) array of current states
        y_t (np.ndarray): (1, dimY) array for the observation
    
        Returns
        ------------
        optimal_dist (particles.distributions.ProbDist): Mv ProbDist object representing the optimal proposal for x_t|x_s.
        """
        raise NotImplementedError(self._error_msg('optimal_proposal_dist'))
            
    def Cov(self, t: float, x: np.ndarray):
        """  
        Input an array-like of shape (N, dimX)
        Output an ndarray of shape (N, dimX, dimX)
        """
        rt_cov = self.sigma(t, x)
        rt_cov_T = np.einsum('ijk->ikj', rt_cov)
        return rt_cov @ rt_cov_T

    def dCov(self, t: float, x: float):
        #     # self.sigma(t, x) (N, dimX, dimX)
        #     return np.einsum('ijkl,',self.sigma(t, x), self.dsigma(t, x)) + np.einsum(''self.dsigma(t, x), self.sigma(t, x))
        raise NotImplementedError(self._error_msg('dCov'))
    
    @property
    def dimW(self):
        return self.dimX // (self.n_smooth + 1)

class MvEllipticSDE(MvSDE):

    n_smooth = 0
    
    def b(self, t: float, x: np.ndarray):
        """
        **Mandatory Function**
        Placeholder for drift coeffciient of an SDE

        Inputs
        ------------
        t (float): time
        x (np.ndarray): (N, dimX) array of current states
    
        Returns
        ------------
        b (np.ndarray): (N, dimX) array of drift coefficient values
        """
        raise NotImplementedError(self._error_msg('b'))

    def sigma(self, t: float, x):
        """
        **Mandatory Function**
        Placeholder for drift coefficient of an SDE

        Inputs
        ------------
        t (float): time
        x (np.ndarray): (N, dimX) array of current states
    
        Returns
        ------------
        sigma (np.ndarray): (N, dimX, dimW) array of diffusion coefficient values
        """
        raise NotImplementedError(self._error_msg('sigma'))

    def db(self, t: float, x):
        """
        **Optional Function**
        Placeholder for derivative of the drift coefficient of an SDE.
        Used in higher order numerical schemes, and in finding coefficients 
        of 1st order linear SDE approximations.

        Inputs
        ------------
        t (float): time
        x (np.ndarray): (N, dimX) array of current states
    
        Returns
        ------------
        db (np.ndarray): (N, dimX, dimX) array of db values
        """
        return NotImplementedError(self._error_msg('db'))

    def dsigma(self, t: float, x):
        """
        **Optional Function**
        Placeholder for derivative of the diffusion coefficient of an SDE.
        Used in higher order numerical schemes.

        Inputs
        ------------
        t (float): time
        x (np.ndarray): (N, dimX) array of current states
    
        Returns
        ------------
        dsigma (np.ndarray): (N, dimX, dimW, dimX) array of dsigma values
        """
        raise NotImplementedError(self._error_msg('dsigma'))

class HypoellipticSDE(MvSDE):

    """
    Base class for Hypoelliptic SDEs.
    
    In the current package implementation, all Hypoelliptic SDEs are p-1-times integrated SDEs.
    To build a hypoelliptic SDE, we define the following methods:
    
    'b_rough'
    'sigma_rough'
    """
    numerical_scheme_cls = HypoellipticEulerMaruyama
    
    def __init__(self, **kwargs):
        MvSDE.__init__(self, **kwargs)
        self._check_dims()
    
    def _check_dims(self):
        ns_1 = self.n_smooth + 1; dx = self.dimX
        err_str = f'Dimenion of Hypoelliptic Linear SDEs must be divisible by n_smooth + 1: dimX = {dx}, n_smooth+1= {ns_1}'
        if self.dimX % (self.n_smooth + 1) != 0:        
            raise ValueError(err_str)
    
    @property
    def dimS(self):
        return self.dimX - self.dimW
    
    def b(self, t, x):
        return np.concatenate([x[:, -self.dimS:], self.b_rough(t, x)], axis=1)
    
    def sigma(self, t, x):
        return np.concatenate([np.zeros((self.N, self.dimS, self.dimX)), self.sigma_rough(t, x)], axis=1)

    def db(self, t, x): # (N, dimX, dimX)
        raise NotImplementedError(self._error_msg('db'))
    
    def dsigma(self, t, x): # (N, dimX, dimW, dimX)
        raise NotImplementedError(self._error_msg('dsigma'))
        
    def transform_X_to_W(self, X: np.ndarray, t_start: float, x_start) -> np.ndarray:
        raise ValueError('Not possible to transform from diffusion paths to the driving noise, for hypoelliptic SDEs.')

class IntegratedSDE(HypoellipticSDE):
    """
    To define a non-linear Integrated (Hypoelliptic) SDE:
    
    - Subclass integrated SDE, and define the following:

    # Must be defined:    
    `dimX` as a class attribute: must be a multiple of 2.
    _diag_cov: Indicator for whether the diffusion covariance matrix of the rough component is diagonal.
    `b_rough` and `sigma_rough` methods: the drift and diffusion coefficients for the rough component.
        Outputs must have dimension (N, dimW) and (N, dimW, dimW) respectively.

    #Optionally:
    `default_params`: dictionary of default parameters for the SDE.
    
    """
    n_smooth = 1

    def b_rough(self, t, x):
        """
        Drift coefficient restricted to the rough component:
        may also depend on the smooth component.
        
        Used to parameterise integrated Linear SDEs within particle algorithms.
        Output should be of the form (N, dimW)
        """
        raise NotImplementedError(self._error_msg('b_rough'))

    def db_rough(self, t, x):
        """
        First derivative of the drift, restricted to the rough component
        (i.e we take each rough component and evaluate its derivative w.r.t its rough component)

        Output should be of the form (N, dimW, dimW)
        """
        raise NotImplementedError(self._error_msg('db_rough'))
    
    def sigma_rough(self, t, x):
        """
        Diffusion coefficient restricted to the rough component:
        Output should be of the form (N, dimW, dimW)
        """
        raise NotImplementedError(self._error_msg('sigma_rough'))

class TwiceIntegratedSDE(HypoellipticSDE):
    """
    To define a non-linear Twice Integrated (Hypoelliptic) SDE:
    
    - Subclass integrated SDE, and define the following:

    # Must be defined:    
    `dimX` as a class attribute: must be a multiple of 3.
    _diag_cov: Indicator for whether the diffusion covariance matrix of the rough component is diagonal.
    `b_rough` and `sigma_rough` methods: the drift and diffusion coefficients for the rough component.
        Outputs must have dimension (N, dimW) and (N, dimW, dimW) respectively.

    #Optionally:
    `default_params`: dictionary of default parameters for the SDE.
    
    """
    n_smooth = 2
    
    def b_rough(self, t, x):
        """
        Drift coefficient restricted to the rough component:
        may also depend on the smooth component.

        Inputs
        ------------
        t (float): time
        x (np.ndarray): (N, dimX) array of current states
        
        Returns
        ------------
        b_rough (np.ndarray): (N, dimW) array of rough drift coefficient values
        """
        raise NotImplementedError(self._error_msg('b_rough'))

    def db_rough(self, t, x):
        """
        Drift coefficient restricted to the rough component:
        may also depend on the smooth component.

        Inputs
        ------------
        t (float): time
        x (np.ndarray): (N, dimX) array of current states
        
        Returns
        ------------
        db_rough (np.ndarray): (N, dimW) array of rough drift coefficient values
        """
        raise NotImplementedError(self._error_msg('db_rough'))
    
    def sigma_rough(self, t, x):
        """
        Diffusion coefficient restricted to the rough component:
        Output should be of the form (N, dimW, dimW)
        """
        raise NotImplementedError(self._error_msg('sigma_rough'))

#--------------------------------------------- Examples of Multivariate SDEs -------------------------------------------

class FitzHughNagumo(MvEllipticSDE):
    """
    Stochastic FitzhughNagumo model with additive noise in both components: 
    
    Note: An alternative variant of this model is available through transformation, that is an integrated diffusion.
    """
    dimX = 2
    default_params = {'rho': np.array([1.4, 1.5, 10.]),
                      'phi': np.array([0.25, 0.2])
                        }
    _diag_cov = True

    def b(self, t, x):
        x_1 = self.rho[0] * ((-x[:, 0] ** 3) + x[:, 0] - x[:, 1] + 0.5) # (N, )
        x_2 = self.rho[1] * x[:, 0] - x[:, 1] + self.rho[2] # (N, )
        return np.stack([x_1, x_2], axis=1)

    def sigma(self, t, x):
        N = x.shape[0]
        return np.stack([np.diag(self.phi)]*N, axis=0)
    
    def db(self, t, x):
        N = x.shape[0]
        db_1_dx_1 = self.rho[0] * (3 * (x[:, 0] ** 2) + 1) # (N, )
        db_1_dx_2 = np.array([-self.rho[1]]*N)
        db_2_dx_1 = np.array([self.rho[2]]*N)
        db_2_dx_2 = np.array([-1.]*N)
        db = np.stack([db_1_dx_1, db_1_dx_2, db_2_dx_1, db_2_dx_2], axis=1).reshape(N, 2, 2)
        return db

    # def db_diag(self, t, x):
    #     N = x.shape[0]
    #     db_1_dx_1 = self.rho[0] * (3 * (x[:, 0] ** 2) + 1) # (N, )
    #     db_2_dx_2 = np.array([-1.]*N) #(N, )
    #     return np.stack([db_1_dx_1, db_2_dx_2], axis=1) # (N, 2)
        
    def dsigma(self, t, x):
        N = x.shape[0]
        return np.zeros((N, self.dimX, self.dimX, self.dimX))

class LotkaVolterra(MvEllipticSDE):
    """
    The stochastic Lotka-Volterra model for predator-prey dynamics.
    
    dX_{1, t} = \alpha X_{1, t} - \beta X_{1, t}X_{2, t}dt + s_1 dW_{1, t}
    dX_{2, t} = \delta X_{1, t}X_{2, t} -\gamma γX_{2, t}dt + s_2 dW_{2, t}

    X_1: Prey population,
    X_2: Predator population
    
    \alpha : prey growth rate,
    \beta: predation rate,
    \delta: predator reproduction rate per prey consumed,
    \gamma: predator mortality rate,
    s_1, s_2 :noise intensities for prey and predator.
    
    Note: Good initial values for the process are:
    X_1 = 50, X_2 = 10
    """
    dimX = 2
    default_params = {'alpha': 0.1,
                      'beta': 0.02,
                      'delta': 0.01,
                      'gamma': 0.5,
                      's': np.array([0.1, 0.1])}
    _diag_cov = True
    
    def b(self, t, x):
        x_1 = self.alpha * x[:, 0] - self.beta * x[:, 0] * x[:, 1] # (N, )
        x_2 = self.delta * x[:, 0]*x[:, 1] - self.gamma * x[:, 1] # (N, )
        return np.stack([x_1, x_2], axis=1)

    def sigma(self, t, x):
        N = x.shape[0]
        diag_sigma = x*self.s # (N, 2)
        return np.stack([np.diag(diag_sigma[i]) for i in range(N)], axis=0)
    

#---------------------------------- Abstract Base Classes for Multivariate Elliptic Linear SDEs ----------------------------------

class MvLinearSDE(MvSDE):
    """
    Multivariate linear SDEs. These objects are used in forward and backward proposals within DA particle filters.
    
    When initialised, the __init__method should do the following things:
    - Check that the parameters of the SDE, given the above are well-defined.
    
    Also, make a note of how the input parameters are used build the linear SDE.
    
    Implements the following methods:
        
    'grad_log_px' - Used in simulation/transformations of diffusion bridges in backward proposals.
    'grad_log_py' - Used in simulation/transformations of forward proposals.
    'exact_simulate' - Not used currently in SMC algorithms.
    'optimal_proposal_dist' - Used to construct end point proposals in backward proposals.
    'transition_dist' - Used in weights for VanderMeulenSchauer proposals.
    
    When initialised, one needs to provide:
    
    N - The number of particles
    dimX - The dimension of the SDE
    
    One also needs to provide parameters for each of the N proposal SDEs. Each N corresponds to a different 
    proposal bridge. 
    
    To do:
    ------------
    
    - Some of the MvLinear SDEs that you have defined have diagonal drift and covariance matrices for simplicity.
        In these cases, change the form of the output of A, B, C and _a, _b, _v so that they have different dimensions.
        - Then, change the methods b, and sigma and Cov so that they are computed faster for these special cases.
        - Then, write a new version of 'optimal_proposal_dist' that impelments the special case faster.
        - Then, change the code of 'mv_grad_log_linear_gaussian' to take advantage of the special case.
            Give it a kwarg diag=True for this case to speed up computations. 
        - When you do this, you may want to think about a version of the code that only takes the diagonal of the covariance matrix. 
    - Start writing the code for the evaluation of the weights.
    """
    exact_scheme = MvLinearExact
    
    def __init__(self, N=1, dimX=2, **kwargs):
        self.dimX = dimX
        self.N = N
        super().__init__(**kwargs)
        self.check_input_params()

    def A(self, t):
        """
        Inputs:
        -------
        t: float: The time step
        
        Returns
        -------
        A: ndarray: A matrix of dimension (self.N, dimX)
        """
        raise NotImplementedError(self._error_msg('A'))
    
    def B(self, t):
        """
        Inputs:
        -------
        t: float: The time step
        
        Returns
        -------
        B: ndarray: A tensor of dimension (self.N, dimX, dimX)
        """
        NotImplementedError(self._error_msg('B'))
    
    def C(self, t):
        """
        Inputs:
        -------
        t: float: The time step
        
        Returns
        -------
        C: ndarray: A matrix of dimension (self.N, dimX, dimX)
        """
        raise NotImplementedError(self._error_msg('C'))
            
    def b(self, t, x):
        """
        Input (N, dimX)
        Output (N, dimX)
        """
        self._check_input_x(x)
        M = x.shape[0]; N = self.N
        A = np.concatenate([self.A(t)]*M, axis=0) if N == 1 else self.A(t)
        B = np.concatenate([self.B(t)]*M, axis=0) if N == 1 else self.B(t)
        return A + np.einsum('ijk,ik->ij', B, x)
    
    def sigma(self, t, x):
        """
        Input (N, dimX)
        Output (N, dimX, dimW)
        """
        self._check_input_x(x)
        M = x.shape[0]; N = self.N
        C = np.concatenate([self.C(t)]*M, axis=0) if N == 1 else self.C(t)
        return C

    # def b_vec(self, t, x):
    #     return self.A_vec(t) + self.B_vec(t) * x
    
    # def sigma_vec(self, t, x):
    #     return self.C_vec(t)

    # def Cov_vec(self, t, x):
    #     return self.sigma_vec(t, x) ** 2

    def db(self, t, x):
        """
        db/dx is a 2D Jacobian, which we evaluate for each of the N samples.
        
        Inputs: t: the time step, x: (N, dimX)
        Returns: (N, dimX, dimX)
        """
        self._check_input_x(x); M = x.shape[0]
        B = np.concatenate([self.B(t)]*M, axis=0) if self.N == 1 and M > 1 else self.B(t)
        return B
            
    def dsigma(self, t, x):
        """
        dsigma/dx is a 3D tensor, which we evaluate for each of the N samples.
        """
        self._check_input_x(x); M = x.shape[0]
        return np.zeros((M, self.dimX, self.dimX, self.dimX))

    def grad_log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        Gradient of the log transition density
        """
        # When simulating:
        
        # s: float
        # t: float
        # x_s: (N, dimX)
        # x_t: (N, dimX)
        return mv_grad_log_linear_gaussian(x_s, x_t, self._a(s, t), self._b(s, t), self._v(s, t))

    def grad_grad_log_px(self, s: float, t: float):
        return mv_grad_grad_log_linear_gaussian(self._a(s, t), self._b(s, t), self._v(s, t))
                    
    def grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: np.ndarray, sigmaY: np.ndarray):
        """
        Gradient of the log transition density of Y_t | X(s), where Y_t | E_t=e_t \sim N(Le_t, \sigma_Y \sigma_Y^T)
        
        s: float
        t: float
        x_s: (N, dimX)
        y_t: (N, dimY)
        LY: (dimY, dimX)
        sigmaY: (dimY, dimY)
        """
        _a = np.einsum('ij,hjk->hik', LY, self._a(s, t)) #(N, dimY, dimX)
        _b = (LY @ self._b(s, t).T).T #(N, dimY)
        V_L_T = np.einsum('ijk,kl->ijl', self._v(s, t), LY.T) #(N, dimX, dimY)
        L_V_L_T = np.einsum('ij,hjk->hik', LY, V_L_T) #(N, dimY, dimY)
        V = L_V_L_T + sigmaY @ sigmaY.T #(N, dimY, dimY)
        return mv_grad_log_linear_gaussian(x_s, y_t, _a, _b, V)

    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix
        """
        return NotImplementedError(self._error_msg('_a'))
    
    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return NotImplementedError(self._error_msg('_b'))
    
    def _v(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix
        """
        return NotImplementedError(self._error_msg('_v'))

    def transition_dist(self, s: float, t: float, x_s: np.ndarray):
        """
        Transition density of the linear SDE, output as a distribution object:
        Object is vectorised over N particles.
        
        Use in DiscreteDiscreteSSMs for exact bootstrap fk or for the weights in guided fk.
        
        Inputs
        ------------
        s: float
        t: float
        x_s: (N, dimX) array
        
        Returns
        ------------
        VaryingCovNormal (particles.distributions.dists): a distribution object.
        """
        if self.N > 1:
            a = self._a(s, t); b = self._b(s, t); v = self._v(s,t) # (N, dimX, dimX)
            return VaryingCovNormal(loc=np.einsum('ijk,ik->ij', a, x_s) + b, cov=v)
        else:
            return MvNormal(loc=(x_s @ self._a(s, t)[0].T) + self._b(s, t)[0], cov=self._v(s, t)[0])
    
    def optimal_proposal_dist(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: np.ndarray, sigmaY: np.ndarray):
        """
        Proposal for the end point using the exact distribution $E_t | E_{t-1}=e_{t-1}, Y_t = y_t$ from a Linear SDE.
        No longer used in the CDSSM SMC algorithms, but used in DiscreteDiscreteSSMs for guided proposals.
                
        x_s: (N, dimX)
        y_t: (1, dimY)
        LY: (dimY, dimX)
        sigmaY: (dimY, dimY)
        """    
        A = self._a(s, t); b = self._b(s, t); S = self._v(s,t); CovY = sigmaY @ sigmaY.T    
        N = self.N; dimX = self.dimX
        if x_s.shape != (N, dimX) and N != 1:
            raise ValueError('Input x_s must have shape (N, dimX)')
        if y_t.shape[0] != 1:
            raise ValueError('Input y_t must have first dimension 1')
        if N == 1:
            M = x_s.shape[0]
            A = np.concatenate([A]*M); b = np.concatenate([b]*M); S = np.concatenate([S]*M)
        mu_x = np.einsum('ijk,ik->ij', A, x_s) + b # (N, dimX, dimX), (N, dimX) -> (N, dimX)
        pred = MeanAndCov(mu_x, S)
        opt_prop_loc, opt_prop_cov = mv_filter_step_var_cov(LY, CovY, pred, y_t)
        opt_prop_dist = VaryingCovNormal(loc=opt_prop_loc, cov=opt_prop_cov) if N > 1 else MvNormal(loc=opt_prop_loc, cov=opt_prop_cov[0])
        return opt_prop_dist

    def check_input_params(self):
        """
        Utility method to check that the input parameters are well-defined 
        for the given SDE, assuming a given number of particles and SDE dimension.
        """
        for name, shapes in self.param_shapes.items():
            N = self.N
            param_shape = self.__dict__[name].shape
            if N > 1 and param_shape != shapes[0]:
                raise ValueError(f"Parameter {name} must have shape {shapes[0]} for N>1. Input shape: {param_shape}")
            if N == 1 and len(shapes) == 1 and param_shape != shapes[0]:
                raise ValueError(f"If N=1, then parameter {name} must be of shape {shapes[0]}. Input shape: {param_shape}")
            if N == 1 and len(shapes) > 1 and param_shape != shapes[1]:
                raise ValueError(f"If N=1, then parameter {name} must be of shape {shapes[1]}. Input shape: {param_shape}")
                
    def _check_input_x(self, x):
        if x.shape[0] != self.N and self.N > 1:
            raise ValueError('Input x must match number of particles in the first dimension.')
        if x.shape[1] != self.dimX:
            raise ValueError('Input x must have the same dimension as the SDE in the second dimension.')


#---------------------------------- Example Classes for Multivariate Elliptic Linear SDEs ----------------------------------

class MvIndepBrownianMotion(MvLinearSDE, MvEllipticSDE):
    """
    Multivariate Scaled Brownian Motion, given by:
    dX_t  = m dt + s dW_t
    
    With m = (m_1, ..., m_d)^T, s = diag(s_1, ..., s_d)
    Independence between the components means that we don't need matrix exponentials.
    
    Input parameters:
    'm': (N, dimX) drift vector
    's': (N, dimX) diagonal diffusion matrix vector
    """
    
    _diag_cov = True

    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        return {'m': np.zeros((N, dx)),
                's': np.ones((N, dx))
                }

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'m': [(N, dx)], 's': [(N, dx)]}

    def A(self, t):
        return self.m

    def B(self, t):
        return np.zeros((self.N, self.dimX, self.dimX))
    
    def C(self, t):
        return np.stack([np.diag(self.s[i, :]) for i in range(self.N)])
        
    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        return np.stack([np.eye(self.dimX)]*self.N)

    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return (t-s) * self.m
        
    def _v(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix
        """
        cov = np.stack([np.diag(self.s[i, :]) for i in range(self.N)])
        return (t-s) * cov
    
class MvBrownianMotion(MvIndepBrownianMotion):
    """
    Multivariate Brownian Motion, given by:
    dX_t  = m dt + s dW_t
    
    With m = (m_1, ..., m_d)^T, s is a general dxd matrix.
    
    Input parameters:

    'm': (N, dimX) drift vector
    's': (N, dimX, dimX) if N>1 else (dimX, dimX) diffusion matrix
    """    
    @property
    def _diag_cov(self):
        N, dx = self.N, self.dimX
        dummy_x = np.zeros((N, dx))
        return np.all([np.all(np.isclose(self.Cov(0., dummy_x)[i]-np.diag(np.diag(self.Cov(0., dummy_x)[i])), np.zeroslike(self.Cov(0., dummy_x)))) for i in range(self.N)])

    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'m': np.zeros((N, dx))}
        default_params['s'] = np.eye(dx) if N == 1 else np.stack([np.eye(dx)]*N)
        return default_params

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'m': [(N, dx)], 's': [(N, dx, dx), (dx, dx)]}
        
    def C(self, t):
        C = self.s.reshape((self.N, self.dimX, self.dimX)) if self.N == 1 else self.s
        return C
        
    def _v(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        dummy_x = np.zeros((self.N, self.dimX))
        cov = self.Cov(t, dummy_x)        
        return (t-s) * cov

    def check_input_params(self):
        if self.m.shape != (self.N, self.dimX):
            raise ValueError('Drift vector m must be of shape (N, dimX).')            
        if self.s.shape not in [(self.N, self.dimX, self.dimX), (self.dimX, self.dimX)]:
            raise ValueError("Diffusion matrix diagonal s must be of shape (N, dimX, dimX) for N>1 or (dimX, dimX) for N=1")
    
class MvIndepOrnsteinUhlenbeck(MvLinearSDE, MvEllipticSDE):
    """
    Multivariate Ornstein-Uhlenbeck process, given by:
    dX_t  = \rho(\mu - X_t) dt + \phi dW_t
    
    With \rho = diag(\rho_1, ..., \rho_d), C = \diag(\phi_1, \dots, \phi_d)
    and \mu = (\mu_1, \dots, \mu_d)^T.
    Independence between the components means that we don't need matrix exponentials.
    
    Input parameters: 
    'rho': (N, dimX) vector of reversion rates
    'mu': (N, dimX) vector of means
    'phi': (N, dimX) vector of diffusion diagonals
    """

    _diag_cov = True

    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'rho': 0.5*np.ones((N, dx)),
                        'mu': np.zeros((N, dx)),
                        'phi': np.ones((N, dx))
                        }
        return default_params

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'rho': [(N, dx)], 'mu': [(N, dx)], 'phi': [(N, dx)]}
    
    def A(self, t):
        return self.mu * self.rho
    
    def B(self, t):
        return np.stack([np.diag(-1.*self.rho[i, :]) for i in range(self.N)])
    
    def C(self, t):
        return np.stack(np.diag(self.phi[i, :]) for i in range(self.N))

    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        return np.stack([np.diag(np.exp(-self.rho[i, :]*(t-s))) for i in range(self.N)])

    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return (1. - np.exp(-self.rho*(t-s))) * self.mu
        
    def _v(self, s, t):
        """
        Should be an (N, dimX, dimX) matrix.
        """
        def univ_v(s, t, rho, phi):
            return (phi ** 2)/(2*rho) * (1 - np.exp(-2*rho*(t-s)))
        return np.stack([np.diag(univ_v(s, t, self.rho[i, :], self.phi[i, :]) )for i in range(self.N)])

class MvOrnsteinUhlenbeck(MvIndepOrnsteinUhlenbeck):
    """
    Multivariate Ornstein-Uhlenbeck process, given by:
    dX_t  = \rho(\mu - X_t) dt + \phi dW_t
    
    With \rho a dxd diagonal matrix
    \mu a 1xd vector
    and \phi a dxd matrix
    
    Diagonal \rho means matrix exponential is not required.
    
    Input parameters: 
    'rho': (N, dimX) matrix of reversion rates
    'mu': (N, dimX) vector of means
    'phi': (N, dimX, dimX)/(dimX, dimX) diffusion matrix
    """

    @property
    def _diag_cov(self):
        N, dx = self.N, self.dimX
        dummy_x = np.zeros((N, dx))
        return np.all([np.all(np.isclose(self.Cov(0., dummy_x)[i]-np.diag(np.diag(self.Cov(0., dummy_x)[i])), np.zeroslike(self.Cov(0., dummy_x)))) for i in range(self.N)])
    
    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'rho': 0.5*np.ones((N, dx)),
                        'mu': np.zeros((N, dx)),
                        'phi': np.eye(dx) if N == 1 else np.stack([np.eye(dx)]*N)
                        }
        return default_params

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'rho': [(N, dx)], 'mu': [(N, dx)], 'phi': [(N, dx, dx), (dx, dx)]}

    def C(self, t):
        C = self.phi.reshape(self.N, self.dimX, self.dimX) if self.N == 1 else self.phi
        return C
    
    def _v(self, s, t):
        diff_cov = np.einsum('ijk,ilk->ijl', self.phi, self.phi) if self.N > 1 else (self.phi @ self.phi.T).reshape(1, self.dimX, self.dimX) # (N, dimX, dimX)
        A_plus_A_T = lambda rho: rho.reshape(1, self.dimX) + rho.reshape(self.dimX, 1) #\rho_ii + \rho_kk
        rho_sums = np.stack([A_plus_A_T((self.rho[i, :])) for i in range(self.N)]) # (N, dimX, dimX)
        cov = diff_cov/rho_sums * (1 - np.exp(-2*rho_sums*(t-s))) # (N, dimX, dimX)
        return cov

class MvFullOrnsteinUhlenbeck(MvOrnsteinUhlenbeck):
    """
    Multivariate Ornstein-Uhlenbeck process, given by:
    dX_t  = \rho(\mu - X_t) dt + \phi dW_t
    
    With \rho a dxd matrix
    \mu a 1xd vector
    and \phi a dxd matrix
    
    Input parameters: 
    'rho': (N, dimX, dimX)/(dimX, dimX) matrix of reversion rates
    'mu': (N, dimX) vector of means
    'phi': (N, dimX, dimX)/(dimX, dimX) diffusion matrix
    
    WARNING: Evaluation of the matrix exponential is required.
    
    To do: Implement the _v(s,t) method for this class. 
    Requires a general expression for the covariance matrix of the random vector
    \int_{s}^t {exp[-\rho(t-u)]\phi} dW_u
    
    Where \rho is a dxd matrix so we have a matrix exponential, \phi is dxd matrix and W_t is a dx1 Brownian motion.
    """ 

    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'rho': 0.5*np.eye(N) if N == 1 else np.stack([0.5*np.eye(dx)]*N),
                    'mu': np.zeros((N, dx)),
                    'phi': np.eye(N) if N == 1 else np.stack([np.eye(dx)]*N)
                }
        return default_params
    
    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'rho': [(N, dx, dx), (dx, dx)], 'mu': [(N, dx)], 'phi': [(N, dx, dx), (dx, dx)]}

    def A(self, t):
        return np.einsum('ijk,ik->ij', self.rho, self.mu)
    
    def B(self, t):
        return -self.rho

    def C(self, t):
        C = self.s.reshape((self.N, self.dimX, self.dimX)) if self.N == 1 else self.s
        return C

    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        return sla.expm(-self.rho*(t-s))
    
    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return np.einsum('ijk,ik->ij', np.stack([np.eye(self.dimX)]*self.N) - sla.expm(-self.rho*(t-s)), self.mu)
    
    def _v(self, s, t):
        """
        Should be an (N, dimX, dimX) matrix.
        """
        return NotImplementedError(self._error_msg('_v'))

#---------------------------------- Abstract Base Classes for Multivariate Hypoelliptic Linear SDEs ----------------------------------

class HypoellipticLinearSDE(MvLinearSDE, HypoellipticSDE):

    numerical_scheme_cls = HypoellipticEulerMaruyama
    
    def __init__(self, **kwargs):
        MvLinearSDE.__init__(self, **kwargs)
        self._check_dims()

    def A_rough(self, t): # (N, dimW)
        raise NotImplementedError(self._error_msg('A_rough'))
    
    def B_rough(self, t): # (N, dimW, dimW)
        raise NotImplementedError(self._error_msg('B_rough'))
    
    def C_rough(self, t): # (N, dimW, dimW)
        raise NotImplementedError(self._error_msg('C_rough'))

    def b_rough(self, t, x):
        """
        Input (N, dimX)
        Output (N, dimX)
        """
        self._check_input_x(x)
        M = x.shape[0]; N = self.N
        A = np.concatenate([self.A_rough(t)]*M, axis=0) if N == 1 else self.A_rough(t)
        B = np.concatenate([self.B_rough(t)]*M, axis=0) if N == 1 else self.B_rough(t)
        return A + np.einsum('ijk,ik->ij', B, x[:, -self.dimW:])
    
    def sigma_rough(self, t, x):
        """
        Input (N, dimX)
        Output (N, dimX, dimW)
        """
        self._check_input_x(x)
        M = x.shape[0]; N = self.N
        C = np.concatenate([self.C_rough(t)]*M, axis=0) if N == 1 else self.C_rough(t)
        return C
        
    def db_rough(self, t, x): # (N, dimW, dimW)
        self._check_input_x(x)
        M = x.shape[0]; N = self.N
        db_rough = np.concatenate([self.B_rough(t)]*M, axis=0) if N == 1 else self.B_rough(t)
        return db_rough

    def A(self, t):
        N, ds = self.N, self.dimS
        return np.concatenate([np.zeros((N, ds)), self.A_rough(t)], axis=1)

    def B(self, t):
        N, ds, dw = self.N, self.dimS, self.dimW
        B_rows_sm = [None] * self.n_smooth
        for i in range(1, self.n_smooth + 1):
            B_row_sm = np.concatenate([np.zeros((dw, dw*i)), np.eye(dw), np.zeros((dw, dw*(self.n_smooth-i)))], axis=1) # (dimW, dimX)
            B_rows_sm[i-1] = np.stack([B_row_sm]*N) # (N, dimW, dimX)
        B_rou = np.concatenate([np.zeros((N, dw, ds)), self.B_rough(t)], axis=2) # (N, dimW, dimX)
        B = np.concatenate(B_rows_sm + [B_rou], axis=1) # (N, dimX, dimX)
        return B

    def C(self, t):
        N, ds, dw = self.N, self.dimS, self.dimW
        C = self.C_rough(t).reshape((N, dw, dw)) if self.N == 1 else self.C_rough(t)
        C = np.concatenate([np.zeros((N, ds, dw)), C], axis=1)
        return C

    def _a(self, s, t): # returns (N, dimX, dimX)
        if hasattr(self, '_a_cached') and self._a_delta_t == t-s:
            return self._a_cached
        delta_t = t - s; ns_1 = self.n_smooth + 1; N = self.N
        a_coefs = self.gen_a_coefs(delta_t) # (ns_1, ns_1, N, dw)
        _a_block = lambda i, j, n: np.diag(a_coefs[i, j, n, :]) # each block is (dw,, dw)
        _a_blocks = [[np.stack([_a_block(i, j, n) for n in range(N)]) for j in range(ns_1)] for i in range(ns_1)] # list of list (ns_1, ns_1) containing (N, dw, dw) arrays
        _a = np.concatenate([np.concatenate(a_row, axis=2) for a_row in _a_blocks], axis=1) # (N, dimX, dimX)
        self._a_cached = _a; self._a_delta_t = delta_t        
        return _a

    def _b(self, s, t): # (N, dimX)
        if hasattr(self, '_b_cached') and self._b_delta_t == t-s:
            return self._b_cached
        delta_t = t - s
        b_coefs = self.gen_b_coefs(delta_t) # (n_smooth + 1, N, dimW)
        _b = np.concatenate([b_coefs[i] * self.A_rough(t) for i in range(self.n_smooth + 1)], axis=1) # (N, dimX)
        self._b_cached = _b; self._b_delta_t = delta_t
        return _b

    def _v(self, s, t): # (N, dimX, dimX)
        if hasattr(self, '_v_cached') and self._v_delta_t == t-s:
            return self._v_cached
        delta_t = t - s; ns_1 = self.n_smooth + 1; N = self.N
        v_coefs = self.gen_v_coefs(delta_t) # (ns_1, ns_1, N, dimW, dimW)
        _v_block = lambda i, j, n: v_coefs[i, j, n] * self.Cov_rough[n]
        _v_blocks = [[np.stack([_v_block(i, j, n) for n in range(N)]) for j in range(ns_1)] for i in range(ns_1)] # (N, dimW, dimW)
        _v = np.concatenate([np.concatenate(v_row, axis=2) for v_row in _v_blocks], axis=1) # (N, dimX, dimX)
        self._v_cached = _v; self._v_delta_t = delta_t
        return _v
    
    def _ps(self, n, t):
        """
        Generates the function t^i/i! for i=0, 1, ..., n-1
        """
        return np.array([1.] + [t / i for i in range(1, n)]).cumprod()

    def transform_X_to_W(self, X, t_start, x_start): # Redefined so that it inherits from HypoellipticSDE.
        return HypoellipticSDE.transform_X_to_W(self, X, t_start, x_start)
    
class HypoellipticIndepBrownianMotion(HypoellipticLinearSDE):
    """
    Cannot be used without subclassing and defining the attribute 'n_smooth'
    """
    _diag_cov = True

    @property
    def default_params(self):
        N, dw = self.N, self.dimW
        return {'m': np.zeros((N, dw)),
                's': np.ones((N, dw))
                }

    @property
    def param_shapes(self):
        N, dw = self.N, self.dimW
        return {'m': [(N, dw)], 's': [(N, dw)]}

    def A_rough(self, t):
        return self.m

    def B_rough(self, t):
        return np.zeros((self.N, self.dimW, self.dimW))

    def C_rough(self, t): # (N, dw, dw)
        return np.stack([np.diag(self.s[i]) for i in range(self.N)]) 

    def gen_a_coefs(self, t):
        ns_1 = self.n_smooth + 1 # ns_1
        ps = self._ps(ns_1, t)
        a = lambda i, j: 0. if i > j else ps[j-i]
        a_coefs = np.array([[a(i, j) for j in range(ns_1)] for i in range(ns_1)]) # (ns_1, ns_1)
        a_coefs = np.stack([a_coefs]*self.N, axis=2) # (ns_1, ns_1, N)
        a_coefs = np.stack([a_coefs]*self.dimW, axis=3) # (ns_1, ns_1, N, dw)
        return a_coefs # (ns_1, ns_1, N, dw)

    def gen_b_coefs(self, t):
        ps = self._ps(self.n_smooth + 2, t) # (ns_1 + 1)
        b_coefs = ps[1:] # (ns_1)
        b_coefs = np.stack([b_coefs]*self.N, axis=1) # (ns_1, N)
        b_coefs = np.stack([b_coefs]*self.dimW, axis=2)[::-1, :, :] # (ns_1, N, dw)
        return b_coefs
        
    def gen_v_coefs(self, t):
        ns_1 = self.n_smooth + 1
        ps = self._ps(2*self.n_smooth + 2, t) # (2 n_s, )
        v = lambda i, j: math.comb(2*ns_1 - i - j, ns_1 - i) * ps[2*ns_1 - i - j + 1]
        v_coefs = np.array([[v(i, j) for j in range(1, ns_1+1)] for i in range(1, ns_1+1)])
        v_coefs = np.stack([v_coefs]*self.N, axis=2) # (n_smooth +1, n_smooth +1, N)
        v_coefs = np.stack([v_coefs]*self.dimW, axis=3) # (n_smooth +1, n_smooth +1, N, dimW)
        v_coefs = np.stack([v_coefs]*self.dimW, axis=4) # (n_smooth +1, n_smooth +1, N, dimW, dimW)
        return v_coefs

    @property
    def Cov_rough(self): # (N, dw, dw)
        return np.stack([np.diag(self.s[i]*self.s[i]) for i in range(self.N)])
    
class HypoellipticBrownianMotion(HypoellipticIndepBrownianMotion):
    """
    Cannot be used without subclassing and defining the attribute 'n_smooth'
    """
    @property
    def _diag_cov(self):
        return np.all([np.all(np.isclose(self.Cov_rough[i]-np.diag(np.diag(self.Cov_rough[i])), np.zeroslike(self.Cov_rough))) for i in range(self.N)])

    @property
    def default_params(self):
        N, dw = self.N, self.dimW
        return {'m': np.zeros((N, dw)),
                's': np.eye(dw) if N == 1 else np.stack([np.eye(dw)]*N)
                }

    @property
    def param_shapes(self):
        N, dw = self.N, self.dimW
        return {'m': [(N, dw)], 's': [(N, dw, dw), (dw, dw)]}
    
    def C_rough(self, t):
        N, dw = self.N, self.dimW
        return self.s.reshape((N, dw, dw)) if N == 1 else self.s

    @property
    def Cov_rough(self): # (N, dw, dw)
        N, dw = self.N, self.dimW
        s = self.s.reshape((N, dw, dw)) if N == 1 else self.s
        return np.einsum('ijk,ilk->ijl', s, s)

class HypoellipticIndepOrnsteinUhlenbeck(HypoellipticLinearSDE):
    """
    Cannot be used without subclassing and defining the attribute 'n_smooth'
    """    
    _diag_cov = True

    @property
    def default_params(self):
        N, dw = self.N, self.dimW
        default_params = {'rho': 0.5*np.ones((N, dw)),
                        'mu': np.zeros((N, dw)),
                        'phi': np.ones((N, dw))
                        }
        return default_params

    @property
    def param_shapes(self):
        N, dw = self.N, self.dimW
        return {'rho': [(N, dw)], 'mu': [(N, dw)], 'phi': [(N, dw)]}
                        
    def A_rough(self, t):
        return self.mu * self.rho

    def B_rough(self, t): # (N, dw, dw)
        return np.stack([np.diag(-1.*self.rho[i]) for i in range(self.N)])

    def C_rough(self, t):
        return np.stack([np.diag(self.phi[i]) for i in range(self.N)])

    def gen_a_coefs(self, t):
        ns_1 = self.n_smooth + 1
        ps = self._ps(self.n_smooth, t)
        qs = self._qs(self.n_smooth + 1, t) # (n_smooth + 1, N, dimW)
        a_coefs = np.zeros((self.n_smooth + 1, self.n_smooth + 1, self.N, self.dimW))
        for i in range(ns_1):
            for j in range(ns_1):
                if j >= i:
                    a_coefs[i, j] = ps[j-i] *np.ones((self.N, self.dimW)) if j != self.n_smooth else qs[j-i]
        return a_coefs
        
    def gen_b_coefs(self, t):
        qs = self._qs(self.n_smooth + 2, t) # (n_smooth + 2, N, dimW) 
        b_coefs = qs[1:] # (n_smooth + 1, N, dimW)
        return b_coefs[::-1, :, :]

    def gen_v_coefs(self, t):
        ns_1 = self.n_smooth + 1
        gs = self._gs(ns_1, t) # (n_smooth + 1, n_smooth + 1, N, dimW)
        v_coefs = np.zeros((ns_1, ns_1, self.N, self.dimW))
        for i in range(ns_1):
            for j in range(ns_1):
                v_coefs[i, j] = gs[self.n_smooth - i, self.n_smooth - j]
        v_coefs = np.stack([v_coefs]*self.dimW, axis=4) # (n_smooth + 1, n_smooth + 1, N, dimW, dimW)
        return v_coefs

    def _qs(self, n, t):
        """
        Generates the function q_i(t) for i=0, 1, 2, ..., n-1.
        
        Does this for each \rho in the OU process.
        q_0(t) = exp[-\rho t]. We then have recursively:
        q_i(t) = \int_0^t q_{i-1}(s) ds

        We obtain the following recursion from direct integration:
        q_i(t) = -1/\rho (q_{i-1}(t) + (-t)^{i-1})        
        """
        rho = self.rho # (N, dw)
        ps = self._ps(n-1, t) # (n-1, )
        qs = np.zeros((n, self.N, self.dimW)) 
        qs[0] = np.exp(-rho*t)
        for i in range(1, n):
            qs[i] = 1./rho * (ps[i-1] - qs[i-1]) 
        return qs
    
    def _gs(self, n, t):
        """
        Generates the function g_ij(t) for 0 <= i,j <= n-1.
        So, output is an nxnxdimW array.
                
        Where g_n,m(t) is defined as:
        g_n,m(t) = q_n(t)q_m(t)dt g_n,m(0) 0
        
        We have the following recursions from integration by parts:
        g_{n,m}(t) = q_{n}(t)q_{m+1}(t) - g_{n-1,m+1}(t)
        g_{n, 0}(t) = 
        
        With initial condition obtained from direct integration:
        
        g_{0,0}(t) = 1/(2 \rho) (1 - exp[-2\rho t])
        """
        qs = self._qs(2*n-1, t) # (p, N, dimW)
        rho = self.rho
        gs = np.zeros((2*n-1, 2*n-1, self.N, self.dimW))
        gs[0, 0] = 1/(2.*rho) * (1 - np.exp(-2.*rho*t))
        for i in range(1, 2*n-1):
            gs[0, i] = 1./rho * (gs[0, i-1] - qs[0] * qs[i])
            for j in range(i):
                gs[j+1,  i-j-1] = qs[j+1] * qs[i-j] - gs[j, i-j]
        return gs[:n, :n, :, :] # (n, n, N, dimW)

    @property
    def Cov_rough(self): # (N, dw, dw)
        return np.stack([np.diag(self.phi[i]*self.phi[i]) for i in range(self.N)])
    
class HypoellipticOrnsteinUhlenbeck(HypoellipticIndepOrnsteinUhlenbeck):
    """
    Cannot be used without subclassing and defining the attribute 'n_smooth'
    """
    @property
    def default_params(self):
        N, dw = self.N, self.dimW
        default_params = {'rho': 0.5*np.ones((N, dw)),
                        'mu': np.zeros((N, dw)),
                        'phi': np.eye(dw) if N == 1 else np.stack([np.eye(dw)]*N)
                        }
        return default_params

    @property
    def _diag_cov(self):
        return np.all([np.all(np.isclose(self.Cov_rough[i]-np.diag(np.diag(self.Cov_rough[i])), np.zeroslike(self.Cov_rough))) for i in range(self.N)])
    
    @property
    def param_shapes(self):
        N, dw = self.N, self.dimW
        return {'rho': [(N, dw)], 'mu': [(N, dw)], 'phi': [(N, dw, dw), (dw, dw)]}

    def C_rough(self, t):
        N, dw = self.N, self.dimW
        return self.phi.reshape((N, dw, dw)) if N == 1 else self.phi

    @property
    def Cov_rough(self): # (N, dw, dw)
        N, dw = self.N, self.dimW
        phi = self.phi.reshape((N, dw, dw)) if N == 1 else self.phi
        return np.einsum('ijk,ilk->ijl', phi, phi)    

    def _gs(self, n, t):
        """
        Generates the function g_ij(t) for 0 <= i,j <= n-1.
        So, output is an nxnxdimW array.
                
        Where g_n,m(t) is defined as:
        g_n,m(t) = q_n(t)q_m(t)dt g_n,m(0) 0
        
        We have the following recursions from integration by parts:
        g_{n,m}(t) = g_{n-1,m+1}(t) - q_{n}(t)q_{m+1}(t)
        g_{n, 0}(t) = 
        
        With initial condition obtained from direct integration:
        
        g_{0,0}(t) = 1/(2 \rho) (1 - exp[-2\rho t])
        """
        qs = self._qs(2*n-1, t) # (p, N, dimW)
        rho = self.rho; dw = self.dimW; N = self.N
        A_dot_B_T = lambda A, B: A.reshape(dw, 1) * B.reshape(1, dw)
        rho_mat = np.stack([rho]*dw, axis=2) # (N, dw, dw)
        rho_sums = np.stack([self.rho[i, :].reshape(1, dw) + self.rho[i, :].reshape(dw, 1) for i in range(N)]) # (N, dw, dw)
        gs = np.zeros((2*n-1, 2*n-1, self.N, self.dimW, self.dimW))
        gs[0, 0] = 1/rho_sums * (1 - np.exp(-rho_sums*t))
        for i in range(1, 2*n-1):
            q_0_q_i = np.stack([A_dot_B_T(qs[0, k, :], qs[i, k, :]) for k in range(N)], axis=0) # (N, dw, dw)
            gs[0, i] = 1./rho_mat * (gs[0, i-1] - q_0_q_i)
            for j in range(i):
                q_prod = np.stack([A_dot_B_T(qs[j+1, k, :], qs[i-j, k, :]) for k in range(N)], axis=0) # (N, dw, dw)
                gs[j+1,  i-j-1] = q_prod - gs[j, i-j]
        return gs[:n, :n, :, :, :] # (n, n, N, dimW, dimW)

    def gen_v_coefs(self, t):
        ns_1 = self.n_smooth + 1
        gs = self._gs(ns_1, t) # (n_smooth + 1, n_smooth + 1, N, dimW, dimW)
        v_coefs = np.zeros((ns_1, ns_1, self.N, self.dimW, self.dimW))
        for i in range(ns_1):
            for j in range(ns_1):
                v_coefs[i, j] = gs[self.n_smooth - i, self.n_smooth - j]
        return v_coefs

#---------------------------------- Example Classes for Multivariate Hypoelliptic Linear SDEs ------------------------

class IntegratedIndepBrownianMotion(HypoellipticIndepBrownianMotion, IntegratedSDE):
    pass

class IntegratedBrownianMotion(HypoellipticBrownianMotion, IntegratedSDE):
    pass

class IntegratedIndepOrnsteinUhlenbeck(HypoellipticIndepOrnsteinUhlenbeck, IntegratedSDE):
    pass

class IntegratedOrnsteinUhlenbeck(HypoellipticOrnsteinUhlenbeck, IntegratedSDE):
    pass

class TwiceIntegratedIndepBrownianMotion(HypoellipticIndepBrownianMotion, TwiceIntegratedSDE):
    pass

class TwiceIntegratedBrownianMotion(HypoellipticBrownianMotion, TwiceIntegratedSDE):
    pass

class TwiceIntegratedIndepOrnsteinUhlenbeck(HypoellipticIndepOrnsteinUhlenbeck, TwiceIntegratedSDE):
    pass

class TwiceIntegratedOrnsteinUhlenbeck(HypoellipticOrnsteinUhlenbeck, TwiceIntegratedSDE):
    pass

#---------------------------------- Abstract Base class for time-switching SDEs ----------------------------------

class TimeSwitchingSDE(SDEBase):
    """
    General class for a diffusion process that switches between two possible regimes.
    
    To do: you could come back to this and change the API a bit, so that 
    the init method takes in the parameters of the two sdes and the switching time, 
    as opposed to the 2 SDEs themselves.
    
    Parameter to supply to the __init__ method are:
    - The parameters from sde1, appended with _1
    - The parameters from sde2, appended with _2
    - The switching time t_switch
    
    To constuct your own time-switching SDE, subclass and 
    define the two SDEs that you want to switch between as the class attributes
    
    - 'sde1_cls' 
    - 'sde2_cls'
    'simulate': None, 
    """
    t_switching_methods = {'simulate': None, 'b': 0, 'sigma': 0, 'db': 0, 'dsigma': 0, 'A': 0, 'B': 0, 'C': 0, '_a': 1, '_b': 1, '_v': 1}
    sde1_cls = None
    sde2_cls = None

    @property
    def params(self):
        sde1_params = {name + '_1': param for name, param in self.sde1.params.items()}
        sde2_params = {name + '_2': param for name, param in self.sde2.params.items()}
        params = {**sde1_params, **sde2_params, **{'t_switch': self.t_switch}}
        return params
    
    def __init__(self, **kwargs):
        sde1_kwargs = {k[:-2]: v for k, v in kwargs.items() if k.endswith('_1')}
        sde2_kwargs = {k[:-2]: v for k, v in kwargs.items() if k.endswith('_2')}
        if 'dimX' in kwargs.keys():
            sde1_kwargs.update({'dimX': kwargs['dimX']})
            sde2_kwargs.update({'dimX': kwargs['dimX']})
            del(kwargs['dimX'])
        self.sde1 = self.sde1_cls(**sde1_kwargs)
        self.sde2 = self.sde2_cls(**sde2_kwargs)
        self.N = 1
        SDE.__init__(self, **kwargs)
        self.base_sde_cls = self._gen_base_sde_cls()
        
        # Check that sde1 and sde2 have compatible dimensions
        assert self.sde1.dimX == self.sde2.dimX, 'Attribute dimX must be the same for both SDEs'
        assert self.sde1.dimW == self.sde2.dimW, 'Attribute dimW must be the same for both SDEs'

        sde1_methods = set(get_methods(self.sde1))
        sde2_methods = set(get_methods(self.sde2))
        methods = sde1_methods.intersection(sde2_methods)
        t_switching_methods = set(self.t_switching_methods.keys()).intersection(methods)
        base_methods = methods - t_switching_methods
        t_switching_methods = {method: self.t_switching_methods[method] for method in t_switching_methods}
        # Dynamically add methods t_switching methods from sde1 and sde2
        self._add_t_switching_methods(t_switching_methods)

        # Dynamically add methods from base_sde_cls
        self._add_base_methods(base_methods)

    @property
    def dimX(self):
        return self.sde1.dimX
    
    @property
    def dimW(self):
        return self.sde1.dimW
        
    @property
    def _diag_cov(self):
        return True if self.sde1._diag_cov and self.sde2._diag_cov else False
    
    @property
    def isLinear(self):
        return self.sde1.isLinear and self.sde2.isLinear

    @property
    def default_params(self):
        sde1_params = {name + '_1': param for name, param in self.sde1.default_params.items()}
        sde2_params = {name + '_2': param for name, param in self.sde2.default_params.items()}
        params = {**sde1_params, **sde2_params, **{'t_switch': 10}}
        return params

    def _add_base_methods(self, method_names):
        # Dynamically bind methods from base_sde_cls
        for method_name in method_names:
            if hasattr(self.base_sde_cls, method_name):
                method = getattr(self.base_sde_cls, method_name)
                # Bind the method to the instance
                setattr(self, method_name, method.__get__(self))

    def _gen_base_sde_cls(self):
        sde1_mro = self.sde1.__class__.mro()
        sde2_mro = self.sde2.__class__.mro()
        for cls in sde1_mro:
            if cls in sde2_mro:
                return cls
            
    def _add_t_switching_methods(self, method_dict):
        for method_name, t_idx in method_dict.items():
            m1 = getattr(self.sde1, method_name)
            m2 = getattr(self.sde2, method_name)
            method = self._t_switching_method(m1, m2, t_idx)
            setattr(self, method_name, method.__get__(self))

    def _t_switching_method(self, m1, m2, t_idx):
        def method(_self, *args, **kwargs):
            t = args[t_idx] if t_idx is not None else kwargs['t_end']
            cond = t < _self.t_switch if t_idx == 0 else t <= _self.t_switch
            return m1(*args, **kwargs) if cond else m2(*args, **kwargs)
            # else:
            #     pass   
            #     # Placeholder to deal with vectorised time in 1D case.             
            #     # cond = np.where(t - tol < _self.t_switch, 1., 0.)
            #     # return cond*m1(*args, **kwargs) 
        return method

#---------------------------------- Example classes for time-switching SDEs ----------------------------------
        
class TS_MvOrnsteinUhlenbeck(TimeSwitchingSDE, MvLinearSDE, MvEllipticSDE):
    sde1_cls = MvOrnsteinUhlenbeck
    sde2_cls = MvOrnsteinUhlenbeck

class TS_OrnsteinUhlenbeck(TimeSwitchingSDE, LinearSDE):
    sde1_cls = OrnsteinUhlenbeck
    sde2_cls = OrnsteinUhlenbeck