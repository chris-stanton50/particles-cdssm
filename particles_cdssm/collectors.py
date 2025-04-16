"""
Predictive Collectors:

In this module, we implement collectors (see particles.collectors) module that obtain at each time step of the particle filter, 
a prediction of the k-step ahead predictive distribution of either the latent state (X_{t+k} | Y_{1:t}) or of the observations 
(Y_{t+k} | Y_{1:t}). By using these collectors as arguments to the SMC or CDSSM_SMC class, we can use particle filters for online prediction,
as well as online filtering.

These collectors are comptible with both the SMC class and the CDSSM_SMC class, so can be used for prediction on both standard SSMs and CDSSMs.

Using a predictive Collector:

Say that we have defined a State Space Model (SSM) and a Feynman-Kac class representation of this ssm. Then 
we can use the SMC class to run a particle filter on this model, using code as follows:

```
import particles.state_space_models as ssms
from particles_cdssm.collectors import PredictiveParticles, ObservationPredictiveParticles
from particles import SMC

T = 100; N=100                          # Number of time steps and number of particles

ssm = ssms.StochVol()                   # Define the SSM
x, y = ssm.simulate(T)                  # Simulate data from the SSM
fk = ssms.Bootstrap(ssm=ssm, data=y)    # Define the Feynman-Kac class

collector = PredictiveParticles(K=1)    # Define the collector for K=1 step ahead prediction of X_{t+1} | Y_{1:t}
smc = SMC(fk=fk, N=N, collect=[col])    # Define the SMC object with the predictive collector as an argument

smc.run()                              # Run the SMC algorithm
```

When the SMC algorithm generated from this code is run, the collector will at eah time step t, after the filtering step, 
use simuation to generate a prediction of the k-step ahead predictive distribution of the latent state X_{t+k} | Y_{1:t} 

After the SMC algorithm has been run, the results of the collection can be accessed through the relevant attribute of the smc.summaries object:

```
predictive_particles = smc.summaries.predictive_particles
```

This will return a list of tuples, where each tuple contains the weights and the particles that have been generated to estimate the predictive distribution
of the latent state X_{t+k} | Y_{1:t}.

The same can be done for the observations Y_{t+k} | Y_{1:t} by using the ObservationPredictiveParticles collector instead of the PredictiveParticles collector.
In this case, after running the SMC algorithm with this collector, we can obtain the output of the collection instead accessing:

```
predictive_particles = smc.summaries.obs_predictive_particles
```

Other predictive collectors are also available, such as the PredictiveMoments/ObservationPredictiveMoments collectors, which collect the empirical moments of the predictive 
distribution estimates at each time step, or the NLPD collector, which collects the negative log posterior density of the observation predictive distribution at each time step.

One can also build your own predictive collector by inheriting from the PredictiveCollector class and implementing the out_func_X and/or out_func_Y methods, which are used to obtain
the output of the collection.

Implemented Predictive Collectors:

- PredictiveParticles: Collects the particles and weights for the predictive distribution of latent state X_{t+k} | Y_{1:t}
- ObservationPredictiveParticles: Collects the particles and weights for the predictive distribution of the observations Y_{t+k} | Y_{1:t}
- PredictiveMoments: Collects the empirical moments of the particles for the predictive distribution of the latent state X_{t+k} | Y_{1:t}
- ObservationPredictiveMoments: Collects the empirical moments of the particles for the predictive distribution of the observations Y_{t+k} | Y_{1:t}
- NLPD: Collects the negative log posterior density of the observation predictive distribution at each time step
- LowVarianceNLPD: Collects the negative log posterior density of the observation predictive distribution at each time step. Uses Gaussian KDE 
            with automatic bandwidth calibration to estimate the density of the particles.
- AbsoluteError: Collects the absolute error of the k-step ahead prediction against the true data at each time step
"""

import numpy as np
from scipy.stats import gaussian_kde

from particles import resampling as rs
from particles.collectors import Collector

#-----------------------------------------------Base Class for Predictive Collectors--------------------------------------------------------

class WeightedPredictiveMixin:
    """
    Mixin Class with methods used to obtain estimators of the predictive distribution 
    via adjusting the estisting weights from the particle filter, instead of
    using a simulation based approach.
    
    This has the following advantage over using simulation:
    
    - Lower compute cost, as no simulation is required.
    
    However, it has the following disadvanatages:
    
    - Not extensible to arbitrary k-step ahead predictive distribution (1 step only)
    - Estimator of predictive distribution may have low ESS under this approach (particularly for GuidedPF).
    
    Thus, the default approach is to use simulation, but this can be changed by setting the method to 'weighted' when 
    initialising the PredictiveCollector.
    """

    def PY_logpdf(self):
        """
        Returns the predictive densities of the observations $Y_t | X_t = x_t$
        for each of the x_t particles, evaluated at the observed data point.
        """
        if hasattr(self.fk, 'ssm'):   
            return self.fk.ssm.PY(self.t + self.k, self.Xp, self.X).logpdf(self.y_true)
        if hasattr(self.fk, 'cdssm'):
            return self.fk.cdssm.PY(self.t + self.k, self.Xp, self.X).logpdf(self.y_true)
        raise NotImplementedError("PY is not defined for this Feynman-Kac class.")

    @property
    def isBPF(self):
        return 'Bootstrap' in self.fk.__class__.__name__

    @property
    def has_y_true(self):
        return self.t + self.k < self.fk.T

    @property
    def y_true(self):
        return self.fk.data[self.t + self.k].ravel()
        
    def update_predictive_weights(self, smc):
        # Obtain predictive weights
        if (self.isBPF and smc.rs_flag) or (self.isBPF and smc.t == 0):
            # smc.pred_wgts = rs.Weights() # Empty rs.Weights object represents equal weights on particles
            self.rs_flag = True
            smc.pred_wgts = rs.Weights(np.log(np.array([1/smc.N]*smc.N)))
        else:
            self.rs_flag = False
            wgt_update = -1.*self.PY_logpdf()
            smc.pred_wgts = smc.wgts.add(wgt_update) # Creates a new Weights object
        self.wgts = smc.pred_wgts; self.X = smc.X
        
    def fetch_weighted(self, smc):
        self.X = smc.X; self.Xp = smc.Xp
        self.update_predictive_weights(smc)
        if not self.predict_observations:
            return self.out_func_X(self.wgts, self.X)
        else:
            self.predict()
            return self.out_func_Y(self.wgts, self.Y)

class SimulatedPredictiveMixin:
    """
    Mixin class with methods used to obtain Collectors that obtain 
    estimators of a predictive distribution via simulation, as opposed
    to adjusting the existing weights in the particle filter.
    
    Simulation has the following benefits over using adjusted weights:
       - Extensible to arbitrary k-step ahead predictive distribution
         for any k>1.
       - Robust performance even when using guided particle filters.
       
    However, these methods do have a higher computational cost. Simulation
    is the default approach, however this can be changed.
    """

    def PX_simulate(self):
        """
        Simulate from the transition density $X_t | X_{t-1} = x_{t-1}$.
        """
        if hasattr(self.fk, 'ssm'):
            return self.fk.ssm.PX(self.t + self.k, self.Xp).rvs(self.N)
        if hasattr(self.fk, 'cdssm'):
            t = self.t + self.k - 1 if self.fk.cdssm.isobservedat0 else self.t + self.k
            return self.fk.model_sde.simulate(self.N, self.Xp[self.Xp.dtype.names[-1]], t_start=self.fk.cdssm.S(t), t_end=self.fk.cdssm.S(t+1), num=self.fk.num)
        raise NotImplementedError("PX is not defined for this Feynman-Kac class.")
    
    def PY_simulate(self):
        """
        Simulate from the observation density $Y_t | X_t = x_t$.
        """
        if hasattr(self.fk, 'ssm'):
            return self.fk.ssm.PY(self.t + self.k, self.Xp, self.X).rvs(size=self.N)
        if hasattr(self.fk, 'cdssm'):
            return self.fk.cdssm.PY(self.t + self.k, self.Xp, self.X).rvs(size=self.N)
        raise NotImplementedError("PY is not defined for this Feynman-Kac class.")
    
    def time_to_resample(self, smc):
        """When to resample."""
        return smc.wgts.ESS < smc.N * smc.ESSrmin
    
    def resample(self, smc):
        """
        Resample the particles if the ESS is below the threshold set for the smc object.
        """
        self.rs_flag = self.time_to_resample(smc)
        if self.rs_flag:
            A = rs.resampling(smc.resampling, smc.W, M=smc.N)
            self.X = smc.X[A]
            # self.wgts = rs.Weights() # Empty rs.Weights object represents equal weights on particles
            self.wgts = rs.Weights(np.log(np.array([1/smc.N]*smc.N)))
        else:
            self.X = smc.X
            self.wgts = smc.wgts

    def move(self):
        """
        Move the particles to the next time step.
        """
        self.k += 1
        self.Xp = self.X
        self.X = self.PX_simulate()
        
    def predict(self):
        """
        Simulate from the predictive distribution
        """
        self.Y = self.PY_simulate()

    def fetch_single(self, smc):
        """Returns function of simulated particles for $X_{t+k) | Y_{1:t}$."""
        self.resample(smc)
        while self.k < self.K:
            self.move()
        if not self.predict_observations:   
            return self.out_func_X(self.wgts, self.X)
        else:
            self.predict()
            return self.out_func_Y(self.wgts, self.Y)
    
    def fetch_multi(self, smc):
        """Returns function of simulated particles for $X_{t+k) | Y_{1:t}$
            for each k"""
        predictive_output = {}
        self.resample(smc)
        while self.k < max(self.K):
            self.move()
            if self.k in self.K:
                if not self.predict_observations:
                    predictive_output.update({str(self.k): self.out_func_X(self.wgts, self.X)})
                else:
                    self.predict()
                    predictive_output.update({str(self.k): self.out_func_Y(self.wgts, self.Y)})
        # Return dict of simulated particles
        return predictive_output
    
class PredictiveCollector(Collector, WeightedPredictiveMixin, SimulatedPredictiveMixin):
    """
    Abstract base class for Predictive Collectors. Not to be used directly.
    
    To create a new predictive collector, inherit from this class and 
    
    - Set the `predict_observations` attribute
    - Implement the `out_func_X` and/or `out_func_Y` methods to define how to obtain the output of the collection from the input   

    We set the `predict_observations` attribute to True if the collector is used to collect the predictive distribution of the observations Y_{t+k} | Y_{1:t}.
    Otherwise, we set it to False (default), if the collector is used to collect the predictive distribution of the latent state X_{t+k} | Y_{1:t}.

    Then, we either implement the `out_func_X` and/or `out_func_Y` method based on whether we want to collect the predictive distribution of the latent state or
    the observations, respectively.
    
    Optionally, we can also change the 'signature' attribute if further input parameters are required for the collector.
    We can also change the `summary_name` attribute that determines the name of the attribute in the smc.summaries object 
    that will be used to store the output of the collection.
    """
    signature = {'K': 1, 'method': 'simulated'}
    predict_observations = False
    
    def __init__(self, **kwargs):
        """
        Inputs to the PredictiveCollector (defined through the signature dictionary):
        
        - K (list/int): number of steps ahead to predict
        - method (str): method to use to get particles for predictive distribution: 'simulated' or 'weighted'
        """
        Collector.__init__(self, **kwargs)
        self.t = 0
        self.k = 0        
        self._check_inputs()
        self._set_method_ext()
        if type(self.K) is int and self.method == 'simulated':
            for _ in range(self.K):
                self.summary.append(None)

    def _set_method_ext(self):
        if type(self.K) is int and self.method == 'simulated':
            self.method_ext = 'simulated_single'
        elif type(self.K) is list and self.method == 'simulated':
            self.method_ext = 'simulated_multi'
        elif type(self.K) is int and self.method == 'weighted':
            self.method_ext = 'weighted'

    def _check_inputs(self):
        if self.method not in ['simulated', 'weighted']:
            raise ValueError(f"PredictiveCollector: method must be set to either 'simulated' or 'weighted'.")
        if self.K != 1 and self.method == 'weighted':
            raise ValueError(f"PredictiveCollector: method 'weighted' only supported for K=1.")

    def first_collection(self, smc):
        return smc.t == 0
    
    def last_collection(self, smc):
        return smc.t >= self.fk.T - 1

    def pre_process(self, smc):
        self.fk = smc.fk
        self.N = smc.N
        
    def post_process(self, smc):
        """
        Post process the predictive distribution.

        # self.summary = do_something(self.summary)
        # setattr(smc.summaries, self.summary_name, self.summary)
        """
        pass
            
    def fetch(self, smc):
        if self.first_collection(smc):
            self.pre_process(smc)
        if self.method_ext == 'weighted':
            out = self.fetch_weighted(smc) # Implemented in WeightedPredictiveMixin
        elif self.method_ext == 'simulated_single':
            out = self.fetch_single(smc) # Implemented in SimulatedPredictiveMixin
        else: # 'simulated_multi'
            out = self.fetch_multi(smc) # Implemented in SimulatedPredictiveMixin
        if self.last_collection(smc):
            self.post_process(smc)
        self.t += 1
        self.k = 0
        return out

    def out_func_X(self, wgts, X):
        raise NotImplementedError("class PredictiveCollector should not be used directly. Please subclass and define `out_func_X` method.")

    def out_func_Y(self, wgts, X):
        """Returns the simulated particles."""
        raise NotImplementedError("class PredictiveCollector should not be used directly. Please subclass and define `out_func_Y` method.")

#-----------------------------------------------Implemented examples of Predictive Collectors--------------------------------------------------------

class PredictiveParticles(PredictiveCollector):

    summary_name = "predictive_particles"
    predict_observations = False
    
    def out_func_X(self, wgts, X):
        """Returns the simulated particles."""
        return (wgts, X)

class ObservationPredictiveParticles(PredictiveCollector):
    
    summary_name = "obs_predictive_particles"
    predict_observations = True
    
    def out_func_Y(self, wgts, Y):
        """Returns the simulated particles."""
        return (wgts, Y)
    
class PredictiveMoments(PredictiveCollector):        
    """
    Collects empirical moments (e.g. mean and variance) of the particles
    for the predictive distribution of X_{t+k} | X_{1:t}.

    Moments are defined through a function phi with the following signature:

        def mom_func(W, X):
           return np.average(X, weights=W)  # for instance

    If no function is provided, the default moment of the Feynman-Kac class
    is used (mean and variance of the particles, see ``core.FeynmanKac``).
    """
    signature = {"K": 1, "method": 'simulated', "mom_func": None}
    summary_name = "predictive_moments"
    predict_observations = False

    def default_moments(self, W, X):
        """
        Default moments of the particles representing the signal
        """
        return self.fk.default_moments(W, X)

    def out_func_X(self, wgts, X):
    # Calculate moments
        f = self.default_moments if self.mom_func is None else self.mom_func
        return f(wgts.W, X)

class ObservationPredictiveMoments(PredictiveCollector):
    
    signature = {"K": 1, "method": 'simulated', "mom_func": None}
    summary_name = "obs_predictive_moments"
    predict_observations = True

    def default_moments(self, W, Y):
        """
        Default moments of particles for the observations
        """
        return rs.wmean_and_var(W, Y)
    
    def out_func_Y(self, wgts, Y):
    # Calculate moments
        f = self.default_moments if self.mom_func is None else self.mom_func
        return f(wgts.W, Y)

class NLPD(PredictiveCollector):
    """
    Calculates the negative log posterior density, based on the Gaussian KDE.
    Current implementation may have high compute cost due to algorithm used to 
    calculate the bandwidth.
    """
    summary_name = "nlpd"
    predict_observations = True
    
    def out_func_Y(self, wgts, Y):
        """Returns the NLPD, calculated using a KDE of particles Y."""
        if not self.has_y_true:
            return None
        else:
            W = wgts.W if not self.rs_flag else None
            kde = gaussian_kde(Y.T, weights=W) # May have high compute cost. Try different implementation if so.
            return -1. * kde.logpdf(self.y_true)

class LowVarianceNLPD(PredictiveCollector):
    """
    Calculates the negative log posterior density of the kernel density
    estimator of the predictive distribution $Y_{t+k}| Y_{1:t}}$.
    
    Uses the particle representation of X to come up with the kde for the density of y.
    Thus, no need to calibrate the bandwith of the kde. May have better performance.
    """    
    summary_name = "low_variance_nlpd"
    predict_observations = False
    summary_name = "nlpd"

    def out_func_X(self, wgts, X):
        if not self.has_y_true:
            return None
        else:
            return -1.*np.log(np.sum(wgts.W * np.exp(self.PY_logpdf())))
            
class AbsoluteError(PredictiveCollector):
    """
    Calculates the absolute error of K-step ahead prediction against the true data.
    """
    summary_name = "absolute_error"
    predict_observations = True
    
    def out_func_Y(self, wgts, Y):
        if not self.has_y_true:
            return None
        else:
            return np.abs(np.average(Y, weights=wgts.W, axis=0) - self.y_true)

#-----------------------------------------------------------------------------------------------