import numpy as np
import scipy.stats as stats

import particles.mcmc as mcmc
import particles.state_space_models as ssms
import particles.smc_samplers as ssp
import particles.distributions as dists
from particles.core import SMC
from particles.kalman import Kalman
from particles.mcmc import CSMC

import sdes.continuous_discrete_ssms as cdssms
import sdes.feynman_kac as sfk
from sdes.core import CDSSM_SMC, CDSSM_FeynmanKac
from sdes.tools import init_kwargs_dict


"""
Smoothing (X_{1:T} | Y_{1:T}), with no involvement of the parameter.

    For standard State Space Models:

    PIMH: (Particle Independent Metropolis-Hastings) *Ready to be tested*
    CSMC: (Conditional SMC MCMC): Conditional SMC for standard SSMs. *Ready to be tested*

    For CDSSMs:
    
    CDSSM_PIMH: (Particle Independent Metropolis-Hastings) *Ready to be tested*    
    CSMC: (Conditional SMC MCMC): Conditional SMC for CDSSMs. *Ready to be tested*
    
    
Joint Smoothing (X_{1:T}, \theta | Y_{1:T}).

IMMH: (Ideal Marginal Metropolis Hastings) (LGSSM only) *Ready to be tested*
    Marginal MCMC algorithm that targets theta for Linear Gaussian 
        State space models. Likelihood is computed using Kalman filter.

PMMH: (Particle Marginal Metropolis Hastings) *Ready to be tested*
    Extends the version developed in the `particles` package by addtionally, storing the latent states.


Particle Gibbs:

AutoParticleGibbs
    - PG sampler for state space models. Automatically updates the 
    parameter by using a Metropolis-within-Gibbs step for the whole parameter vector.

For CD-SSMs:

CDSSMParticleGibbs
    - PG sampler for CD-SSMs.
"""

class GenericGibbs(mcmc.GenericGibbs):

    def step(self, n):
        """ Added to correct mistake in particles package."""
        self.chain.theta[n] = self.update_theta(self.chain.theta[n - 1], self.x)
        self.x = self.update_states(self.chain.theta[n], self.x)
        if self.store_x:
            self.chain.x[n] = self.x

class ParticleGibbs(mcmc.ParticleGibbs):
        
    def step(self, n):
        """ Added to correct mistake in particles package."""
        self.chain.theta[n] = self.update_theta(self.chain.theta[n - 1], self.x)
        self.x = self.update_states(self.chain.theta[n], self.x)
        if self.store_x:
            self.chain.x[n] = self.x

class X_MCMC(mcmc.MCMC):
    """
    MCMC Algorithms for State Space Models that only target the latent states X, and not the parameter.
    """
    def print_progress(self, n):
        # Need to overwrite method in MCMC class, as it makes reference to `self.chain.theta`
        msg = "Iteration %i" % n
        if hasattr(self, "nacc") and n > 0:
            msg += ", acc. rate=%.3f" % (self.nacc / n)
        print(msg)
    
class CDSSM_MCMC(mcmc.MCMC):
    
    def _check_cdssm_and_fk(self, cdssm_cls, fk_cls):
        if not issubclass(cdssm_cls, cdssms.CDSSM):
            raise TypeError('cdssm class must be a subclass of CDSSM.')
        if not issubclass(fk_cls, CDSSM_FeynmanKac):
                raise TypeError('fk_cls must be a subclass of CDSSM_FeynmanKac.')

# Uncomment this if you use it when building the CDSSM_SMC algorithms.            
    # def _build_cdssm(self, cdssm_options, theta):
    #     if cdssm_options is not None:
    #         return self.cdssm_cls(**{**cdssm_options, **ssp.rec_to_dict(theta)})
    #     else:
    #         return self.cdssm_cls(**ssp.rec_to_dict(theta))
    
# The API for the standard particle filter        
class PIMH(X_MCMC):
    """
    Particle Independent Metropolis-Hastings. For offline smoothing of the x process, via MCMC.
    We generate an MCMC chain of samples from the smoothing distribution.
    
    Ready to be tested.
    """
    def __init__(
        self,
        fk=None,
        niter=10,
        Nx=100,
        verbose=10,
        smc_options=None
        ):
        """
        Parameters
        ----------
        fk: FeynmanKac model instance (default=None)
            The Feynman-Kac formalism of a selected ssm to use.
            Already contains the underlying ssm and the data.
        niter: int
            number of iterations for the MCMC
        Nx: int
            number of particles (for the particle filter that evaluates the
            likelihood)
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        smc_options: dict
            Options to pass to the SMC class.
        """
        for k in ["fk", "niter", "Nx", "verbose"]:
            setattr(self, k, locals()[k])
        self.nacc = 0
        # do not collect summaries, store the history
        self.smc_options = {"collect": "off", "store_history": True}
        if smc_options is not None:
            self.smc_options.update(smc_options)
        x = self.state_container(niter, len(self.fk.data))
        self.chain = ssp.ThetaParticles(x=x, lpost=np.empty(shape=niter))

    def state_container(self, N, T):
        law_X0 = self.fk.ssm.PX0()
        dim = law_X0.dim
        shape = [N, T]
        if dim > 1:
            shape.append(dim)
        return np.empty(shape, dtype=law_X0.dtype)

    def step0(self):
        pf = self.alg_instance()
        pf.run()
        self.chain.x[0] = pf.hist.extract_one_trajectory()
        self.chain.lpost[0] = pf.logLt
        
    def step(self, n):
        pf = self.alg_instance()
        pf.run()
        lp_acc = pf.logLt - self.chain.lpost[n - 1]
        if np.log(stats.uniform.rvs()) < lp_acc:  # accept
            self.chain.x[n] = pf.hist.extract_one_trajectory()
            self.nacc += 1
            self.chain.lpost[n] = pf.logLt
        else:  # reject
            self.chain.x[n] = self.chain.x[n - 1].copy()
            self.chain.lpost[n] = self.chain.lpost[n - 1]

    def alg_instance(self):
        pf = SMC(fk=self.fk, N=self.Nx, **self.smc_options)
        return pf

    @property
    def acc_rate(self):
        return self.nacc / (self.chain.N - 1)

class CDSSM_PIMH(PIMH, CDSSM_MCMC):
    """
    Pathspace implementation of PIMH, for CDSSMs.
    
    Ready to be tested.
    """
    def __init__(
        self,
        fk=None,
        niter=10,
        verbose=10,
        Nx=100,
        num=10,
        smc_options=None
    ):
        """
        Parameters
        ----------
        fk: FeynmanKac model instance (default=None)
            The Feynman-Kac formalism of a selected cdssm to use.
            Already contains the underlying ssm and the data.
        niter: int
            number of iterations for the MCMC
        Nx: int
            number of particles (for the particle filter that evaluates the
            likelihood)
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        num: int
            The number of imputed points (for the paths in the SMC)
        smc_options: dict
            Options to pass to the CDSSM_SMC class.
            
        Ready to be tested.
        """
        for k in ["fk", "niter", "verbose", "Nx", "num"]:
            setattr(self, k, locals()[k])
        self.nacc = 0
        self.smc_options = {"collect": "off", "store_history": True}
        if smc_options is not None:
            self.smc_options.update(smc_options)
        cdssm = fk.cdssm
        x = cdssm.__class__.state_container(niter, len(self.fk.data), num, cdssm.delta_s, dimX=cdssm.model_sde.dimX)
        self.chain = ssp.ThetaParticles(x=x, lpost=np.empty(shape=niter))
        
    def alg_instance(self):
        pf = CDSSM_SMC(fk=self.fk, N=self.Nx, num=self.num, **self.smc_options)
        return pf
             
class ICSMC(X_MCMC):
    """
    Iterated Conditional Sequential Monte Carlo. For offline smoothing of the x process, via MCMC.
    We generate an MCMC chain of samples from the smoothing distribution.
    Uses the CSMC Kernel to update the states, and not the parameter.
    
    Ready to be tested.
    """        
    def __init__(
        self,
        fk=None,
        niter=10,
        Nx=100,
        backward_step=False,
        verbose=10,
    ):
        """
        Parameters
        ----------
        fk: FeynmanKac model instance (default=None)
            The Feynman-Kac formalism of a selected ssm to use.
            Already contains the underlying ssm and the data.
        niter: int
            number of iterations for the MCMC
        Nx: int
            number of particles (for the CPF)
        backward_step: bool (default=False)
            whether to run the backward step
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        
        Ready to be tested.
        """
        for k in ["fk", "niter", "Nx", "verbose", "backward_step"]:
            setattr(self, k, locals()[k])
        x = self.state_container(niter, len(self.fk.data))
        self.chain = ssp.ThetaParticles(x=x)

    def state_container(self, N, T):
        law_X0 = self.fk.ssm.PX0()
        dim = law_X0.dim
        shape = [N, T]
        if dim > 1:
            shape.append(dim)
        return np.empty(shape, dtype=law_X0.dtype)
    
    def step0(self):
        pf = self.alg_instance(0)
        pf.run()
        if self.backward_step:
            self.x = pf.hist.backward_sampling_ON2(1)
        else:
            self.x = pf.hist.extract_one_trajectory()
        self.chain.x[0] = self.x

    def step(self, n):
        cpf = self.alg_instance(n)
        cpf.run()
        if self.backward_step:
            self.x = cpf.hist.backward_sampling_ON2(1)
        else:
            self.x = cpf.hist.extract_one_trajectory()
        self.chain.x[n] = self.x

    def alg_instance(self, n):
        if n == 0:
            # Consistent with `particles` package: no SMC options for CSMC
            pf = SMC(fk=self.fk, N=self.Nx, store_history=True)
        else:
            # Consistent with `particles` package: no SMC options for CSMC
            pf = mcmc.CSMC(fk=self.fk, N=self.Nx, xstarp=self.x)
        return pf

class CDSSM_CSMC(CSMC, CDSSM_SMC):
    """
    CSMC (as a subclass of an SMC object) for CDSSMs. Inherits the methods
    `self.resample_move' and `self.generate_particles` from the CSMC class
    in the `particles` package to fix the star trajectory.  
    """
    def __init__(self, fk=None, N=100, ESSrmin=0.5, xstar=None, num=10):
        CDSSM_SMC.__init__(
            self,
            fk=fk,
            N=N,
            resampling="multinomial",
            ESSrmin=ESSrmin,
            store_history=True,
            collect="off",
            num=num
        )
        self.xstar = xstar

class CDSSM_ICSMC(ICSMC, CDSSM_MCMC):
    """
    Pathspace implementation of CSMC, for CDSSMs.
    
    Ready to be tested.
    """
    def __init__(
        self,
        fk=None,
        niter=10,
        Nx=100,
        num=10,
        backward_step=False,
        verbose=10
    ):
        """
        Parameters
        ----------
        fk: FeynmanKac model instance (default=None)
            The Feynman-Kac formalism of a selected cdssm to use.
            Already contains the underlying ssm and the data.
        niter: int
            number of iterations for the MCMC
        Nx: int
            number of particles (for the CPF)
        num: int
            The number of imputed points (for the paths in the CSMC)        
        backward_step: bool (default=False)
            whether to run the backward step
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        """
        for k in ["fk", "niter", "Nx", "num", "backward_step", "verbose"]:
            setattr(self, k, locals()[k])
        cdssm = fk.cdssm
        x = cdssm.__class__.state_container(niter, len(self.fk.data), num, cdssm.delta_s, dimX=cdssm.model_sde.dimX)
        self.chain = ssp.ThetaParticles(x=x)
        
    def alg_instance(self, n):
        if n == 0:
            # Consistent with `particles` package: no `SMC options` for CSMC
            pf = CDSSM_SMC(fk=self.fk, N=self.Nx, store_history=True, num=self.num)
        else:
            # Consistent with `particles` package: no `SMC options` for CSMC
            pf = CDSSM_CSMC(fk=self.fk, N=self.Nx, xstar=self.x, num=self.num)
        return pf
                         
# class IMMH(mcmc.GenericRWHM):
#     """
#     Implementation of IMMH: Ideal Marginal Metropolis-Hastings.
#     Marginal MCMC algorithm that targets theta for Linear Gaussian
#     State space models. Likelihood is computed using Kalman filter.
#     """
#     def __init__(
#         self,
#         niter=10,
#         verbose=0,
#         ssm_cls=None,
#         prior=None,
#         data=None,
#         theta0=None,
#         adaptive=True,
#         scale=1.0,
#         rw_cov=None,
#     ):    
#         """
#         Parameters
#         ----------
#         niter: int
#             number of iterations
#         verbose: int (default=0)
#             print some info every `verbose` iterations (never if 0)
#         ssm_cls: MVLinearGauss class
#             The considered parametric class of linear, gaussian state-space models.
#             Must be a subclass of MVLinearGauss, so that one can implement the Kalman
#             filter.
#         prior: StructDist
#             the prior
#         data: list-like
#             the data
#         theta0: structured array of length=1
#             starting point (generated from prior if =None)
#         adaptive: True/False
#             If true, random walk covariance matrix is adapted recursively
#             based on past samples; see also scale and rw_cov for extra info.
#         scale: positive scalar (default = 1.)
#             in the adaptive case, covariance of the proposal is scale^2 times
#             (2.38^2 / d) times the current estimate of the target covariance
#         rw_cov: (d, d) array (defaults to Identity matrix if not provided)
#             covariance matrix of the random walk proposal if adaptive=False;
#             if adaptive=True, rw_cov is used as a preliminary guess for the
#             covariance matrix of the target.
#         """
#         if not issubclass(ssm_cls, ssms.MVLinearGauss):
#             raise TypeError('ssm_cls must be a subclass of MVLinearGauss.')
#         self.ssm_cls = ssm_cls
#         self.prior = prior
#         self.data = data
#         generic_rwhm_kwargs_dict = init_kwargs_dict(mcmc.GenericRWHM, locals())
#         mcmc.GenericRWHM.__init__(self, **generic_rwhm_kwargs_dict)

#     def alg_instance(self, theta):
#         return Kalman(ssm=self.ssm_cls(**theta), data=self.data)
    
#     def compute_post(self):
#         self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
#         if np.isfinite(self.prop.lpost[0]):
#             kf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
#             kf.filter()
#             self.prop.lpost[0] += np.sum(kf.logpyt)

# class PMMH(mcmc.GenericRWHM):
#     """Particle Marginal Metropolis Hastings.

#     PMMH is class of Metropolis samplers where the intractable likelihood of
#     the considered state-space model is replaced by an estimate obtained from
#     a particle filter.
    
#     Extends the implementation in the `particles` package to store the latent states.
#     """
#     def __init__(
#         self,
#         niter=10,
#         verbose=0,
#         ssm_cls=None,
#         smc_cls=SMC,
#         prior=None,
#         data=None,
#         smc_options=None,
#         fk_cls=ssms.Bootstrap,
#         Nx=100,
#         theta0=None,
#         adaptive=True,
#         scale=1.0,
#         rw_cov=None,
#         store_x=False
#     ):
#         """
#         Parameters
#         ----------
#         niter: int
#             number of iterations
#         verbose: int (default=0)
#             print some info every `verbose` iterations (never if 0)
#         ssm_cls: StateSpaceModel class
#             the considered parametric class of state-space models
#         smc_cls: class (default: particles.SMC)
#             SMC class
#         prior: StructDist
#             the prior
#         data: list-like
#             the data
#         smc_options: dict
#             options to pass to class SMC
#         fk_cls: (default=ssms.Bootstrap)
#             FeynmanKac class associated to the model
#         Nx: int
#             number of particles (for the particle filter that evaluates the
#             likelihood)
#         theta0: structured array of length=1
#             starting point (generated from prior if =None)
#         adaptive: True/False
#             If true, random walk covariance matrix is adapted recursively
#             based on past samples; see also scale and rw_cov for extra info.
#         scale: positive scalar (default = 1.)
#             in the adaptive case, covariance of the proposal is scale^2 times
#             (2.38^2 / d) times the current estimate of the target covariance
#         rw_cov: (d, d) array (defaults to Identity matrix if not provided)
#             covariance matrix of the random walk proposal if adaptive=False;
#             if adaptive=True, rw_cov is used as a preliminary guess for the
#             covariance matrix of the target.
#         store_x: bool (default=False)
#             Whether to store the latent states.
            
#         Ready to be tested.
#         """
#         mcmc.PMMH.__init__(
#             self,
#             niter=niter,
#             verbose=verbose,
#             ssm_cls=ssm_cls,
#             smc_cls=smc_cls,
#             prior=prior,
#             data=data,
#             smc_options=smc_options,
#             fk_cls=fk_cls,
#             Nx=Nx,
#             theta0=theta0,
#             adaptive=adaptive,
#             scale=scale,
#             rw_cov=rw_cov,
#             store_x=store_x
#             )
#         if store_x:
#             # Overwrite self.chain and self.arr to include storage for the latent states            
#             self.chain = ssp.ThetaParticles(
#                                         theta=np.empty(shape=niter, dtype=self.prior.dtype),
#                                         x = ssm_cls.state_container(niter, len(self.data)),
#                                         lpost=np.empty(shape=niter)
#                                         )
#             self.arr = ssp.view_2d_array(self.chain.theta)
#             self.smc_options['store_history'] = True

#     def step0(self):
#         th0 = self.prior.rvs(size=1) if self.theta0 is None else self.theta0
#         if self.store_x: 
#             self.prop = ssp.ThetaParticles(theta=th0,
#                                            x=self.ssm_cls.state_container(1, len(self.data)), 
#                                            lpost=np.zeros(1)
#                                            )
#         else:
#             self.prop = ssp.ThetaParticles(theta=th0, lpost=np.zeros(1))
#         self.prop_arr = ssp.view_2d_array(th0)
#         self.compute_post()
#         self.chain.copyto_at(0, self.prop, 0)
                    
#     def compute_post(self):
#         self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
#         if np.isfinite(self.prop.lpost[0]):
#             pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
#             pf.run()
#             self.prop.lpost[0] += pf.logLt
#         if self.store_x:
#             self.prop.x[0] = pf.hist.extract_one_trajectory()

#     def alg_instance(self, theta):
#         return mcmc.PMMH.alg_instance(self, theta)
            
# class MetropoliswithinGibbs(mcmc.GenericRWHM):
#     """
#     Use within Gibbs samplers for automated parameter updates. Not to be used alone.
    
#     Note: Could make this class more efficient by not storing all of the theta particles,
#     as the ones we are interested in will be stored in the underlying Gibbs sampler.
#     """
#     def __init__(
#         self, niter=10, verbose=0, ssm_cls=None, prior=None, data=None, theta0=None, adaptive=True, scale=1.0, rw_cov=None
#     ):
#         """
#         Parameters
#         ----------
#         niter: int
#             number of MCMC iterations
#         verbose: int (default=0)
#             progress report printed every (niter/verbose) iterations (never if 0)
#         ssm_cls: StateSpaceModel class
#             the considered parametric class of state-space models
#         prior: StructDist
#             the prior
#         data: list-like
#             the data
#         theta0: structured array of size=1 or None
#             starting point, simulated from the prior if set to None
#         adaptive: True/False
#             If true, random walk covariance matrix is adapted recursively
#             based on past samples; see also scale and rw_cov for extra info.
#         scale: positive scalar (default = 1.)
#             in the adaptive case, covariance of the proposal is scale^2 times
#             (2.38^2 / d) times the current estimate of the target covariance
#         rw_cov: (d, d) array (defaults to Identity matrix if not provided)
#             covariance matrix of the random walk proposal if adaptive=False;
#             if adaptive=True, rw_cov is used as a preliminary guess for the
#             covariance matrix of the target.
#         """
#         self.ssm_cls = ssm_cls
#         self.prior = prior
#         self.data = data
#         generic_rwhm_kwargs_dict = init_kwargs_dict(GenericRWHM, locals())
#         GenericRWHM.__init__(self, **generic_rwhm_kwargs_dict)

#     def loglik(self, theta, x):
#         ssm = self.ssm_cls(**{k: theta[k] for k in theta.dtype.names})
#         loglik = ssm.PX0().logpdf(x[0]) + ssm.PY(0, None, x[0]).logpdf(self.data[0][0])
#         for t in range(1, len(self.data)):
#             loglik += ssm.PX(t, x[t-1]).logpdf(x[t]) + ssm.PY(t, x[t-1], x[t]).logpdf(self.data[t][0])
#         return loglik
    
#     def compute_post(self):
#         self.prop.lpost = self.prior.logpdf(self.prop.theta) + self.loglik(self.prop.theta, self.x)


# class AutoGibbs(GenericGibbs):
#     """
#     Gibbs sampler for state space models. Automatically updates the 
#     parameter by using a Metropolis-within-Gibbs step.
    
#     Must be subclassed with the method `update_states` defined.    
#     """
#     def __init__(
#         self,
#         niter=10,
#         verbose=10,
#         theta0=None,
#         ssm_cls=None,
#         prior=None,
#         data=None,
#         store_x=False, 
#         adaptive=True,
#         scale=1.0,
#         rw_cov=None,
#         N_steps=1,
#     ):  
#         generic_gibbs_kwargs_dict = init_kwargs_dict(GenericGibbs, locals())
#         GenericGibbs.__init__(self, **generic_gibbs_kwargs_dict)
#         self.N_steps = N_steps
#         self.theta0 = self.prior.rvs(size=1) if theta0 is None else theta0 # Assign theta0 early so it can be passed to MWG
#         mwg_kwargs_dict = {**init_kwargs_dict(MetropoliswithinGibbs, locals()), **{'niter': self.N_steps*niter, 'theta0': self.theta0}}
#         self.mwgibbs = MetropoliswithinGibbs(**mwg_kwargs_dict)
#         self.n_mwg = 0

#     def update_theta(self, theta, x):
#         self.mwgibbs.x = x # Pass the current value of x to the MWG
#         for _ in range(self.N_steps):
#             if self.n_mwg == 0:
#                 self.mwgibbs.step0()
#             else:
#                 self.mwgibbs.step(self.n_mwg)
#             self.n_mwg += 1
#         theta = self.mwgibbs.chain.theta[self.n_mwg - 1]
#         return theta

# class AutoParticleGibbs(ParticleGibbs, AutoGibbs):
#     """
#     Implementation of Particle Gibbs that automatically updates theta
#     using a Metropolis-within-Gibbs step. Can be used without subclassing.
#     """
#     def __init__(
#         self,
#         niter=10,
#         verbose=0,
#         ssm_cls=None,
#         prior=None,
#         data=None,
#         theta0=None,
#         Nx=100,
#         fk_cls=None,
#         regenerate_data=False,
#         backward_step=False,
#         store_x=False,
#         adaptive=True,
#         scale=1.0,
#         rw_cov=None,
#         N_steps=1
#         ):
#             autogibbs_kwargs = init_kwargs_dict(AutoGibbs, locals())
#             AutoGibbs.__init__(self, **autogibbs_kwargs)
#             self.Nx = Nx
#             self.fk_cls = ssms.Bootstrap if fk_cls is None else fk_cls
#             self.regenerate_data = regenerate_data
#             self.backward_step = backward_step

# class CDSSM_PMMH(mcmc.PMMH, CDSSM_MCMC):

#     def __init__(self, num=10., cdssm_options=None, **kwargs): # This code doesn't assign any default values. Could do this the other way around.
#         PMMH.__init__(self, **kwargs)
#         self.cdssm_options = cdssm_options
#         self.num = num

#     def alg_instance(self, theta):
#         if self.cdssm_options is not None:
#             cdssm = self.ssm_cls(**{**self.cdssm_options, **theta})
#         else:
#             cdssm = self.ssm_cls(**theta)
#         return self.smc_cls(
#                             fk=self.fk_cls(cdssm=cdssm, data=self.data), 
#                             N=self.Nx, 
#                             num=self.num,
#                             **self.smc_options
#                             )
                
# class CDSSM_MetropoliswithinGibbs(MetropoliswithinGibbs, CDSSM_MCMC):
#     """
#     Metropolis within Gibbs for Continuous-Discrete State Space Models. Not to be used alone.
#     """
#     def __init__(
#         self, niter=10, verbose=0, cdssm_cls=None, cdssm_options=None, fk_cls=None, prior=None, data=None, theta0=None, adaptive=True, scale=1.0, rw_cov=None
#     ):
#         """
#         Parameters
#         ----------
#         niter: int
#             number of MCMC iterations
#         verbose: int (default=0)
#             progress report printed every (niter/verbose) iterations (never if 0)
#         cdssm_cls: CDSSM class
#             the considered parametric class of state-space models
#         cdssm_options: dict
#             Additional options for CDSSMs that are not parameters. 
#             Possible keys include starting point 'x0' and time step size 'delta_s'.
#         fk_cls: CDSSM_FeynmanKac class
#             The Feynman-Kac model for the CDSSM.
#         prior: StructDist
#             the prior
#         data: list-like
#             the data
#         theta0: structured array of size=1 or None
#             starting point, simulated from the prior if set to None
#         adaptive: True/False
#             If true, random walk covariance matrix is adapted recursively
#             based on past samples; see also scale and rw_cov for extra info.
#         scale: positive scalar (default = 1.)
#             in the adaptive case, covariance of the proposal is scale^2 times
#             (2.38^2 / d) times the current estimate of the target covariance
#         rw_cov: (d, d) array (defaults to Identity matrix if not provided)
#             covariance matrix of the random walk proposal if adaptive=False;
#             if adaptive=True, rw_cov is used as a preliminary guess for the
#             covariance matrix of the target.
#         """
#         self._check_ssm_and_fk(cdssm_cls, fk_cls)
#         mwg_kwargs_dict = init_kwargs_dict(MetropoliswithinGibbs, locals())        
#         MetropoliswithinGibbs.__init__(self, **mwg_kwargs_dict)
#         self.fk_cls = fk_cls
#         self.cdssm_cls = cdssm_cls

#     def loglik(self, theta, x):
#         theta = {k: theta[k] for k in theta.dtype.names}
#         cdssm = self.cdssm_cls(**theta)
#         fk_mod = self.fk_cls(cdssm=cdssm, data=self.data)
#         loglik = fk_mod.logpt(0, None, x[0]) + cdssm.PY(0, None, x[0]).logpdf(self.data[0][0])
#         for t in range(1, len(self.data)):
#             loglik += fk_mod.logpt(t, x[t-1], x[t]) + cdssm.PY(t, x[t-1], x[t]).logpdf(self.data[t][0])
#         return loglik    
    
# class CDSSM_ParticleGibbs(AutoParticleGibbs, CDSSM_MCMC):
#     """"""    
#     def __init__(
#         self,
#         niter=10,
#         verbose=0,
#         cdssm_cls=None,
#         cdssm_options=None,
#         prior=None,
#         data=None,
#         theta0=None,
#         Nx=100,
#         fk_cls=BootstrapReparameterisedDA_DH,
#         num=10,
#         regenerate_data=False,
#         backward_step=False,
#         store_x=False,
#         adaptive=True,
#         scale=1.0,
#         rw_cov=None,
#         N_steps=1
#         ):  
#             self._check_ssm_and_fk(cdssm_cls, fk_cls)
#             local_vars = locals()
#             for k in ["cdssm_cls", "prior", "data", "theta0", "niter", "store_x", "verbose", "N_steps", "Nx", "fk_cls", "regenerate_data", "backward_step", "num", "cdssm_options"]:
#                 setattr(self, k, local_vars[k])
#             self.theta0 = self.prior.rvs(size=1) if theta0 is None else theta0 # Assign theta0 early so it can be passed to MWG
#             mwg_kwargs = {**init_kwargs_dict(CDSSM_MetropoliswithinGibbs, local_vars), **{'theta0': self.theta0, 'niter': N_steps*niter}}
#             self.mwgibbs = CDSSM_MetropoliswithinGibbs(**mwg_kwargs) # Do this a few times to make the code look nicer
#             self.n_mwg = 0
#             self.delta_s = cdssm_options['delta_s'] if (cdssm_options and 'delta_s' in cdssm_options) else 1.
#             self.build_chain_container()

#     def build_chain_container(self):
#         theta = np.empty(shape=self.niter, dtype=self.prior.dtype)
#         if self.store_x:
#             # Remember: when changing this code, 'state_container' 
#             # the keyword argument 'dimX'
#             x = self.cdssm_cls.state_container(self.niter, len(self.data), self.num, self.delta_s)
#             self.chain = ssp.ThetaParticles(theta=theta, x=x)
#         else:
#             self.chain = ssp.ThetaParticles(theta=theta)

#     def fk_mod(self, theta):
#         cdssm = self._build_cdssm(self.cdssm_options, theta)
#         return self.fk_cls(cdssm=cdssm, data=self.data)

#     def update_states(self, theta, x):
#         fk = self.fk_mod(theta)
#         if x is None:
#             cpf = CDSSM_SMC(fk=fk, N=self.Nx, store_history=True, num=self.num)
#         else:
#             cpf = CDSSM_CSMC(fk=fk, N=self.Nx, xstar=x, num=self.num)
#         cpf.run()
#         if self.backward_step:
#             new_x = cpf.hist.backward_sampling_ON2(1)
#         else:
#             new_x = cpf.hist.extract_one_trajectory()
#         if self.regenerate_data:
#             self.data = fk.ssm.simulate_given_x(new_x)
#         return new_x

#     def samples_transform_W_to_X(self):
#         if hasattr(self, 'transformed'):
#             raise ValueError('The samples have already been transformed.')
#         for i in range(self.niter):
#             theta = self.chain.theta[i]
#             fk = self.fk_mod(theta)
#             trans_x = fk.sample_transform_W_to_X(self.chain.x[i])
#             self.chain.x[i] = trans_x
#         self.transformed = True
        
# # Below are particular classes that run algorithms on LGSSMs.
# class SingleSiteLGGibbs(GenericGibbs):

#     def update_states(self, theta, x):
#         """
#         Input: theta: structured array containing single theta.
#                 x: (T, ) numpy arrray containing states.
#         Output: x: (T, ) numpy array containing states.
#         """
#         T = len(self.data)
#         if x is None:
#             # Initialise x by simulating from the model given initial parameter
#             ssm = self.ssm_cls(**ssp.rec_to_dict(theta))
#             x_list, _ = ssm.simulate(T)
#             x_new = np.array([x_t[0] for x_t in x_list])
#             return x_new
#         x_new = x.copy()
#         x_new[0] = self.single_state_cond_dist(theta, 0., x[1], self.data[0][0]).rvs()
#         for t in range(1, T-1):
#             x_new[t] = self.single_state_cond_dist(theta, x[t-1], x[t+1], self.data[t][0]).rvs()
#         x_new[-1] = self.final_state_cond_dist(theta, x[-2], self.data[-1][0]).rvs()
#         return x_new

#     def single_state_cond_dist(self, theta, xp, xf, yt):
#         sigmaY_2 = 0.01 ** 2
#         # A  = (1 + theta['rho'] ** 2)/theta['sigmaX_2'] + 1./theta['sigmaY_2']
#         # B = (theta['rho']*(xp + xf))/theta['sigmaX_2'] + yt/theta['sigmaY_2'] 
#         A  = (1 + theta['rho'] ** 2)/theta['sigmaX_2'] + 1./sigmaY_2
#         B = (theta['rho']*(xp + xf))/theta['sigmaX_2'] + yt/sigmaY_2 
        
#         loc = B/A; scale=np.sqrt(1/A)
#         return dists.Normal(loc=loc, scale=scale)
    
#     def final_state_cond_dist(self, theta, xp, yt):
#         sigmaY_2 = 0.01 ** 2
#         # A  = 1./theta['sigmaX_2'] + 1./theta['sigmaY_2']
#         # B = (theta['rho']*(xp))/theta['sigmaX_2'] + yt/theta['sigmaY_2'] 
#         A  = 1./theta['sigmaX_2'] + 1./sigmaY_2
#         B = (theta['rho']*(xp))/theta['sigmaX_2'] + yt/sigmaY_2 

#         loc = B/A; scale=np.sqrt(1/A)
#         return dists.Normal(loc=loc, scale=scale)
    
#     def update_theta(self, theta, x):
#         """
#         Input: theta: structured array containing single theta.
#                 x: (T, ) numpy arrray containing states.
#         Output: new_theta: (T, ) structured array containing single theta.
#         """
#         posterior_dist = self.ssm_cls.posterior(x, self.data, **self.prior.hyperparams)        
#         new_theta = posterior_dist.rvs()
#         return new_theta
    
# class SingleSiteAutoLGGibbs(AutoGibbs, SingleSiteLGGibbs):
    
#     def update_states(self, theta, x):
#         return SingleSiteLGGibbs.update_states(self, theta, x)

# class LGPGibbs(ParticleGibbs):
    
#     def update_theta(self, theta, x):
#         """
#         Input: theta: structured array containing single theta.
#                 x: (T, ) numpy arrray containing states.
#         Output: new_theta: list containing single theta.
#         """
#         posterior_dist = self.ssm_cls.posterior(np.array(x), self.data, **self.prior.hyperparams)        
#         new_theta = posterior_dist.rvs() 
#         return new_theta