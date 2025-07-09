# API

We present a high level API below for the functions implemented within the `particles_cdssm` package:

# Algorithms for v0.1

The classes/functions listed below will comprise the first release version of the `particles_cdssm` package. 

Particle-MCMC based classes remain under development.

## Online Filtering ($X_t | Y_{1:t}=y_{1:t}$)

### Standard SSMs

- Particle Filters (`particles_cdssm.feynman_kac.SMC`) (new version with `genealogy tracking` included in the `ParticleHistory` object)

### CD-SSMs

- Particle Filters (`particles_cdssm.feynman_kac.CDSSM_SMC`)
    - Bootstrap Particle Filter 
        - (`particles_cdssm.feynman_kac.BootstrapDA`)
        - (`particles_cdssm.feynman_kac.BootstrapReparameterisedDA`)
    - Guided Particle Filter (Forward Proposals)
        - (`particles_cdssm.feynman_kac.ForwardGuidedDA`)
        - (`particles_cdssm.feynman_kac.ForwardReparameterisedDA`)
    - Guided Particle Filter (Backward Proposals)
        - (`particles_cdssm.feynman_kac.BackwardGuidedDA`)
        - (`particles_cdssm.feynman_kac.BackwardReparameterisedDA`)

- Parallel Particle Filters (`particle_cdssm.core.multiCDSSM_SMC`)

Used to run `CDSSM_SMC` algorithms in parallel. To run `SMC` algorithms in parallel, use 
the corresponding function `particles.core.multiSMC` in the particles package.

## Offline Smoothing ($X_{1:t} | Y_{1:t}=y_{1:t}$)

### Standard SSMs

- Particle-based smoothers (other smoothers for standard SSMs are implemented in the particles package)
    - Genealogy Tracking (`particles_cdssm.smoothing.backward_sampling_genealogy`)

- Particle MCMC-based smoothers
    - PIMH (`particles_cdssm.mcmc.PIMH`)
    - iCMSC (`particles_cdssm.mcmc.ICSMC`)

### CD-SSMs

- Particle-based smoothers 
    - Genealogy Tracking (`particles_cdssm.smoothing.CDSSM_ParticleHistory.backward_sampling_genealogy`)
    - FFBS-ON2 (`particles_cdssm.smoothing.CDSSM_ParticleHistory.backward_sampling_ON2`)
    - FFBS-MCMC (`particles_cdssm.smoothing.CDSSM_ParticleHistory.backward_sampling_mcmc`)

- Parallel Offline Smoothing (`particle_cdssm.core.smoothing_worker`)

Implements a particle based smoothing method, either for standard `SSMs` or for `CD-SSMs` for multiple additive functions. 
To be used in conjunction with `particles.utils.multiplexer` to run either `SMC` or `CDSSM_SMC` based smoothing algorithms in parallel.

# For future development

The items listed below will be a part of future developments:

## Offline Smoothing ($X_{1:t} | Y_{1:t}=y_{1:t}$)

- Particle MCMC-based smoothers
    - PIMH (`particles_cdssm.mcmc.CDSSM_PIMH`) (*Under Development*)
    - iCMSC (`particles_cdssm.mcmc.CDSSM_ICSMC`) (*Under Development*)

# Joint Offline Smoothing ($X_{1:t}, \theta | Y_{1:t}=y_{1:t}$)

- Particle MCMC
    - Particle Marginal Metropolis Hastings (PMMH) (`particles_cdssm.mcmc.CDSSM_PMMH`)
    - Particle Gibbs (PG) (`particles_cdssm.mcmc.CDSSM_ParticleGibbs`) (`backward_step = False`)
    - Particle Gibbs with Backward Step (PGBS) (`particles_cdssm.mcmc.CDSSM_ParticleGibbs`)(`backward_step = True`)

## Online Smoothing for Additive Functionals ($X_{1:t} | Y_{1:t}=y_{1:t}$)

In the draft version, each of the collectors has been extended to store estimators w.r.t multiple additive functionals.

- Naive online smoothing (equiv to genealogy tracking) (`particles_cdssm.collectors.MultiOnline_smooth_naive`)
- Forward Additive $\mathcal{O}(N^2)$ (`particles_cdssm.collectors.MultiOnline_smooth_ON2`)
- Forward Additive MCMC (`particles_cdssm.collectors.MultiOnline_smooth_mcmc`)

## Joint Online Smoothing for CD-SSMs: ($X_{1:t}, \theta | Y_{1:t}=y_{1:t}$)

One could also in the future implement a version of $SMC^2$ that is compatible for CD-SSMs.