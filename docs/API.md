# API

We present a high level API below for the functions implemented within the `particles_cdssm` package:

# Algorithms for v0.1

The classes/functions listed below will comprise the first release version of the `particles_cdssm` package. 

Particle-MCMC based classes that extend to parameter vector $\theta$ remain under development.

## Online Filtering ($X_t | Y_{1:t}=y_{1:t}: t \in \{1, \dots, T\}$)

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

Used to run `CDSSM_SMC` algorithms in parallel. To run `SMC` algorithms in parallel, use the corresponding function `particles.core.multiSMC` in the particles package.

## Particle based Offline Smoothing ($X_{1:T} | Y_{1:T}=y_{1:T}$)

### Standard SSMs

- Genealogy Tracking (`particles_cdssm.smoothing.backward_sampling_genealogy`)

### CD-SSMs
 
- Genealogy Tracking (`particles_cdssm.smoothing.CDSSM_ParticleHistory.backward_sampling_genealogy`)
- FFBS-ON2 (`particles_cdssm.smoothing.CDSSM_ParticleHistory.backward_sampling_ON2`)
- FFBS-MCMC (`particles_cdssm.smoothing.CDSSM_ParticleHistory.backward_sampling_mcmc`)
- Parallel Smoothing (`particles_cdssm.core.smoothing_worker`)

## Particle-MCMC based Offline Smoothing ($X_{1:T} | Y_{1:T}=y_{1:T}$)

### Standard SSMs

- Particle Independent Metropolis Hastings (PIMH) (`particles_cdssm.mcmc.PIMH`)
- Iterated Conditional Sequential Monte Carlo (iCSMC) `particles_cdssm.mcmc.ICSMC`
- Parallel Particle MCMC `particles_cdssm.mcmc.mcmc_worker`

### CD-SSMs

- Particle Independent Metropolis Hastings (PIMH) (`particles_cdssm.mcmc.CDSSM_PIMH`)
- Iterated Conditional Sequential Monte Carlo (iCSMC) (`particles_cdssm.mcmc.CDSSM_ICSMC`)
- Parallel Particle MCMC (`particles_cdssm.mcmc.mcmc_worker`)

# For future development

The items listed below will be a part of future developments:

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