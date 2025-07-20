from particles_cdssm.cdssm_lib import CDSSM_LIB, build_cdssm
from particles_cdssm.core import CDSSM_SMC
import particles_cdssm.feynman_kac as sfk
import particles_cdssm.auxiliary_bridges as axb
from time import perf_counter

import numpy as np
import scipy.stats as stats

T = 10; N=100; num=10
"""
Total time for simulation:  

Full filter run, for T=10:

Original: 21.24s vs 0.28s vs 0.005 (iou vs mv_ou vs ou): Could be an around 100x speedup on the table!
First Correction: 0.90s vs 0.06s vs 0.005 (iou vs mv_ou): Managed to get 24x speedup

Original: Simulation of M: 0.65s vs 0.003s (iou vs mv_ou)
First Correction: Simulation of M: 0.015s vs 0.005s (iou vs mv_ou)

mv_ou: Indep False /BrownianMotion
diag_cov True/False: (0.282, 0.283, 0.260)/(0.185, 0.162, 0159)

iou: Indep False /BrownianMotion
diag_cov True/False: (0.819, 0.805, 0.813)/(0.792, 0.764, 0.757)

So, assuming diagonal covariance makes the code run slower, likely due to looping over N particles.

iou: Indep False / OrnsteinUhlenbeck
diag_cov True/False: (0.475)/(0.26274)

Available speedups removing looping over N:

- _a and _v of HypoellipticLinearSDE: will affect all runs when model SDE is hypoelliptic.
- C and _v of MvIndepBrownianMotion: will affect MvEllipticSDEs when auxiliary LinearSDE is BM, covariance matrix is diagonal.
- B, C, _a and _v of MvIndepOrnsteinUhlenbeck: will affect MvEllipticSDEs when the auxiliary LinearSDE is OU
"""

# Build the cdssm
cdssm_str = 'ou'
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, False, indep=False)

# Simulate synthetic data from the cdssm
x, y = cdssm.simulate(T)
cdssm.model_sde._diag_cov = True

# Use a backward guided model:
if cdssm_str.startswith('mv'):
    # For the OU model, we can use the MvDriftBrownianAuxBridge
    fk = sfk.BackwardGuidedDA(cdssm, y, auxiliary_bridge_cls=axb.MvDriftBrownianAuxBridge)
elif cdssm_str.startswith('i'):
    fk = sfk.BackwardGuidedDA(cdssm, y, auxiliary_bridge_cls=axb.IntegratedDriftBrownianAuxBridge)
else:
    fk = sfk.BackwardGuidedDA(cdssm, y, auxiliary_bridge_cls=axb.DriftBrownianAuxBridge)

# Build the algorithm:
alg = CDSSM_SMC(fk=fk, N=N, num=10, store_history=True)
alg.run()
    
# Time the simulation of particles
n_repeats = 100
"""
LinearSDEs to check:
"""
cpus = []
for _ in range(n_repeats):
    cpu = perf_counter()
    alg = CDSSM_SMC(fk=fk, N=N, num=10, store_history=True)
    alg.run()
    cpu = perf_counter() - cpu
    cpus.append(cpu)
cpu = np.array(cpus).ravel()

print(f'Run complete: average cpu: {round(np.mean(cpu), 5)} std cpu: {round(np.std(cpu), 6)} seconds')
print()