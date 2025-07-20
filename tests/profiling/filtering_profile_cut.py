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

mv_ou: 21.24s vs 0.28s (iou vs mv_ou): Could be an around 100x speedup on the table!

Single run: 

Simulation of M: 0.65s vs 0.003s (iou vs mv_ou)

We get the same run times when we reduce the above to simulation of the auxiliary bridge.

Evaluation of logG:
"""

# Build the cdssm
cdssm_str = 'mv_ou'
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, False)


# Simulate synthetic data from the cdssm
x, y = cdssm.simulate(T)

# Use a backward guided model:
if cdssm_str.startswith('mv'):
    # For the OU model, we can use the MvDriftBrownianAuxBridge
    fk = sfk.BackwardGuidedDA(cdssm, y, auxiliary_bridge_cls=axb.MvDriftBrownianAuxBridge)
elif cdssm_str.startswith('i'):
    fk = sfk.BackwardGuidedDA(cdssm, y, auxiliary_bridge_cls=axb.IntegratedDriftBrownianAuxBridge)

# Build the algorithm:
alg = CDSSM_SMC(fk=fk, N=N, num=10, store_history=True)
alg.next()

xp = alg.X
t = alg.t

N = xp.shape[0]; x_start = xp[xp.dtype.names[-1]]
end_point_proposal = alg.fk._build_end_point_proposal(t, x_start)
x_end = end_point_proposal.rvs(N)
aux_bridge = alg.fk._build_aux_bridge(t, x_start, x_end)
numerical_scheme = aux_bridge.numerical_scheme_cls(aux_bridge)


size=N
t_start = 0.0
t_end = 1.0
t_diff = t_end - t_start
sims = numerical_scheme._create_state_container(t_diff, num, size, dim=numerical_scheme.SDE.dimX)
param_names = sims.dtype.names
step, first_param = t_diff/num, param_names[0]


s = 0.0
x = x_start

# Compute drift and diffusion
drift = numerical_scheme.SDE.b(t_start, x_start)  # shape (N, d)
diffusion = numerical_scheme.SDE.sigma(t_start, x_start)  # shape (n_sim, d, m)
# Generate Brownian increments
dW = np.sqrt(step) * stats.norm.rvs(size=(size, numerical_scheme.SDE.dimW))
# Euler-Maruyama update
x_step = x + drift * step + np.einsum('ijk,ik->ij', diffusion, dW)
sims[first_param] = x_step
i=1
prev_time, curr_param, prev_param = t_start + i*step, param_names[i], param_names[i-1]

# def M(self, t, xp):
#     N = xp.shape[0]; x_start = xp[xp.dtype.names[-1]]
#     end_point_proposal = self._build_end_point_proposal(t, x_start)
#     x_end = end_point_proposal.rvs(N)
#     aux_bridge = self._build_aux_bridge(t, x_start, x_end)
#     aux_bridge.simulate(N, x_start, num=self.num)
    
# Time the simulation of particles
n_repeats = 10

"""
LinearSDEs to check:
"""
cpus = []
for _ in range(n_repeats):
    cpu = perf_counter()
    for _ in range(num):
        # aux_bridge.LinearSDE._a(t_start, t_end)
        # aux_bridge.LinearSDE._b(t_start, t_end)
        aux_bridge.LinearSDE._v(t_start, t_end)
        if hasattr(aux_bridge.LinearSDE, '_a_cached'):
            # del aux_bridge.LinearSDE._a_cached
            # del aux_bridge.LinearSDE._b_cached
            del aux_bridge.LinearSDE._v_cached
    cpu = perf_counter() - cpu
    cpus.append(cpu)

print(f'Run complete: average cpu: {np.mean(cpu)} std cpu: {np.std(cpu)} seconds')
print(f'All cpus: {cpus}')

def simulation_step(self, size, t, x, step):
    """
    Given starting point(s) 'x' and a time 't', uses the Euler-Maruyama scheme
    to simulate the SDE at time 't+step':

    This is a *critical* function. This will be called multiple times in any SMC
    algorithm, so speeding this up could have significant impacts on algorithm performance.
    
    Inputs
    -------------
    size (int): The number of SDE paths.
    t (float): The current time 
    x (np.ndarray): A (size, dimX) array for the current point on the SDE.
    step (float): The step size

    Returns
    -------------
    x_step (np.ndarray): An array of shape (size, dimX) for the SDE paths at time t+step
    """
    return x_step