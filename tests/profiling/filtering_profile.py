from particles_cdssm.sdes import IntegratedIndepOrnsteinUhlenbeck, IntegratedIndepBrownianMotion

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

# m = stats.norm.rvs(loc=-0.35, scale=0.7, size=(100, 1))
# s = np.ones((100, 1), dtype=np.float64)

# linear_sde = IntegratedIndepBrownianMotion(N=100, dimX=2, m=m, s=s)

rho = 0.3*np.ones((100, 1), dtype=np.float64)
mu = np.zeros((100, 1), dtype=np.float64)
phi =np.ones((100, 1), dtype=np.float64)

linear_sde = IntegratedIndepOrnsteinUhlenbeck(N=100, dimX=2, rho=rho, mu=mu, phi=phi)

self=linear_sde
s=0.; t=1.
"""
LinearSDEs to check:
"""
@profile
def covs(): # (N, dimX, dimX)
    if hasattr(self, '_v_cached') and self._v_delta_t == t-s:
        return self._v_cached
    delta_t = t - s; ns_1 = self.n_smooth + 1; N = self.N
    v_coefs = self.gen_v_coefs(delta_t) # (ns_1, ns_1, N, dimW, dimW)
    _v_block = lambda i, j, n: v_coefs[i, j, n] * self.Cov_rough[n]
    _v_blocks = [[np.stack([_v_block(i, j, n) for n in range(N)]) for j in range(ns_1)] for i in range(ns_1)] # (N, dimW, dimW)
    _v = np.concatenate([np.concatenate(v_row, axis=2) for v_row in _v_blocks], axis=1) # (N, dimX, dimX)
    self._v_cached = _v; self._v_delta_t = delta_t
    return _v

if __name__ == '__main__':
    covs()
    
# cpus = []
# for _ in range(n_repeats):
#     cpu = perf_counter()
#     for _ in range(num):
#         linear_sde._v(0., 1.)
#         if hasattr(linear_sde, '_v_cached'):
#             del linear_sde._v_cached
#     cpu = perf_counter() - cpu
#     cpus.append(cpu)

# kernprof -l -v your_script.py



# print(f'Run complete: average cpu: {np.mean(cpu)} std cpu: {np.std(cpu)} seconds')
# print(f'All cpus: {cpus}')
