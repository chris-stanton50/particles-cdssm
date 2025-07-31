import arviz as az

import particles.state_space_models as ssms
from particles.kalman import LinearGauss
from particles.utils import multiplexer

from cdssm_lib import CDSSM_LIB
from particles_cdssm.mcmc import PIMH, CDSSM_PIMH, ICSMC, CDSSM_ICSMC, mcmc_worker
from particles_cdssm.state_space_models import DiscreteDiscreteSSM
import particles_cdssm.feynman_kac as sfk

# Data params
T = 5
cdssm_str = 'iou'
smoothing=True

# MCMC Params
niter=100;  Nx=10; num=5
methods = ['pimh', 'icsmc', 'icsmc_bs']

cdssm = build_cdssm(cdssm_str, smoothing=smoothing)
x, y = cdssm.simulate(T)
fk = sfk.BackwardReparameterisedDA(cdssm=cdssm, data=y)

print(f'Running CDSSM MCMC algorithms for cdssm {cdssm_str} in parallel:')
out = multiplexer(f=mcmc_worker, nruns=8, nprocs=0, seeding=True, fk=fk, method=methods, niter=niter, Nx=Nx, num=num)
print('Algorithm runs complete.')

"""
This bit tests the algorithms on standard SSMs and FKs, no data augmentation:
"""

example_ssms = [ssms.Gordon_etal(), ssms.BearingsOnly(), ssms.DiscreteCox(), ssms.ThetaLogistic(), LinearGauss()]
example_fks = [None]*len(example_ssms)

for i, ssm in enumerate(example_ssms):
    x, y = ssm.simulate(T)
    fk_cls = ssms.Bootstrap if ssm.__class__.__name__ != 'LinearGauss' else ssms.GuidedPF
    fk = fk_cls(ssm=ssm, data=y)
    example_fks[i] = fk

# Add a final ssm that is the discrete time version of a CD-SSM:
cdssm = build_cdssm(cdssm_str, smoothing=smoothing)
if cdssm.islgssm:
    ssm = cdssm.lgssm()
else:
    ssm = DiscreteDiscreteSSM(cdssm=cdssm)

x, y = ssm.simulate(T)
fk = ssms.GuidedPF(ssm=ssm, data=y)

example_fks.append(fk)
    
# Test that implementations of PIMH and CSMC (excl. the parameter) run:
        
print(f'Running SSM MCMC algorithms for different ssms in parallel:')
out = multiplexer(f=mcmc_worker, nruns=2, nprocs=0, seeding=True, fk=example_fks, method=methods, niter=niter, Nx=Nx)
print('Algorithm runs complete.')

# Test that implementations of the PIMH and CSMC (excl. the parameter) run: