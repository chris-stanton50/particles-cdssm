"""
Script for running fhn_smoothing experiment in the paper in parallel.

python fhn_smoothing.py 10 
"""

import sys
import dill
from cdssm_lib import CDSSM_LIB

from particles.utils import multiplexer

from particles_cdssm.tools import build_cdssm
from particles_cdssm.mcmc import mcmc_worker, CDSSM_ICSMC
import particles_cdssm.feynman_kac as sfk
import arviz as az

# if not len(sys.argv) >= 2:
#     raise ValueError('Please provide a run_id as an argument when running the script.')

run_id = 10

# Data params
T = 10 # May need adjusting

# MCMC Params
niter=100;  Nx=10; num=10

cdssm_spec = CDSSM_LIB['ifhn']
cdssm = build_cdssm(cdssm_spec)

print('Simulating synthetic data...')
x, y = cdssm.simulate(T)

fks = {'bootstrap': sfk.BootstrapDA(cdssm=cdssm, data=y),
       'backward_guided': sfk.BackwardGuidedDA(cdssm=cdssm, data=y), 
       'backward_reparameterised': sfk.BackwardReparameterisedDA(cdssm=cdssm, data=y)}
    
alg_boot = CDSSM_ICSMC(fks['bootstrap'], niter=niter, Nx=Nx, num=num, backward_step=False, verbose=10)
alg_guided = CDSSM_ICSMC(fks['backward_guided'], niter=niter, Nx=Nx, num=num, backward_step=False, verbose=10)
alg_reparameterised = CDSSM_ICSMC(fks['backward_reparameterised'], niter=niter, Nx=Nx, num=num, backward_step=False, verbose=10)

alg_boot.run()
alg_guided.run()
alg_reparameterised.run()

# print(f'Running MCMC bootstrap in parallel...')
# out_boot = multiplexer(f=mcmc_worker, nruns=8, nprocs=0, seeding=True, fk=fks['bootstrap'], method=['icsmc'], niter=niter, Nx=Nx, num=num)

# print(f'Running MCMC backward guided in parallel...')
# out_guided = multiplexer(f=mcmc_worker, nruns=8, nprocs=0, seeding=True, fk=fks['backward_guided'], method=['icsmc'], niter=niter, Nx=Nx, num=num)

# print(f'Running MCMC backward reparameterised in parallel...')
# out_reparameterised = multiplexer(f=mcmc_worker, nruns=8, nprocs=0, seeding=True, fk=fks['backward_reparameterised'], method=['icsmc_bs'], niter=niter, Nx=Nx, num=num)

# idata_bootstrap = az.concat([r['output'] for r in out_boot], dim='chain')
# idata_guided = az.concat([r['output'] for r in out_guided], dim='chain')
# idata_reparameterised = az.concat([r['output'] for r in out_reparameterised], dim='chain')

# idatas = {'bootstrap_icsmc': idata_bootstrap, 'guided_icsmc': idata_guided, 'reparameterised_icsmc_bs': idata_reparameterised}

# for name, idata in idatas.items():
#     idata.posterior = idata.posterior.sel({'time': obs_times_to_store(T, discrete_time=False)}) # Only store a subset of times 
#     idata.to_netcdf(f'./results/run_{run_id}_{name}.nc')
    
# metadata = {'x': x,
#             'y': y, 
#             'cdssm': cdssm,
#             }

# with open(f'./results/run_{run_id}_meta.pkl', 'wb') as f:
#     dill.dump(metadata, f)