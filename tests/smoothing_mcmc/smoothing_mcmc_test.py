"""
Test whether the smoothing algorithms are working well for the given CDSSM class:
"""
import dill
import time
import sys
import numpy as np
import pandas as pd
import arviz as az

import particles.state_space_models as ssms
from particles.utils import multiplexer
from particles.kalman import Kalman

from particles_cdssm.cdssm_lib import CDSSM_LIB, build_cdssm
from particles_cdssm.continuous_discrete_ssms import MvCDSSM
from particles_cdssm.state_space_models import DiscreteDiscreteSSM
import particles_cdssm.feynman_kac as sfk
from particles_cdssm.mcmc import mcmc_worker


def obs_times_to_store(T, discrete_time=True):
    """
    Returns an array of 10 observation times and preprends time 1, giving a total of
    11 times. Uses the integer valued, discrete time index.
    """
    Ts_to_store_excl_1 = np.linspace(T // 10, T, num=10, dtype=np.int64)
    arr_1 = np.array([1], dtype=np.int64)
    obs_times_to_store = np.concatenate([arr_1] + [Ts_to_store_excl_1], axis=0)
    obs_times_to_store = obs_times_to_store - 1 # subtract 1 from all times
    if discrete_time:
        return obs_times_to_store
    else:
        if cdssm.isobservedat0:
            return np.array([cdssm.S(t) for t in obs_times_to_store], dtype=np.float64)
        else:
            return np.array([cdssm.S(t+1) for t in obs_times_to_store], dtype=np.float64)
            

def kalman_results_to_df(kalman):
    """
    Converts the output of the RTS smoother to a pd.DataFrame, containing the true value quantities 
    of interest for the experiment.
    """
    T = len(kalman.data)
    results_df_dict = {}
    obs_times = obs_times_to_store(T)
    results_df_dict['T'] = obs_times.tolist()
    filt = kalman.filt; smth = kalman.smth
    dimX = kalman.filt[0].mean.ravel().shape[0]
    for d in range(dimX):
        results_df_dict[f'x_t_filt_mean_{d+1}'] = [filt[t].mean.ravel()[d] for t in obs_times]
        results_df_dict[f'x_t_filt_std_{d+1}'] = [np.sqrt(np.diag(filt[t].cov))[d] for t in obs_times]
        results_df_dict[f'x_t_smth_mean_{d+1}'] = [smth[t].mean.ravel()[d] for t in obs_times]
        results_df_dict[f'x_t_smth_std_{d+1}'] = [np.sqrt(np.diag(smth[t].cov))[d] for t in obs_times]
    results_df = pd.DataFrame(results_df_dict)
    results_df.set_index('T', inplace=True)
    return results_df

T = 100

# MCMC Params
niter=100; Nx=100; num=20

debug = False

if debug:
    run_id = 50
    cdssm_str = 'ou'
else:
    if not len(sys.argv) >= 2:
        raise ValueError('Please provide a run_id and cdssm_str as an argument when running the script.')

    run_id = int(sys.argv[1])
    cdssm_str = str(sys.argv[2])

# Build the cdssm
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, smoothing=True)

np.random.seed(cdssm_spec['seed'])
x, y = cdssm.simulate(T, num=100)

is1d = not isinstance(cdssm, MvCDSSM)

# Pull fk_names from the cdssm_spec
fk_names = cdssm_spec['smoothing_fk_names']
filt_fk_names = [name for name in fk_names if name[2] != 'R']
smth_fk_names = [name for name in fk_names if name[2] == 'R']

# Define LGSSM (if possible) and a ddssm:
if cdssm.islgssm:
    lgssm=cdssm.lgssm()
    
ddssm = DiscreteDiscreteSSM(cdssm=cdssm)

# Part 1: Extract the true values from the synthetic data using the RTS smoother:
#-----------------------------------------------------------------------------------------

part_1_cpu = time.perf_counter()
print('Part 1: Running RTS smoothers and benchmark particle smoothers')

results_dfs_1 = []

if not cdssm.islgssm:
    print('Cannot express CDSSM as an LGSSM. Skipping Kalman filter.')
else:
    print('Running Kalman filter for each value of T:')    
    # Run the RTS smoother for the LGSSM for each value of T
    kalman = Kalman(ssm=lgssm, data=y)
    kalman.smoother()
    print('RTS smoother runs complete.')
    results_df_1_kalman = kalman_results_to_df(kalman)
    results_dfs_1.append(results_df_1_kalman)

results_df_1 = pd.concat(results_dfs_1, axis=1)

part_1_cpu = time.perf_counter() - part_1_cpu
print(f'Part 1 complete. Run time: {round(part_1_cpu, 2)} seconds.')

# Part 2: Run the MCMC based smoothing methods for SSMs and CD-SSMs using a fixed number of particles and imputed points:
#-----------------------------------------------------------------------------------------

part_2_cpu = time.perf_counter()
print(f'Part 2: Running SSM/CDSSM MCMC algorithms for cdssm {cdssm_str} in parallel:')
    
filt_fks = {'Bootstrap': ssms.Bootstrap(ssm=ddssm, data=y), 'GuidedPF': ssms.GuidedPF(ssm=ddssm, data=y)}
smth_fks = {'Bootstrap': ssms.Bootstrap(ssm=ddssm, data=y), 'GuidedPF': ssms.GuidedPF(ssm=ddssm, data=y)}

cdssm_filt_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)
cdssm_smth_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=True)

filt_fks.update(cdssm_filt_fks)
smth_fks.update(cdssm_smth_fks)

results_pimh_icsmc = multiplexer(f=mcmc_worker, nruns=1, nprocs=0, seeding=True, fk=filt_fks, method=['pimh', 'icsmc'], niter=niter, Nx=Nx, num=num)
results_icsmc_bs = multiplexer(f=mcmc_worker, nruns=1, nprocs=0, seeding=True, fk=smth_fks, method=['icsmc_bs'], niter=niter, Nx=Nx, num=num)
print('Algorithm runs complete.')

idata_pimh = az.concat([r['output'] for r in results_pimh_icsmc if r['method'] == 'pimh'], dim='chain')
idata_icsmc = az.concat([r['output'] for r in results_pimh_icsmc if r['method'] == 'icsmc'], dim='chain')
idata_icsmc_bs = az.concat([r['output'] for r in results_icsmc_bs], dim='chain')

part_2_cpu = time.perf_counter() - part_2_cpu
print(f'Part 2 complete. Run time: {round(part_2_cpu, 2)} seconds.')

# Part 3: Collect metadata
#-----------------------------------------------------------------------------------------

part_3_cpu = time.perf_counter()
print('Part 3: Collecting metadata and store all results')

metadata = {'stored_obs_times': obs_times_to_store(T),
            'num': num,
            'x': x,
            'y': y,
            'cdssm': cdssm,
            'lgssm': lgssm if cdssm.islgssm else None,
            'ddssm': ddssm,
            }

# Store data from the 3 parts:

results_df_1.to_json(f'./results/smoothing_mcmc_test_run_{run_id}_{cdssm_str}_part_1.json', index=False)

idatas = {'pimh': idata_pimh, 'icsmc': idata_icsmc, 'icsmc_bs': idata_icsmc_bs}
for name, idata in idatas.items():
    idata.posterior = idata.posterior.sel({'time': obs_times_to_store(T, discrete_time=False)}) # Only store a subset of times 
    idata.to_netcdf(f'./results/smoothing_mcmc_test_run_{run_id}_{cdssm_str}_part_2_{name}.nc')

with open(f'./results/smoothing_test_run_{run_id}_{cdssm_str}_meta.pkl', 'wb') as f:
    dill.dump(metadata, f)
    
part_3_cpu = time.perf_counter() - part_3_cpu
print(f'Part 3 complete. Run time: {round(part_3_cpu, 2)} seconds.')

print('Smoothing MCMC experiment complete.')

print(f'Part 1 run time: {round(part_1_cpu, 2)} seconds')
print(f'Part 2 run time: {round(part_2_cpu, 2)} seconds')
print(f'Part 3 run time: {round(part_3_cpu, 2)} seconds')

part_cpus = [part_1_cpu,  part_2_cpu, part_3_cpu]
total_cpu = sum(part_cpus)

print(f'Total CPU time: {round(total_cpu, 2)} seconds')
