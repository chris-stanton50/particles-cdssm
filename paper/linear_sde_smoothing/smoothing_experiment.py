#!/usr/bin/env python3
"""
Smoothing experiment for linear SDEs used in the paper: 

Can be run with the command:

python smoothing_experiment.py config_name

where config_name is the name of a JSON config file in ./config.

Saves the results in the ./results directory.
 

In this file, we will write a new smoothing experiment:

- We will fix T, and evaluate performance of estimation across the entire smoothing distribution.
- We store Kalman smoothing means and stds across the CD-SSM.
    - (T, dimX, 2) true values are stored (mean and std)
- We then generate estimators of E[X_t] and E[X_t^2] at each t < T, across each dimension
    and store the results in a dataframe. 
    - (T, nruns, dimX, 2) estimators are generated
    For this part, think a bit about how to construct the df and store the data.
- We then store the metadata.

To calculate the run time of hypoelliptic smoothing, take run time of mv smoothing and multiply by 40.
"""
import dill
import time
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path

from particles.utils import multiplexer
from particles.kalman import Kalman
from particles.collectors import Moments
import particles.state_space_models as ssms

from particles_cdssm.tools import build_cdssm
from particles_cdssm.continuous_discrete_ssms import MvCDSSM
from particles_cdssm.state_space_models import DiscreteDiscreteSSM
from particles_cdssm.core import SMC, CDSSM_SMC
import particles_cdssm.feynman_kac as sfk
from particles_cdssm.core import CDSSM_SMC, smoothing_worker

from particles_cdssm.cdssm_lib import CDSSM_LIB
from particles_cdssm.tools import isremote
from utils import obs_times_to_store, kalman_results_to_df, multismooth_results_to_df, benchmark_pf_results_to_df
"""
`mv_ou'
Ts = [10, 31, 100, 316, 1000]
N=100; num=10; nruns=2
benchmark_N = 10000

Smoothing experiment complete.
Part 1 run time: 178.79 seconds
Part 2 run time: 8.71 seconds
Part 3 run time: 87.49 seconds
Part 4 run time: 27.98 seconds
Total CPU time: 135.83 seconds

Smoothing experiment complete.
Part 1 run time: 178.79 seconds
Part 2 run time: 10.3 seconds
Part 3 run time: 283.04 seconds
Part 4 run time: 28.12 seconds
Total CPU time: 500.25 seconds
"""

parser = argparse.ArgumentParser(description="Run a linear SDE smoothing experiment.")
parser.add_argument(
    "-c",
    "--config",
    required=True,
    type=Path,
    help="Name of the JSON configuration file, with or without the .json suffix.",
)
args = parser.parse_args()

config_name = args.config.stem
config_path = Path("config") / args.config
if config_path.suffix != ".json":
    config_path = config_path.with_suffix(".json")

print(f"Running smoothing experiment for config {config_name}.json")

with config_path.open("r") as file:
    config = json.load(file)

T = config['T']
quantiles = np.array(config['quantiles'])
N_FFBS_MCMC = config['N_FFBS_MCMC']
N_genealogy = config['N_genealogy']
num = config['num']
nruns = config['nruns']
benchmark_N = config['benchmark_N']
debug = config['debug']
pf_benchmark = config['pf_benchmark']
part_3 = config['part_3']
cdssm_str = config['cdssm']
fk_names = config['fk_model_snames']

cdssm_spec = CDSSM_LIB[cdssm_str]

# Build the cdssm
cdssm = build_cdssm(cdssm_spec)

np.random.seed(config['seed'])
x, y = cdssm.simulate(T)

is1d = not isinstance(cdssm, MvCDSSM)

filt_fk_names = [name for name in fk_names if name[2] != 'R']
smth_fk_names = [name for name in fk_names if name[2] == 'R']

# Define LGSSM (if possible) and a ddssm:
if cdssm.islgssm:
    lgssm=cdssm.lgssm()
    
ddssm = DiscreteDiscreteSSM(cdssm=cdssm)

# Part 1 (a): Extract the true values from the synthetic data using the RTS smoother:
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

# part 1 (b): Run the Guided PF (QMC) and FFBS_MCMC:
#-----------------------------------------------------------------------------------------
if pf_benchmark:
    print('Running benchmark filter: Guided PF (QMC):')

    fk = ssms.GuidedPF(ssm=ddssm, data=y)
    alg = SMC(fk=fk, N=benchmark_N, collect=[Moments], qmc=True)
    alg.run()

    filt_out = alg.summaries.moments
    print('Running benchmark smoother: Guided PF (QMC):')
    smth_out = smoothing_worker(method='FFBS_MCMC', N=benchmark_N, fk=fk, smc_cls=SMC)

    results_df_1_pf = benchmark_pf_results_to_df(filt_out, smth_out)
    results_dfs_1.append(results_df_1_pf)

results_df_1 = pd.concat(results_dfs_1, axis=1)

part_1_cpu = time.perf_counter() - part_1_cpu
print(f'Part 1 complete. Run time: {round(part_1_cpu, 2)} seconds.')

# Part 2: Run the smoothing methods using standard smc algorithms, using a fixed number of particles:
#-----------------------------------------------------------------------------------------
part_2_cpu = time.perf_counter()
print('Part 2: Running SMC smoothing algorithms:')

fks = {'Bootstrap': ssms.Bootstrap(ssm=ddssm, data=y), 'GuidedPF': ssms.GuidedPF(ssm=ddssm, data=y)}
genealogy_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['genealogy'], N=N_genealogy, fk=fks, smc_cls=SMC, quantiles=quantiles)
FFBS_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['FFBS_MCMC'], N=N_FFBS_MCMC, fk=fks, smc_cls=SMC, quantiles=quantiles)
results = genealogy_results + FFBS_results
results_df_2 = multismooth_results_to_df(results)

part_2_cpu = time.perf_counter() - part_2_cpu
print(f'Part 2 complete. Run time: {round(part_2_cpu, 2)} seconds.')

# # Part 3: Run the smoothing methods for CD-SSMs using a fixed number of particles and imputed points:
#-----------------------------------------------------------------------------------------
if part_3:
    part_3_cpu = time.perf_counter()
    print('Part 3: Running CDSSM_SMC smoothing algortithms for different values of T:')
    filt_fks = sfk.gen_fk_models(cdssm, y, smoothing=False, fk_names=filt_fk_names) 
    smth_fks = sfk.gen_fk_models(cdssm, y, smoothing=True, fk_names=smth_fk_names)
    genealogy_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['genealogy'], N=N_genealogy, fk=filt_fks, num=num, smc_cls=CDSSM_SMC, quantiles=quantiles)
    FFBS_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['FFBS_MCMC'], N=N_FFBS_MCMC, fk=smth_fks, num=num, smc_cls=CDSSM_SMC, quantiles=quantiles)
    results = genealogy_results + FFBS_results
    results_df_3 = multismooth_results_to_df(results)

    part_3_cpu = time.perf_counter() - part_3_cpu
    print(f'Part 3 complete. Run time: {round(part_3_cpu, 2)} seconds.')
else:
    print('Skipping Part 3.')
    part_3_cpu = 0.

# Part 4: Collect metadata
#-----------------------------------------------------------------------------------------

part_4_cpu = time.perf_counter()
print('Part 4: Collecting metadata and store all results')

metadata = {'N_FFBS_MCMC': N_FFBS_MCMC,
            'N_genealogy': N_genealogy,
            'quantiles': quantiles,
            'stored_obs_times': obs_times_to_store(T),
            'num': num,
            'x': x,
            'y': y,
            'cdssm': cdssm,
            'lgssm': lgssm if cdssm.islgssm else None,
            'ddssm': ddssm,
            'nruns': nruns,
            'fk_names': filt_fk_names + smth_fk_names,
            }

# Store data from the 4 parts:

# Store data from the 4 parts:
local_or_remote = "remote" if isremote() else "local"
results_dir = Path("results") / local_or_remote
results_dir.mkdir(parents=True, exist_ok=True)
print(f"Storing results in {results_dir}")

results_df_1.to_json(results_dir / f"res_{config_name}_part_1.json", index=False)
results_df_2.to_json(results_dir / f"res_{config_name}_part_2.json", index=False)
if part_3:
    results_df_3.to_json(results_dir / f"res_{config_name}_part_3.json", index=False)

with (results_dir / f"res_{config_name}_meta.pkl").open("wb") as f:
    dill.dump(metadata, f)
    
part_4_cpu = time.perf_counter() - part_4_cpu
print(f'Part 4 complete. Run time: {round(part_4_cpu, 2)} seconds.')
print('Smoothing experiment complete.')

# for i, cpu_time in enumerate(part_cpus):
#     print(f'Part {i+1} run time: {round(cpu_time, 2)} seconds')

print(f'Part 1 run time: {round(part_1_cpu, 2)} seconds')
print(f'Part 2 run time: {round(part_2_cpu, 2)} seconds')
print(f'Part 3 run time: {round(part_3_cpu, 2)} seconds')
print(f'Part 4 run time: {round(part_4_cpu, 2)} seconds')

part_cpus = [part_1_cpu, part_2_cpu, part_3_cpu, part_4_cpu]
total_cpu = sum(part_cpus)

print(f'Total CPU time: {round(total_cpu, 2)} seconds')
#-----------------------------------------------------------------------------------------

# Run smoothing worker with multiplexer for all the fk models
# Note: If different add_funcs are set instead of the default choices, then 
# the add_funcs need to be passed as `protected_args`  to the multiplexer function.