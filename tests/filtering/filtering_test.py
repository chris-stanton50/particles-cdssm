"""
Test for the filtering: 

We write in this script an experiment to test the CDSSM_SMC filters. 

The output of `multiCDSSM_SMC` will be of the following format:

{'run': 0, 'N': 10, 'num': 10, 'fk': 'BwG_DBr_NDBBrP', 'seed': 7401915, 'output': <particles.collectors.Summaries object at 0x104c3d150>}

- A list of dictionaries, where each dictionary corresponds to a single run of the SMC algorithm.
- Each dictionary will contain the following
    - 'run': int from 0 to nruns - 1
    - 'seed': int: the seed used in the run
    - 'output': the output of the out_func that was passed to the SMC object. In this case, it is the summaries object (with cpu time passed).
    - The unprotected inputs to the SMC object that were fed in as a list (e.g could be N, num, fk).
"""

import time
import dill
import numpy as np
import pandas as pd
import sys

from particles import SMC, multiSMC
from particles.collectors import Moments
import particles.state_space_models as ssms
from particles.kalman import Kalman

from particles_cdssm.core import summaries
from particles_cdssm.core import multiCDSSM_SMC
from particles_cdssm.continuous_discrete_ssms import MvCDSSM
from particles_cdssm.state_space_models import DiscreteDiscreteSSM
from particles_cdssm.tools import build_cdssm
import particles_cdssm.feynman_kac as sfk
from utils import kalman_results_to_frame, multismc_results_to_df
from cdssm_lib import CDSSM_LIB

T=10; N=100
nruns_smc = 96; nruns_cdssm_smc = 96
num=50
seed = True


if not len(sys.argv) >= 3:
    raise ValueError('Please provide a run_id and cdssm_string as an argument when running the script.')

run_id = int(sys.argv[1])
cdssm_str = str(sys.argv[2])

# run_id =30
# cdssm_str = 'iou'

# Build a cdssm for each noise level
cdssm_spec = CDSSM_LIB[cdssm_str]

cdssm = build_cdssm(cdssm_spec)
is1d = not isinstance(cdssm, MvCDSSM)

if seed:
    np.random.seed(cdssm_spec['seed'])
x, y = cdssm.simulate(T)

# Build the corresponding lgssm/ddssm for each noise level
if cdssm.islgssm:
    lgssm = cdssm.lgssm()
ddssm = DiscreteDiscreteSSM(cdssm=cdssm)

#-----------------------------------------------------------------------------------------
# Part 1: Extract the true values from the synthetic data with the Kalman filter/particle filter

part_1_cpu = time.perf_counter()
print('Part 1: Running Kalman filter to obtain true values:')

if not cdssm.islgssm:
    print('Cannot express CDSSM as an LGSSM. Skipping Kalman filter.')
else:
    print('Running Kalman filter for each noise level:')
    # Build a Kalman filter:
    kalman = Kalman(ssm=lgssm, data=y)

    # Run the Kalman filter:
    kalman.filter()
    
    # Extract the true values from the Kalman filter:
    true_vals_df = kalman_results_to_frame(kalman)

    print('Kalman filter runs complete.')

# print(f'Running benchmark guided particle filter with N={benchmark_N} and QMC for each noise level:')

# # Define FK/SMC objects with the ddssms at each noise level. Use a guided pf and QMC, with large N.
# benchmark_fks = [ssms.GuidedPF(ssm=ddssm, data=y_s[i]) for i, ddssm in enumerate(ddssms)]
# benchmark_algs = [SMC(fk=fk, N=benchmark_N, qmc=True, collect=[Moments]) for fk in benchmark_fks]

# # Run the algorithms
# for benchmark_alg in benchmark_algs:
#     benchmark_alg.run()

# # Extract the filtering means and marginal likelihood at time T from the Kalman filter:
# if isinstance(cdssms[0], MvCDSSM):
#     for i in range(cdssm.model_sde.dimX):
#         true_vals_df_dict[f'true_T_{i+1}_pf'] = [benchmark_alg.summaries.moments[-1]['mean'][i] for benchmark_alg in benchmark_algs]
# else:
#     true_vals_df_dict[f'true_T_pf'] = [benchmark_alg.summaries.moments[-1]['mean'] for benchmark_alg in benchmark_algs]
# true_vals_df_dict['logLt_pf'] = [benchmark_alg.summaries.moments[-1]['mean'] for benchmark_alg in benchmark_algs]
# print('Benchmark particle filter runs complete.')

# results_df_1 = pd.DataFrame(true_vals_df_dict)

part_1_cpu = time.perf_counter() - part_1_cpu
print(f'Part 1 complete. Run time: {round(part_1_cpu, 2)} seconds.')

#-----------------------------------------------------------------------------------------
# Part 2: Run the multiSMC algorithm for Bootstrap and Guided PF
part_2_cpu = time.perf_counter()
print('Part 2: Running multiSMC for different noise levels:')


# Extract the lgssm and synthetic data
fks = {'Bootstrap': ssms.Bootstrap(ssm=lgssm, data=y), 'GuidedPF': ssms.GuidedPF(ssm=lgssm, data=y)}

print(f'Running multiSMC for Bootstrap and Guided PF:')
results = multiSMC(nruns=nruns_smc, nprocs=0, out_func=summaries, collect=[Moments], fk=fks, N=N)
results_df_2 = multismc_results_to_df(results, continuous_discrete=False)
    
part_2_cpu = time.perf_counter() - part_2_cpu
print(f'Part 2 complete. Run time: {round(part_2_cpu, 2)} seconds.')
#-----------------------------------------------------------------------------------------

# Part 3: Run the multiCDSSM_SMC algorithm for different noise levels
part_3_cpu = time.perf_counter()
print('Part 3: Running multiCDSSM_SMC:')

# Generate all possible fk models for the given cdssm
fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)

# Run the multiCDSSM_SMC algorithm at the given noise level
tic = time.perf_counter()
results = multiCDSSM_SMC(nruns=nruns_cdssm_smc, nprocs=0, out_func=summaries, collect=[Moments], fk=fks, N=N, num=num)
cpu = time.perf_counter() - tic
results_df_3 = multismc_results_to_df(results, continuous_discrete=True)


part_3_cpu = time.perf_counter() - part_3_cpu
print(f'Part 3 complete. Run time: {round(part_3_cpu, 2)} seconds.')

#-----------------------------------------------------------------------------------------

# Part 4: Collect metadata
part_4_cpu = time.perf_counter()
print('Part 4: Collecting metadata and store all results')

metadata = {'N': N,
            'x': x,
            'y': y, 
            'cdssm': cdssm,
            'lgssm': lgssm, 
            'nruns_smc': nruns_smc, 
            'nruns_cdssm_smc': nruns_cdssm_smc, 
            }

# Store data from the 4 parts:

true_vals_df.to_json(f'./results/filtering_test_run_{run_id}_{cdssm_str}_part_1.json', index=False)
results_df_2.to_json(f'./results/filtering_test_run_{run_id}_{cdssm_str}_part_2.json', index=False)
results_df_3.to_json(f'./results/filtering_test_run_{run_id}_{cdssm_str}_part_3.json', index=False)

with open(f'./results/filtering_test_run_{run_id}_{cdssm_str}_meta.pkl', 'wb') as f:
    dill.dump(metadata, f)
    
part_4_cpu = time.perf_counter() - part_4_cpu
print(f'Part 4 complete. Run time: {round(part_4_cpu, 2)} seconds.')
#-----------------------------------------------------------------------------------------
    
print('Filtering experiment complete.')

part_cpus = [part_1_cpu, part_2_cpu, part_3_cpu, part_4_cpu]
total_cpu = sum(part_cpus)

for i, cpu_time in enumerate(part_cpus):
    print(f'Part {i+1} run time: {round(cpu_time, 2)} seconds')

print(f'Total CPU time: {round(total_cpu, 2)} seconds')