"""
Start off by writing the experiment to run for a fixed number of particles. Then, show the performance of the methods for fixed n_cpu?
"""
import dill
import time
import sys
import numpy as np
import pandas as pd

from particles.utils import multiplexer
from particles.kalman import Kalman
from particles.collectors import Moments
import particles.state_space_models as ssms

from particles_cdssm.cdssm_lib import CDSSM_LIB, build_cdssm
from particles_cdssm.continuous_discrete_ssms import MvCDSSM
from particles_cdssm.state_space_models import DiscreteDiscreteSSM
from particles_cdssm.core import SMC, CDSSM_SMC
import particles_cdssm.feynman_kac as sfk
from particles_cdssm.core import CDSSM_SMC, smoothing_worker

def multismooth_results_to_df(results):
    """
    Converts the output of the multiplexer + smoothing_worker algorithm to a pd.DataFrame, 
    containing the quantities of interest for the experiment.
    """
    results_df_dict = {} 
    results_df_dict['run_id'] = [r['run'] for r in results]
    results_df_dict['seed'] = [r['seed'] for r in results]
    results_df_dict['method'] = [r['method'] for r in results]
    results_df_dict['fk'] = [r['fk'] for r in results]
    results_df_dict['cpu'] = [r['cpu_time'] for r in results]
    example_mean = results[0]['phi_x'][0]
    quantile_index = results[0]['quantile_index']
    if isinstance(example_mean, float):
        results_df_dict['x_0_est_1'] = [r['phi_x'][0] for r in results]
        results_df_dict['x_0_sq_est_1'] = [r['phi_x_x'][0] for r in results]
        for j, q in enumerate(quantile_index):
            q_str = str(round(q, 2))
            results_df_dict[f'x_0_q{q_str}_est_1'] = [r['quantiles'][j][0] for r in results]
    else:
        for i in range(len(example_mean)):
            results_df_dict[f'x_0_est_{i+1}'] = [r['phi_x'][0][i] for r in results]
            results_df_dict[f'x_0_sq_est_{i+1}'] = [r['phi_x_x'][0][i] for r in results]
            for j, q in enumerate(quantile_index):
                q_str = str(round(q, 2))
                results_df_dict[f'x_0_q{q_str}_est_{i+1}'] = [r['quantiles'][j][0][i] for r in results]
    return pd.DataFrame(results_df_dict)

    
quantiles = np.linspace(0.05, 0.95, num=19)
Ts = [10, 31, 100, 316, 1000]

# Ts = [10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000]
N_FFBS_MCMC=1000; 
N_genealogy=1000; num=10; nruns=100
benchmark_N = 10000

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
Part 1 run time: 178.79 seconds # 17879
Part 2 run time: 10.3 seconds # 
Part 3 run time: 283.04 seconds # 14150
Part 4 run time: 28.12 seconds
Total CPU time: 500.25 seconds
"""

# if not len(sys.argv) >= 2:
#     raise ValueError('Please provide a run_id and cdssm_str as an argument when running the script.')

# run_id = int(sys.argv[1])
# cdssm_str = str(sys.argv[2])
run_id = 10
cdssm_str = 'mv_ou'

# Build the cdssm
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, smoothing=True)

np.random.seed(cdssm_spec['seed'])
x, y = cdssm.simulate(Ts[-1], num=100)

is1d = not isinstance(cdssm, MvCDSSM)

# Pull fk_names from the cdssm_spec
fk_names = cdssm_spec['smoothing_fk_names']
filt_fk_names = [name for name in fk_names if name[2] != 'R']
smth_fk_names = [name for name in fk_names if name[2] == 'R']

# Define LGSSM (if possible) and a ddssm:
if cdssm.islgssm:
    lgssm=cdssm.lgssm()
    
ddssm = DiscreteDiscreteSSM(cdssm=cdssm)
#-----------------------------------------------------------------------------------------

# Part 1: Extract the true values from the synthetic data using the RTS smoother:
part_1_cpu = time.perf_counter()
print('Part 1: Running RTS smoothers and benchmark particle smoothers for different values of T')

true_vals_arr = np.array([1] + Ts).reshape((-1, 1)) # (len(Ts)+1, 1)
colnames = ['T']

if not cdssm.islgssm:
    print('Cannot express CDSSM as an LGSSM. Skipping Kalman filter.')
else:
    print('Running Kalman filter for each value of T:')    
    # Run the RTS smoother for the LGSSM for each value of T
    kalmans = [Kalman(ssm=lgssm, data=y[:T]) for T in Ts]
    for kalman in kalmans:
        kalman.smoother()

    # Extract the filtering means and stds
    filt_mean = kalmans[0].filt[0].mean.ravel().reshape((1, -1)) # (1, dimX)
    filt_std = np.sqrt(np.diag(kalmans[0].filt[0].cov).ravel().reshape((1, -1))) # (1, dimX)

    # Extract the smoothing means and stds
    smth_means = np.array([kalman.smth[0].mean.ravel() for kalman in kalmans]) # (len(Ts), dimX)
    smth_stds = np.sqrt(np.array([np.diag(kalman.smth[0].cov) for kalman in kalmans])) # (len(Ts), dimX) # (len(Ts), dimX)

    means = np.concatenate([filt_mean, smth_means], axis=0) # (len(Ts)+1, dimX)
    stds = np.concatenate([filt_std, smth_stds], axis=0) # (len(Ts)+1, dimX)

    true_vals_arr = np.concatenate([true_vals_arr, means, stds], axis=1) # (len(Ts)+1, 3*dimX + 1)
    colnames += [f'x_0_mean_{i+1}' for i in range(cdssm.dimX)] + [f'x_0_std_{i+1}' for i in range(cdssm.dimX)]
    print('Kalman filter runs complete.')        

if False:
    print(f'Running benchmark FFBS_MCMC with N={benchmark_N} and 5 IMH steps for each noise level:')

    # Run the Guided PF for each value of T:
    benchmark_filt_fk = ssms.GuidedPF(ssm=ddssm, data=y[:1])
    benchmark_filt_alg = SMC(fk=benchmark_filt_fk, N=benchmark_N, qmc=True, collect=[Moments])
    benchmark_filt_alg.run()

    # Extract filtering means and stds
    if is1d:
        filt_mean = np.eye(1) * benchmark_filt_alg.summaries.moments[0]['mean'] # (1, 1)
        filt_std = np.eye(1) * np.sqrt(benchmark_filt_alg.summaries.moments[0]['var']) # (1, 1)
    else:
        filt_mean = benchmark_filt_alg.summaries.moments[0]['mean'].reshape((1, -1)) # (1, dimX)
        filt_std = np.sqrt(benchmark_filt_alg.summaries.moments[0]['var'].reshape((1, -1))) # (1, dimX)


    benchmark_fks = [ssms.GuidedPF(ssm=ddssm, data=y[:T]) for T in Ts]
    smth_out = [smoothing_worker(method='FFBS_MCMC_5', N=benchmark_N, fk=fk, smc_cls=SMC) for fk in benchmark_fks]

    if is1d:
        smth_x = np.array([r['phi_x'][0] for r in smth_out]).reshape((-1, 1)) # (len(Ts), 1)
        smth_x_x = np.array([r['phi_x_x'][0] for r in smth_out]).reshape((-1, 1)) # (len(Ts), 1)
    else:
        smth_x = np.array([r['phi_x'][0] for r in smth_out]) # (len(Ts), dimX)
        smth_x_x = np.array([r['phi_x_x'][0] for r in smth_out]) # (len(Ts), dimX)

    smth_means = smth_x
    smth_stds = np.sqrt(smth_x_x - smth_x**2)

    means = np.concatenate([filt_mean, smth_means], axis=0) # (len(Ts)+1, dimX)
    stds = np.concatenate([filt_std, smth_stds], axis=0) # (len(Ts)+1, dimX)
    true_vals_arr = np.concatenate([true_vals_arr, means, stds], axis=1) # (len(Ts)+1, 3*dimX + 1)

    colnames += [f'x_0_est_pf_{i+1}' for i in range(cdssm.dimX)] + [f'x_0_est_std_pf_{i+1}' for i in range(cdssm.dimX)]

print('Benchmark particle filter runs complete.')
results_df_1 = pd.DataFrame(true_vals_arr, columns=colnames)

part_1_cpu = time.perf_counter() - part_1_cpu
print(f'Part 1 complete. Run time: {round(part_1_cpu, 2)} seconds.')

#-----------------------------------------------------------------------------------------

# Part 2: Run the smoothing methods for standard smc algorithms, using a fixed number of particles, and different values of T:
part_2_cpu = time.perf_counter()
print('Part 2: Running SMC smoothing algorithms for different values of T:')

ssm_results_dfs = [None] * len(Ts)

for i, T in enumerate(Ts):
    #fks = {'bootstrap': ssms.Bootstrap(ssm=lgssm, data=y[:T]), 'guided': ssms.GuidedPF(ssm=lgssm, data=y[:T])}
    fks = {'bootstrap': ssms.Bootstrap(ssm=lgssm, data=y[:T])}
    genealogy_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['genealogy'], N=N_genealogy, fk=fks, smc_cls=SMC, quantiles=quantiles)
    FFBS_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['FFBS_MCMC'], N=N_FFBS_MCMC, fk=fks, smc_cls=SMC, quantiles=quantiles)
    results = genealogy_results + FFBS_results
    results_df = multismooth_results_to_df(results)
    results_df['T'] = [T] * results_df.shape[0]
    ssm_results_dfs[i] = results_df

results_df_2 = pd.concat(ssm_results_dfs, axis=0, ignore_index=True)

part_2_cpu = time.perf_counter() - part_2_cpu
print(  f'Part 2 complete. Run time: {round(part_2_cpu, 2)} seconds.')

#-----------------------------------------------------------------------------------------

# Part 3: Run the smoothing methods for CD-SSMs using a fixed number of particles and imputed points, and different values of T:

print('Skipping Part 3 for now.')
# part_3_cpu = time.perf_counter()
# print('Part 3: Running CDSSM_SMC smooting algortithms for different values of T:')

# cdssm_results_dfs = [None] * len(Ts)
# for i, T in enumerate(Ts):
#     filt_fks = sfk.gen_fk_models(cdssm, y[:T], smoothing=False, fk_names=filt_fk_names) 
#     smth_fks = sfk.gen_fk_models(cdssm, y[:T], smoothing=True, fk_names=smth_fk_names)
#     genealogy_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['genealogy'], N=N, fk=filt_fks, num=num, smc_cls=CDSSM_SMC)
#     FFBS_results = multiplexer(f=smoothing_worker, nruns=nruns, nprocs=0, seeding=True, method=['FFBS_MCMC'], N=N, fk=smth_fks, num=num, smc_cls=CDSSM_SMC)
#     results = genealogy_results + FFBS_results
#     results_df = multismooth_results_to_df(results)
#     results_df['T'] = [T] * results_df.shape[0]
#     cdssm_results_dfs[i] = results_df

# results_df_3 = pd.concat(cdssm_results_dfs, axis=0, ignore_index=True)

# part_3_cpu = time.perf_counter() - part_3_cpu
# print(f'Part 3 complete. Run time: {round(part_3_cpu, 2)} seconds.')

#-----------------------------------------------------------------------------------------
# Part 4: Collect metadata
part_4_cpu = time.perf_counter()
print('Part 4: Collecting metadata and store all results')

metadata = {'Ts': Ts, 
            'N_FFBS_MCMC': N_FFBS_MCMC,
            'N_genealogy': N_genealogy,
            'quantiles': quantiles,
            'num': num,
            'x': x,
            'y': y,
            'cdssm': cdssm,
            'lgssm': lgssm,
            'nruns': nruns,
            'fk_names': filt_fk_names + smth_fk_names,
            }

# Store data from the 4 parts:
results_df_1.to_json(f'./results/smoothing_exp_run_{run_id}_{cdssm_str}_part_1.json', index=False)
results_df_2.to_json(f'./results/smoothing_exp_run_{run_id}_{cdssm_str}_part_2.json', index=False)
# results_df_3.to_json(f'./results/smoothing_exp_run_{run_id}_{cdssm_str}_part_3.json', index=False)

with open(f'./results/smoothing_exp_run_{run_id}_{cdssm_str}_meta.pkl', 'wb') as f:
    dill.dump(metadata, f)
    
part_4_cpu = time.perf_counter() - part_4_cpu
print(f'Part 4 complete. Run time: {round(part_4_cpu, 2)} seconds.')
print('Smoothing experiment complete.')

part_cpus = [part_1_cpu, part_2_cpu, part_4_cpu]
total_cpu = sum(part_cpus)

# for i, cpu_time in enumerate(part_cpus):
#     print(f'Part {i+1} run time: {round(cpu_time, 2)} seconds')

print(f'Part 1 run time: {round(part_1_cpu, 2)} seconds')
print(f'Part 2 run time: {round(part_2_cpu, 2)} seconds')
print(f'Part 4 run time: {round(part_4_cpu, 2)} seconds')

print(f'Total CPU time: {round(total_cpu, 2)} seconds')
#-----------------------------------------------------------------------------------------

# Run smoothing worker with multiplexer for all the fk models
# Note: If different add_funcs are set instead of the default choices, then 
# the add_funcs need to be passed as `protected_args`  to the multiplexer function.