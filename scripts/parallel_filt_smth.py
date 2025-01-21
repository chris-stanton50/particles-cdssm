import time
import sys
import dill
import os

import numpy as np
import numpy.linalg as nla

from particles.core import SMC, multiSMC
import particles.state_space_models as ssms
from particles.utils import multiplexer
from particles.kalman import MVLinearGauss, Kalman
from particles.collectors import Moments
import particles.state_space_models as ssms

from sdes.cdssm_lib import CDSSM_LIB, build_cdssm
import sdes.sdes as sdes
from sdes.smoothing import modif_smoothing_worker
import sdes.feynman_kac as sfk
from sdes.tools import timed_func
from sdes.collectors import default_add_funcs, MultiOnline_smooth_naive, MultiOnline_smooth_mcmc

# Process command line arguments

# python parallel_filt_smth.py -local mv_ou -f -live
assert sys.argv[1] in ['-local', '-remote'], "Please specify whether the script is being run locally or remotely"
remote = True if sys.argv[1] == '-remote' else False

assert sys.argv[2] in list(CDSSM_LIB.keys()), "Please specify an existing CDSSM spec as the first cli"
cdssm_spec_name = sys.argv[2]

assert sys.argv[3] in ['-os', '-s', '-f'], "Please specify whether you want to run filtering, smoothing or online-smoothing"
smoothing = True if sys.argv[3] in ['-s', '-os'] else False

assert sys.argv[4] in ['-test', '-live'], "Please specify whether you want to run a test or a live run"
test = True if sys.argv[4] == '-test' else False

# If set to None, takes T from the cdssm_lib. 
T_override = int(sys.argv[5]) if len(sys.argv) > 5 else None

short_objective = sys.argv[3]
objectives_map = {'-f': 'filtering', '-s': 'smoothing', '-os': 'online_smoothing'}
objective = objectives_map[short_objective]

# cdssm_spec_name = 'iou'
# short_objective = '-f'
# objective = 'filtering'
# smoothing=False

# File storage index
i = 20

# Always use 0 when running
i = 0 if test else i
if remote:
    i += 100
# -----------------------------------------------------------------------------------------
# Algorithm Parameters
# -----------------------------------------------------------------------------------------

# ------------------Benchmark parameters------------------

# FK used for both filtering and smoothing (ssm and data is passed to the FK)
bench_ssm_fk_cls = ssms.GuidedPF

# Benchmark DiscreteDiscrete SSM parameters (used when model sde has known transition density)
bench_N_ssm_filt = 10 if test else 1000000
qmc = True

bench_N_ssm_smth = 10 if test else 1000000
# bench_N_ssm_smth = 1000000 # Saved as a separate variable in case we want to use an O(N^2) smoothing algorithm
bench_smth_methods = ['FFBS_MCMC']

# Benchmark CDSSM_SMC parameters (used when model sde has unknown transition density)

# CDSSM_FK used for both filtering and smoothing
bench_cdssm_fk_cls = sfk.ForwardReparameterisedDA
cdssm_fk_kwargs = {'auxiliary_bridge_cls': sfk.MvDelyonHuAuxBridge, 'proposal_sde_cls': sfk.MvOUProposal}

bench_N_cdssm_filt = 1000000
bench_num_filt = 10 # Reduce bias effect on benmark with large number of imputed points.

bench_N_cdssm_smth = 1000000 # Saved as a separate variable in case we want to use an O(N^2) smoothing algorithm
bench_num_smth = 10 # Reduce bias effect on benchmark with large number of imputed points.
bench_cdssm_smth_methods = ['FFBS_MCMC']

#---------------------Multi SMC/CDSSM SMC parameters-------------

# these are used both for filtering and online-smoothing
N_filt=[5] if test else [100]
num_filt=[10, 31, 100, 316, 1000]
nruns_filt=1 if test else 100

# (Offline) Smoothing parameters
N_smth=[5] if test else [100]
num_smth=[10]
nruns_smth=1 if test else 100

methods = ['geneaology', 'FFBS_MCMC']

# Additive functions to store and methods to use for online smoothing
add_funcs = default_add_funcs
# online_smoothing_collectors = [MultiOnline_smooth_naive, MultiOnline_smooth_ON2, MultiOnline_smooth_mcmc]
online_smoothing_collectors = [MultiOnline_smooth_naive, MultiOnline_smooth_mcmc]

# If set to True, then overrides manual setting of FK models in the cdssm_lib, and runs all possible FK models.
fk_names_override = False

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Benchmark CDSSM SMC Filtering Parameters (excl FK model)
bench_multi_smc_opts = {'out_func': sfk.summaries, 'nruns': 1, 'nprocs': 1, 'collect': [Moments], 'N': [bench_N_ssm_filt], 'qmc': [qmc], 'store_history': False}
bench_multi_cdssm_smc_opts = {'out_func': sfk.summaries, 'nruns': 1, 'nprocs': 1, 'collect': [Moments], 'N': [bench_N_cdssm_filt], 'store_history': False, 'num': [bench_num_filt]}

# Multi SMC/CDSSM SMC Smoothing Parameters
bench_multiplexer_options = {'f': modif_smoothing_worker, 'nruns': 1, 'nprocs': 1, 
                       'protected_args': {'add_funcs': add_funcs}, 'method': bench_smth_methods, 'N': [bench_N_ssm_smth], 'smc_cls': SMC}
bench_multi_cdssm_smoothing_options = {'f': modif_smoothing_worker, 'nruns': 1, 'nprocs': 1, 
                       'protected_args': {'add_funcs': add_funcs}, 'method': bench_cdssm_smth_methods, 'N': [bench_N_ssm_smth], 'num': [bench_num_smth], 'smc_cls': sfk.CDSSM_SMC}

# Multi SMC/CDSSM SMC Filtering Parameters (excl FK models)
multi_smc_options = {'out_func': sfk.summaries, 'nruns': nruns_filt, 'nprocs': 1, 'collect': [Moments], 'N': N_filt, 'qmc': [False], 'store_history': False}
multi_cdssm_smc_options = {'out_func': sfk.summaries, 'nruns': nruns_filt, 'nprocs': 1, 'collect': [Moments], 'N': N_filt, 'store_history': False, 'num': num_filt}

# Multi SMC/CDSSM SMC Smoothing Parameters
multiplexer_options = {'f': modif_smoothing_worker, 'nruns': nruns_smth, 'nprocs': 1,
                       'protected_args': {'add_funcs': add_funcs}, 'method': methods, 'N': N_smth, 'smc_cls': SMC}
multi_cdssm_smoothing_options = {'f': modif_smoothing_worker, 'nruns': nruns_smth, 'nprocs': 1, 
                       'protected_args': {'add_funcs': add_funcs}, 'method': methods, 'N': N_smth, 'num': num_smth, 'smc_cls': sfk.CDSSM_SMC}

#------------------------------------------------------------------------------------------------------------
#------------------------------FUNCTIONS USED TO RUN THE SCRIPT----------------------------------------------

def run_benchmark():
    if discrete_ssm is None:
        benchmark = run_smc(cd=True, benchmark=True)
    else:
        benchmark = run_smc(cd=False, benchmark=True)
    return benchmark     

def run_kalman_benchmark():
    print('Running benchmark Kalman filter')
    benchmark_kalman = Kalman(ssm=discrete_ssm, data=y)
    try:
        _, filt_time = timed_func(benchmark_kalman.filter)()
    except ValueError:
        print('Kalman filter failed. Returning None.')
        return None
    print(f'Done. Kalman filter run time: {round(filt_time, 6)}')
    print(f'Running benchmark RTS smoother')
    _, smth_time = timed_func(benchmark_kalman.smoother)()
    print(f'Done. Benchmark smoother run time: {round(smth_time, 6)}')
    return benchmark_kalman

def get_run_msgs(cd=False, benchmark=False):
    benchmark = 'benchmark ' if benchmark else ''
    cdssm_str = 'CDSSM ' if cd else ''
    return f'Running {benchmark + cdssm_str}SMC {objective} algorithms in parallel:', f'Done. {benchmark + cdssm_str }SMC {objective} run time: '
    
def get_multi_func(cd=False, benchmark=False):
    if not cd:
        if not benchmark:
            multiSMC_func = multiSMC if objective != 'smoothing' else multiplexer
        else:
            multiSMC_func = multiSMC if objective == 'filtering' else multiplexer
    else: 
        if not benchmark:
            multiSMC_func = sfk.multiCDSSM_SMC if objective != 'smoothing' else multiplexer
        else:
            multiSMC_func = sfk.multiCDSSM_SMC if objective == 'filtering' else multiplexer
    return timed_func(multiSMC_func)

def get_smc_options(cd=False, benchmark=False):
    """
    Generate dictionary to be passed the mutlt func. 
    Includes the construction of the fk models to be passed to the SMC algorithms.
    """
    if not benchmark and not cd:
        smc_options = multi_smc_options if objective != 'smoothing' else multiplexer_options
        fk_models = {'bootstrap': ssms.Bootstrap(ssm=discrete_ssm, data=y), 'guided': ssms.GuidedPF(ssm=discrete_ssm, data=y)}
        for fk_model in fk_models.values():
            fk_model.add_funcs = default_add_funcs
    elif not benchmark and cd:
        smc_options = multi_cdssm_smc_options if objective != 'smoothing' else multi_cdssm_smoothing_options
        fk_models = sfk.gen_fk_models(cdssm, y, smoothing=smoothing, fk_names=fk_names)
    elif benchmark and not cd:
        smc_options = bench_multi_smc_opts if objective=='filtering' else bench_multiplexer_options
        fk_models = {bench_ssm_fk_cls.__name__: bench_ssm_fk_cls(ssm=discrete_ssm, data=y)}
        fk_models[bench_ssm_fk_cls.__name__].add_funcs = default_add_funcs
    else:
        smc_options = bench_multi_cdssm_smc_opts if objective=='filtering' else bench_multi_cdssm_smoothing_options
        fk_models = {bench_cdssm_fk_cls.__name__: bench_cdssm_fk_cls(cdssm=cdssm, data=y, **cdssm_fk_kwargs)}
    smc_options['fk'] = fk_models 
    return smc_options
    
def run_smc(cd=False, benchmark=False):
    multiSMC_func = get_multi_func(cd=cd, benchmark=benchmark)
    start_msg, end_msg = get_run_msgs(cd=cd, benchmark=benchmark)
    smc_options = get_smc_options(cd=cd, benchmark=benchmark)
    print(start_msg)
    if objective=='online_smoothing' and not benchmark:
        SMC_results = []; run_time = 0
        for col in online_smoothing_collectors:
            print(start_msg + f' collector {col.__name__}')
            smc_options['collect'] = [Moments, col]; 
            col_SMC_results, col_run_time = multiSMC_func(**smc_options)
            col_SMC_results = [{**res, 'col': col.__name__} for res in col_SMC_results]
            run_time += col_run_time; SMC_results += col_SMC_results
            print(end_msg + f'collector {col.__name__} ' + f'{round(col_run_time, 5)}')
    elif objective=='online_smoothing' and benchmark: # An offline smoother is the benchmark for online smoothing.
        SMC_results, run_time = multiSMC_func(**smc_options)
        for result in SMC_results:
            for add_func_name in add_funcs.keys(): 
                # Modify the output so that the additive functions are cumulative sums.
                result['ests'][add_func_name] = result['ests'][add_func_name].cumsum(axis=0)
    else:        
        SMC_results, run_time = multiSMC_func(**smc_options)
    print(end_msg + f'{round(run_time, 5)}')
    return SMC_results, run_time

def store_results():

    # File path where we will store the pickled object
    N = N_filt if objective != 'smoothing' else N_smth
    num = num_filt if objective != 'smoothing' else num_smth
    nruns = nruns_filt if objective != 'smoothing' else nruns_smth
    filename = f'{short_objective[1:]}_N={N}_T={T}_num={num}_n_runs={nruns}_{i}.pkl'
    os.chdir(f'./results/{cdssm_spec_name}/')

    # Pickle (serialize) the object using dill
    t1 = time.perf_counter()
    with open(filename, 'wb') as f:
        dill.dump(out, f)
    storage_time = time.perf_counter() - t1
    print(f"Object has been pickled to {filename}. \n Storage time: {storage_time}")

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

# Build the CDSSM from the spec:
cdssm = build_cdssm(cdssm_spec_name)

# Take the fk_names from the spec:
fk_names = CDSSM_LIB[cdssm_spec_name]['fk_names'] if not fk_names_override else None
T = CDSSM_LIB[cdssm_spec_name]['default_T'] if not T_override else T_override 

# Generate some synthetic data from the process:
np.random.seed(CDSSM_LIB[cdssm_spec_name]['seed'])
x, y = cdssm.simulate(T)

# Extract the ending points: the main target for inference:
et_s = np.concatenate([x_path[x_path.dtype.names[-1]] for x_path in x], axis=0).T # (dimX, T)

# Assuming that the transition density of the SDE is known, we can construct a DiscreteDiscrete SSM:
discrete_ssm = cdssm.discrete_ssm()

if __name__ == '__main__':
    kalman = run_kalman_benchmark() if discrete_ssm and isinstance(discrete_ssm, MVLinearGauss) else None # If possible, run Kalman.
    benchmark, benchmark_run_time = run_benchmark()
    if discrete_ssm:
        SMC_results, standard_run_time = run_smc(cd=False)
    CDSSM_SMC_results, cdssm_run_time = run_smc(cd=True)

    out = {'multiCDSSM_SMC_results': CDSSM_SMC_results,
            'multiSMC_results': SMC_results, 
            'cdssm': cdssm,
            'y': y,
            'x': x,
            'benchmark_results': benchmark,
            'kalman': kalman,
            'run_time': cdssm_run_time,
            'objective': objective
            }
    store_results()