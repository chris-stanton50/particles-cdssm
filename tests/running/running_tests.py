#!/usr/bin/env python3
import time
import numpy as np

from particles.kalman import MVLinearGauss, LinearGauss
from particles.utils import multiplexer
from particles.collectors import Moments
from particles_cdssm.core import SMC, summaries, smoothing_worker

import particles_cdssm
import particles_cdssm.feynman_kac as sfk
import particles.state_space_models as ssms
import particles_cdssm.collectors as cols
from particles_cdssm.tools import build_cdssm

from cdssm_lib import CDSSM_LIB

T = 10; N=100; num=10; M=5
Ns = [10, 20, 30]; nums = [10, 20, 30]; 

cdssm_str = 'ou'

# Build the cdssm
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_spec)

# Simulate synthetic data from the cdssm
x, y = cdssm.simulate(T)

def cdssm_smc():
    # Generate all possible fk models for the given cdssm
    all_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)

    # Run the CDSSM_SMC algorithm for each fk model
    algs = {}
    for fk_name, fk in all_fks.items():
        algs[fk_name] = particles_cdssm.CDSSM_SMC(fk=fk, N=N, num=10, store_history=True)
        
    for fk_name, smc in algs.items():
        print(f'Running CDSSM_SMC for {fk_name}')
        smc.run()

def multi_cdssm_smc():        
    # Generate all possible fk models for the given cdssm
    all_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)

    # `collect` is a protected arg, because it is expected that we pass a list of 
    # collectors to the `CDSSM_SMC` class
    # If no out_func is specified, each indpendent pf run will store the full SMC object.
    # An out_func can instead be passed that ensures that only the required output from the pf is stored.
    out = particles_cdssm.multiCDSSM_SMC(nruns=2, nprocs=0, out_func=summaries, collect=[Moments], fk=all_fks, N=Ns, num=nums)

def cdssm_particle_history_test():
    # Generate all possible fk models for the given cdssm
    all_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)
    all_smth_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=True)

    # Run backward_sampling_genealogy for all filtering fks
    filt_algs = {}
    for fk_name, fk in all_fks.items():
        filt_algs[fk_name] = particles_cdssm.CDSSM_SMC(fk=fk, N=100, store_history=True)
        
    for fk_name, smc in filt_algs.items():
        smc.run()
        print(f'Running backward_sampling_genealogy for {fk_name}')
        samples = smc.hist.backward_sampling_genealogy(M)

    # Run `backward_sampling_` + genealogy, ON2 and mcmc for all smoothing fks
    smth_algs = {}
    for fk_name, fk in all_smth_fks.items():
        smth_algs[fk_name] = particles_cdssm.CDSSM_SMC(fk=fk, N=100, num=10, store_history=True)

    for fk_name, smc in smth_algs.items():
        smc.run()
        print(f'Running backward_sampling_genealogy for {fk_name}')
        samples_g = smc.hist.backward_sampling_genealogy(M)
        print(f'Running backward_sampling_ON2 for {fk_name}')
        samples_o = smc.hist.backward_sampling_ON2(M)
        print(f'Running backward_sampling_mcmc for {fk_name}')
        samples_m = smc.hist.backward_sampling_mcmc(M)
        
def particle_history_test():
    # Generate examples of fks based on all SSMs defined in the particles package, with default choice of parameters:
    bootstrap_ssm_list = [ssms.StochVol(), ssms.StochVolLeverage(), ssms.Gordon_etal(), ssms.BearingsOnly(), ssms.DiscreteCox(), ssms.ThetaLogistic()]
    guided_ssm_list = [MVLinearGauss(covX=np.eye(2), covY=np.eye(2)), LinearGauss()]

    all_fks = {}

    for ssm in bootstrap_ssm_list:
        x, y = ssm.simulate(T)
        all_fks[ssm.__class__.__name__] = ssms.Bootstrap(ssm=ssm, data=y)

    for ssm in guided_ssm_list:
        x, y = ssm.simulate(T)
        all_fks[ssm.__class__.__name__] = ssms.GuidedPF(ssm=ssm, data=y)

    # Run backward_sampling_genealogy for all fks
    filt_algs = {}
    for fk_name, fk in all_fks.items():
        filt_algs[fk_name] = SMC(fk=fk, N=100, store_history=True)

    for fk_name, smc in filt_algs.items():
        smc.run()
        print(f'Running backward_sampling_genealogy for {fk_name}')
        samples = smc.hist.backward_sampling_genealogy(M)
        
def cdssm_smoothing_worker_test():
    Ns=[5, 10, 20]; T = 5; nums = [10, 20]
    smth_methods = ['FFBS_ON2', 'FFBS_MCMC', 'genealogy']

    # Generate all possible fk models for the given cdssm
    all_smth_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=True)
        
    # Run smoothing worker with multiplexer for all the fk models
    # Note: If different add_funcs are set instead of the default choices, then 
    # the add_funcs need to be passed as `protected_args`  to the multiplexer function.
    cdssm_output = multiplexer(f=smoothing_worker, nruns=3, nprocs=0, seeding=True, method=smth_methods, N=Ns, fk=all_smth_fks, num=nums, smc_cls=particles_cdssm.CDSSM_SMC)

def smc_predictive_collector_test():
    col_classes = [cols.PredictiveParticles, cols.ObservationPredictiveParticles, cols.PredictiveMoments, cols.ObservationPredictiveMoments, cols.NLPD, cols.LowVarianceNLPD, cols.AbsoluteError]
    col_kwargs_list = [{'K': 1, 'method': 'simulated'}]*len(col_classes)

    collectors = [col_class(**col_kwargs) for col_class, col_kwargs in zip(col_classes, col_kwargs_list)]

    # Generate examples of fks based on all SSMs defined in the particles package, with default choice of parameters:
    bootstrap_ssm_list = [ssms.StochVol(), ssms.StochVolLeverage(), ssms.Gordon_etal(), ssms.BearingsOnly(), ssms.DiscreteCox(), ssms.ThetaLogistic(), MVLinearGauss(covX=np.eye(2), covY=np.eye(2)), LinearGauss()]
    guided_ssm_list = [MVLinearGauss(covX=np.eye(2), covY=np.eye(2)), LinearGauss()]

    all_fks = {}

    for ssm in bootstrap_ssm_list:
        x, y = ssm.simulate(T)
        all_fks[ssm.__class__.__name__] = ssms.Bootstrap(ssm=ssm, data=y)

    for ssm in guided_ssm_list:
        x, y = ssm.simulate(T)
        all_fks[ssm.__class__.__name__] = ssms.GuidedPF(ssm=ssm, data=y)

    for ssm_name, fk in all_fks.items():
        fk_name = fk.__class__.__name__
        for col in collectors:
            col_name = col.__class__.__name__
            smc = SMC(fk=fk, N=N, store_history=False, collect=[col])
            print(f"Running SMC for Model: {ssm_name} FK: {fk_name}, with collector {col_name}")
            smc.run()

def cdssm_smc_predictive_collector_test():
    col_classes = [cols.PredictiveParticles, cols.ObservationPredictiveParticles, cols.PredictiveMoments, cols.ObservationPredictiveMoments, cols.NLPD, cols.LowVarianceNLPD, cols.AbsoluteError]
    col_kwargs_list = [{'K': 1, 'method': 'simulated'}]*len(col_classes)

    collectors = [col_class(**col_kwargs) for col_class, col_kwargs in zip(col_classes, col_kwargs_list)]

    # Generate all possible fk models for the given cdssm
    all_cdssm_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)

    for fk_name, fk in all_cdssm_fks.items():
        for col in collectors:
            smc = particles_cdssm.CDSSM_SMC(fk=fk, N=N, store_history=False, collect=[col], num=num)
            print(f"Running CDSSM_SMC for Model:{cdssm_str} FK: {fk_name} with Collector:{col.__class__.__name__}")
            smc.run()
            
# Process the output as needed
if __name__ == "__main__":
    print('Running tests for standard SSMs:')
    time.sleep(5)
    particle_history_test()
    smc_predictive_collector_test()

    print('Running tests for CD-SSMs:')

    cdssms = ['ou', 'mv_ou', 'iou']
    cdssm_tests = [cdssm_smc, multi_cdssm_smc, cdssm_particle_history_test, cdssm_smoothing_worker_test, cdssm_smc_predictive_collector_test]
    for cdssm_str in cdssms:
        # Build the cdssm
        cdssm_spec = CDSSM_LIB[cdssm_str]
        cdssm = build_cdssm(cdssm_spec)

        # Simulate synthetic data from the cdssm
        x, y = cdssm.simulate(T)
        for test in cdssm_tests:
            print(f'Running test: {test.__name__} for CDSSM: {cdssm_str}')
            time.sleep(5)
            test()
    
    print('All tests completed.')