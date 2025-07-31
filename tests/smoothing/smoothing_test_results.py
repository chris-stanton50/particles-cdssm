import dill
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from particles.kalman import MVLinearGauss, LinearGauss
import particles.state_space_models as ssms

from particles_cdssm.core import CDSSM_SMC, SMC
import particles_cdssm.plot as splt
import particles_cdssm.feynman_kac as sfk
from particles_cdssm.cdssm_lib import CDSSM_LIB, build_cdssm
from particles_cdssm.continuous_discrete_ssms import NormalCDSSM, MvNormalCDSSM
import particles_cdssm.auxiliary_bridges as axb

# This script is used to generate the results of the smoothing test
run_id = '10'
cdssm_strs = ['ou', 'mv_ou', 'iou']
signif_dist_tol = 1.0 # The filtering mean must be at least `signif_dist_tol` std devs away from the smoothing mean to be considered significant

for cdssm_str in cdssm_strs:
    df_part_1 = pd.read_json(f'./results/smoothing_test_run_{run_id}_{cdssm_str}_part_1.json')
    df_part_2 = pd.read_json(f'./results/smoothing_test_run_{run_id}_{cdssm_str}_part_2.json')
    df_part_3 = pd.read_json(f'./results/smoothing_test_run_{run_id}_{cdssm_str}_part_3.json')

    with open(f'./results/smoothing_test_run_{run_id}_{cdssm_str}_meta.pkl', 'rb') as f:
        metadata = dill.load(f)
        
    true_vals_df = df_part_1.copy()
    results_df = pd.concat([df_part_2, df_part_3], axis=0)

    # # Extract the metadata
    cdssm = metadata['cdssm']
    lgssm = metadata['lgssm']
    x = metadata['x']
    y = metadata['y']
    T = len(y)
    obs_times = metadata['stored_obs_times']
    N_genealogy = metadata['N_genealogy']
    N_FFBS_MCMC = metadata['N_FFBS_MCMC']

    ext = '' if cdssm.islgssm else '_pf'

    print(f'Model: {cdssm_str}, T: {T}, N_genealogy: {N_genealogy}, N_FFBS_MCMC: {N_FFBS_MCMC}')

    for d in range(1, cdssm.dimX + 1):
        true_vals_df[f'x_t_high_diff_{d}'] = np.abs(true_vals_df[f'x_t_smth_mean_{d}'] - true_vals_df[f'x_t_filt_mean_{d}']) > 1.5 * true_vals_df[f'x_t_smth_std_{d}']
        
    n_signif_dists = np.sum([np.sum(true_vals_df[f'x_t_high_diff_{d}']) for d in range(1, cdssm.dimX + 1)])

    if n_signif_dists > 0:
        print(f'Significant distributions found: {n_signif_dists}')
    else:
        raise ValueError('No significant distributions found. Try a different model parameterisation.')

    for d in range(1, cdssm.dimX + 1):
        for t in obs_times:
            if true_vals_df[f'x_t_high_diff_{d}'][t]:
                print(f'Testing significant dist. at t={t}, d={d}')
                # print(f'  x_t_smth_mean: {true_vals_df[f"x_t_smth_mean_{d}"][t]}')
                # print(f'  x_t_filt_mean: {true_vals_df[f"x_t_filt_mean_{d}"][t]}')
                # print(f'  x_t_smth_std: {true_vals_df[f"x_t_smth_std_{d}"][t]}')
                # print(f'  x_t_filt_std: {true_vals_df[f"x_t_filt_std_{d}"][t]}')
                abs_err = np.abs(results_df[f'x_{t}_est_{d}'] - true_vals_df.loc[t, f'x_t_smth_mean_{d}'])
                abs_err_filt = np.abs(results_df[f'x_{t}_est_{d}'] - true_vals_df.loc[t, f'x_t_filt_mean_{d}'])
                if np.all(abs_err < abs_err_filt):
                    print(f'Test passed: all algorithms target the smoothing distribution.')
                else:
                    print(f'Test failed: some algorithms may not target the smoothing distribution.')
                    failed_fks = results_df['fk'][~(abs_err < abs_err_filt)].tolist()
                    print(f'Failed fks: {failed_fks}')