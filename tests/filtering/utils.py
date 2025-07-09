import numpy as np
import pandas as pd

def kalman_results_to_frame(kalman):
    """
    Converts the output of the kalman filter to a pd.DataFrame, containing the true value quantities 
    of interest for the experiment.
    """
    T = len(kalman.data)
    results_df_dict = {}
    obs_times = [T]
    results_df_dict['T'] = obs_times
    filt = kalman.filt
    dimX = kalman.filt[0].mean.ravel().shape[0]
    for d in range(dimX):
        results_df_dict[f'x_T_{d+1}'] = [filt[t-1].mean.ravel()[d] for t in obs_times]
        results_df_dict[f'x_T_var_{d+1}'] = [np.diag(filt[t-1].cov)[d] for t in obs_times]
        results_df_dict[f'x_T_std_{d+1}'] = [np.sqrt(np.diag(filt[t-1].cov))[d] for t in obs_times]
        results_df_dict['logLt'] = [np.concatenate([kalman.logpyt]).ravel().cumsum()[-1]]
    results_df = pd.DataFrame(results_df_dict)
    return results_df

def multismc_results_to_df(results, continuous_discrete=False):
    """
    Converts the output of the multiSMC/multiCDSSM_SMC algorithm to a pd.DataFrame, 
    containing the quantities of interest for the experiment.
    """
    results_df_dict = {} 
    results_df_dict['run_id'] = [r['run'] for r in results]
    results_df_dict['seed'] = [r['seed'] for r in results]
    results_df_dict['fk'] = [r['fk'] for r in results]
    results_df_dict['cpu'] = [r['output'].cpu_time for r in results]
    results_df_dict['logLt_est'] = [r['output'].logLts[-1] for r in results]
    example_mean = results[0]['output'].moments[-1]['mean']
    if isinstance(example_mean, float):
        results_df_dict['x_T_1_est'] = [r['output'].moments[-1]['mean'] for r in results]
        results_df_dict['x_T_var_1_est'] = [r['output'].moments[-1]['var'] for r in results]
        results_df_dict['x_T_std_1_est'] = [np.sqrt(r['output'].moments[-1]['var']) for r in results]
    else:
        for i in range(len(example_mean)):
            results_df_dict[f'x_T_{i+1}_est'] = [r['output'].moments[-1]['mean'][i] for r in results]
            results_df_dict[f'x_T_var_{i+1}_est'] = [r['output'].moments[-1]['var'][i]for r in results]
            results_df_dict[f'x_T_std_{i+1}_est'] = [np.sqrt(r['output'].moments[-1]['var'][i]) for r in results]
    return pd.DataFrame(results_df_dict)

