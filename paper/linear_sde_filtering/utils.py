import numpy as np
import pandas as pd

def obs_times_to_store(T):
    """
    Returns an array of 10 observation times and preprends time 1, giving a total of
    11 times.
    """
    Ts_to_store_excl_1 = np.linspace(T // 10, T, num=10, dtype=np.int64)
    arr_1 = np.array([1], dtype=np.int64)
    obs_times_to_store = np.concatenate([arr_1] + [Ts_to_store_excl_1], axis=0)
    obs_times_to_store = obs_times_to_store - 1 # subtract 1 from all times
    return obs_times_to_store

def kalman_results_to_frame(kalman):
    """
    Converts the output of the kalman filter to a pd.DataFrame, containing the true value quantities 
    of interest for the experiment.
    """
    T = len(kalman.data)
    results_df_dict = {}
    obs_times = obs_times_to_store(T).tolist()
    results_df_dict['t'] = obs_times
    filt = kalman.filt
    dimX = kalman.filt[0].mean.ravel().shape[0]
    for d in range(dimX):
        results_df_dict[f'x_t_{d+1}'] = [filt[t-1].mean.ravel()[d] for t in obs_times]
        results_df_dict[f'x_t_var_{d+1}'] = [np.diag(filt[t-1].cov)[d] for t in obs_times]
        results_df_dict[f'x_t_std_{d+1}'] = [np.sqrt(np.diag(filt[t-1].cov))[d] for t in obs_times]
    results_df_dict['logLt'] = [np.concatenate([kalman.logpyt]).ravel().cumsum()[t] for t in obs_times]
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
    example_mean = results[0]['output'].moments[-1]['mean']
    T = len(results[0]['output'].moments)
    obs_times = obs_times_to_store(T).tolist()
    for t in obs_times:         
        results_df_dict[f'logLt_{t}_est'] = [r['output'].logLts[t] for r in results]
        if isinstance(example_mean, float):
            results_df_dict[f'x_{t}_1_est'] = [r['output'].moments[t]['mean'] for r in results]
            results_df_dict[f'x_{t}_var_1_est'] = [r['output'].moments[t]['var'] for r in results]
            results_df_dict[f'x_{t}_std_1_est'] = [np.sqrt(r['output'].moments[t]['var']) for r in results]
        else:
            for i in range(len(example_mean)):
                results_df_dict[f'x_{t}_{i+1}_est'] = [r['output'].moments[t]['mean'][i] for r in results]
                results_df_dict[f'x_{t}_var_{i+1}_est'] = [r['output'].moments[t]['var'][i]for r in results]
                results_df_dict[f'x_{t}_std_{i+1}_est'] = [np.sqrt(r['output'].moments[t]['var'][i]) for r in results]
    return pd.DataFrame(results_df_dict)