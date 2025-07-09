import dill
import sys
import numpy as np
import pandas as pd

# from particles_cdssm.cdssm_lib import CDSSM_LIB
from cdssm_lib import CDSSM_LIB

"""
Use this script to quickly check the results of the filtering tests, on each different CDSSM:

Please supply the run_id as a command line argument.
The run_id is the same as the one used in the filtering_test.py script.

e.g 
```
python filtering_test_results.py 10
```

Test results will be printed to the console. If one of the tests fails, an AssertionError will be raised,
so the script will stop running.
"""

run_id = sys.argv[1]
cdssm_strs = ['ou', 'mv_ou', 'iou', 'bm', 'mv_bm', 'ibm']

def build_error_dfs(true_vals_df, results_df, dimX):
    """
    Creates dfs for the sq errors and the abs errors of the estimators of the following quantities:
    - LogLt    
    For i in 1, 2, ... dimX:
        - X_T_i | Y_{1:T}
        - X_T_var_i | Y_{1:T}
        - X_T_std_i | Y_{1:T}
    """
    colnames = ['logLt'] + [f'x_T_{d+1}' for d in range(dimX)] + [f'x_T_var_{d+1}' for d in range(dimX)] + [f'x_T_std_{d+1}' for d in range(dimX)]

    for colname in colnames:
        results_df[colname + '_sq_err'] = np.square(results_df[colname + '_est'] - true_vals_df.loc[0, colname])
        results_df[colname + '_abs_err'] = np.abs(results_df[colname + '_est'] - true_vals_df.loc[0, colname])
        
    abs_err_df = results_df[['fk'] + [col for col in results_df.columns.tolist() if 'abs_err' in col]].copy()
    sq_err_df = results_df[['fk'] + [col for col in results_df.columns.tolist() if 'sq_err' in col]].copy()
    cpu_df = results_df[['fk'] + ['cpu']].copy()

    return cpu_df, abs_err_df, sq_err_df

def test_mae_order(mae_order, cdssm_str):
    """
    Test whether performance of CDSSM_SMC estimators of log(p(y_{1:T}))
    are as expected.
    """
    cdssm_spec = CDSSM_LIB[cdssm_str]
    n_b = cdssm_spec['n_bottom_fks']; n_t = cdssm_spec['n_top_fks']
    # Check if the bottom n_b methods are all bootstrap methods
    print(f'Bottom {n_b} fk models:', mae_order[:n_b])
    cond = lambda x: x.startswith('BsR') or x.startswith('Bootstrap')
    assert np.all([cond(mae_str) for mae_str in mae_order[:n_b]]), f'Bottom {n_b} methods for cdssm {cdssm_str} are not all bootstrap methods.'

    # Check if the last n_t methods all use the true diffusion bridge
    print(f'Top {n_t} fk models:', mae_order[-n_t:])
    if cdssm_str.endswith('ou'):
        cond = lambda x: x.startswith('Guided') or x.split('_')[1].endswith('OU')
    else:
        cond = lambda x: x.startswith('Guided') or (not x.split('_')[1].endswith('OU'))
    assert np.all([cond(mae_str) for mae_str in mae_order[-n_t:]]), f'Top {n_t} methods for cdssm {cdssm_str} do not all use the true diffusion bridge.'

    # Check if the last 3 methods are all optimal guided methods ( ou only)
    if cdssm_str == 'ou':
        print('Top 3 fk models:', mae_order[-3:])
        cond = lambda x: x.startswith('Guided') or (x.split('_')[1] == 'OU' and x.split('_')[2] == 'OUP')
        assert np.all([cond(mae_str) for mae_str in mae_order[-3:]])
    print(f'All tests passed. MAE order is as expected for cdssm {cdssm_str}.')

for cdssm_str in cdssm_strs:
    df_part_1 = pd.read_json(f'./results/filtering_test_run_{run_id}_{cdssm_str}_part_1.json')
    df_part_2 = pd.read_json(f'./results/filtering_test_run_{run_id}_{cdssm_str}_part_2.json')
    df_part_3 = pd.read_json(f'./results/filtering_test_run_{run_id}_{cdssm_str}_part_3.json')

    with open(f'./results/filtering_test_run_{run_id}_{cdssm_str}_meta.pkl', 'rb') as f:
        metadata = dill.load(f)
    
    cdssm = metadata['cdssm']
    
    true_vals_df = df_part_1.copy()
    results_df = pd.concat([df_part_2, df_part_3], axis=0)
    cpu_df, abs_err_df, sq_err_df = build_error_dfs(true_vals_df, results_df, cdssm.dimX)
    mae_df = abs_err_df.groupby('fk').mean()
    mse_df = sq_err_df.groupby('fk').mean()

    mae_order = mae_df.sort_values(by='logLt_abs_err', ascending=False).index.tolist()

    # Currently tests the OU process, but can be changed to test other processes. 
    test_mae_order(mae_order, cdssm_str)

print('Filtering tests passed.')