import argparse
import json
from pathlib import Path
from types import MethodType
import inspect
from time import perf_counter

import numpy as np
import pandas as pd

import particles
import particles.state_space_models as ssms
from particles.kalman import Kalman

from particles_cdssm.cdssm_lib import CDSSM_LIB
import settings.add_funcs as add_funcs

from particles.collectors import Online_smooth_naive, Online_smooth_ON2
from particles_cdssm.feynman_kac import gen_all_fk_models
from particles_cdssm.collectors import Online_smooth_MCMC
from particles_cdssm import multiCDSSM_SMC
from particles_cdssm.tools import build_cdssm, isremote

debug = False

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "-c",
                        "--config",
                        type=Path,
                        required=True,
                        help="Name of the JSON configuration file",
                        )
    args = parser.parse_args()

    config_path = Path("config") / args.config

    with config_path.open("r") as f:
        config = json.load(f)
    return args.config, config

def load_config_debug():
    config_path = "./paper/logistic_sde_o_smoothing/config/first_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)
    return "first_config.json", config
        
def load_add_funcs():
    excluded = {
        add_funcs.AddFuncStorageBase,
        add_funcs.AddFuncStorage,
        add_funcs.MvAddFuncStorage,
    }

    return {
        cls.__name__: cls
        for _, cls in inspect.getmembers(add_funcs, inspect.isclass)
        if cls.__module__ == add_funcs.__name__
        and issubclass(cls, add_funcs.AddFuncStorageBase)
        and cls not in excluded
    }
    

def gen_all_ssm_fk_models(ssm: ssms.StateSpaceModel, data):
    ssm_fk_models = {'Bootstrap': ssms.Bootstrap(ssm=ssm, data=data), 
                    'GuidedPF': ssms.GuidedPF(ssm=ssm, data=data),
                    'AuxiliaryPF': ssms.GuidedPF(ssm=ssm, data=data),
                    'AuxiliaryBootstrap': ssms.AuxiliaryBootstrap(ssm=ssm, data=data)
                    }
    return ssm_fk_models

def build_collector(algo_name: str):
    if algo_name == 'naive':
        return Online_smooth_naive()
    if algo_name == 'ON2':
        return Online_smooth_ON2()
    if algo_name.startswith('MCMC'):
        nsteps = int(algo_name.split('_')[-1])
        return Online_smooth_MCMC(nsteps=nsteps)

def gen_out_func(algo_name: str, Ts: list):    
    def out_func(smc: particles.SMC):
        collected = getattr(smc.summaries, 'online_smooth_' + algo_name.split('_')[0])
        output = {str(T): collected[T-1] for T in Ts} 
        output['cpu'] = smc.cpu_time
        return output
    return out_func

def generate_results_cdssm(config_name: str, config: dict) -> pd.DataFrame:

    add_func_dict = load_add_funcs()
    add_func_dict = {k: v for k, v in add_func_dict.items() if k in config['add_funcs'] and hasattr(v, 'cdssm')}
    output_dfs = []

    for cdssm_name, seed in config['cdssm'].items():

        cdssm = build_cdssm(CDSSM_LIB[cdssm_name])
        np.random.seed(seed)
        x, y = cdssm.simulate(config['T'], num=1000)

        for add_func_name, add_func in add_func_dict.items():
            
            cdssm.add_func = MethodType(add_func.cdssm, cdssm)
            
            for algo_name, algo_settings in config['algorithms'].items():

                if not algo_settings['N']:
                    continue
                
                collector = build_collector(algo_name)
                all_fks_dict = gen_all_fk_models(cdssm, y)
                fk_models_dict = {fk_model_s_name: all_fks_dict[fk_model_s_name] for fk_model_s_name in algo_settings["fk_model"]}
                out_func = gen_out_func(algo_name, config['Ts'])
                
                print(f"Running multiCDSSM_SMC for config: {config_name}, cdssm: {cdssm_name}, add_func: {add_func_name}, algo: {algo_name}")
                print(f"N={algo_settings['N']}, fk_models: {list(fk_models_dict.keys())}")
                
                start = perf_counter()
                output = multiCDSSM_SMC(nruns=config['M'], nprocs=1, out_func=out_func, collect=[collector], N=algo_settings['N'], fk=fk_models_dict, **config['cdssm_smc_kwargs'])
                run_time = perf_counter() - start

                print(f"Run complete. CPU time: {run_time}.")
                print(f"Converting results into a dataframe.")
                
                df = pd.json_normalize(output)
                # Column Names: 'run', 'fk', 'N', 'num', 'output.10', 'output.20', 'output.cpu', 
                df['cdssm'] = cdssm_name
                df['cdssm_seed'] = seed
                df['add_func'] = add_func_name
                df['algorithm'] = algo_name
                
                print(f"Conversion complete.")
                output_dfs.append(df)
    
    print("All runs completed. Concatenating dataframes")

    results_df = pd.concat(output_dfs, axis=0, ignore_index=True)
    
    df_columns = ['cdssm', 'cdssm_seed', 'add_func', 'algorithm', 'fk', 'run', 'N', 'seed', 'cpu'] + [str(T) for T in config['Ts']]
    results_df = results_df[df_columns]
    return results_df

def generate_results_lgssm(config_name: str, config: dict) -> pd.DataFrame:

    add_func_dict = load_add_funcs()
    add_func_dict = {k: v for k, v in add_func_dict.items() if k in config['add_funcs'] and hasattr(v, 'ssm')}
    output_dfs = []

    for cdssm_name, seed in config['cdssm'].items():

        cdssm = build_cdssm(CDSSM_LIB[cdssm_name])
        np.random.seed(seed)
        x, y = cdssm.simulate(config['T'], num=config['simulation_num'])

        lgssm = cdssm.lgssm()
        
        for add_func_name, add_func in add_func_dict.items():
                            
            lgssm.add_func = MethodType(add_func.ssm, lgssm)
            
            for algo_name, algo_settings in config['algorithms'].items():

                if not algo_settings['N']:
                    continue
                
                collector = build_collector(algo_name)
                all_ssm_fk_models_dict = gen_all_ssm_fk_models(lgssm, data=y)
                ssm_fk_models_dict = {'SSM_' + fk_name: all_ssm_fk_models_dict[fk_name] for fk_name in config['ssm_fk_models']}
                out_func = gen_out_func(algo_name, config['Ts'])
                
                print(f"Running multi_SMC for config: {config_name}, cdssm: {cdssm_name}, add_func: {add_func_name}, algo: {algo_name}")
                print(f"N={algo_settings['N']}, fk_models: {list(ssm_fk_models_dict.keys())}")
                
                ssm_smc_kwargs = {k: v for k, v in config['cdssm_smc_kwargs'].items() if k != 'num'}
                start = perf_counter()
                output = particles.multiSMC(nruns=config['M'], nprocs=1, out_func=out_func, collect=[collector], N=algo_settings['N'], fk=ssm_fk_models_dict, **ssm_smc_kwargs)
                run_time = perf_counter() - start

                print(f"Run complete. CPU time: {run_time}.")
                print(f"Converting results into a dataframe.")
                
                df = pd.json_normalize(output)
                df['cdssm'] = cdssm_name
                df['cdssm_seed'] = seed
                df['add_func'] = add_func_name
                df['algorithm'] = algo_name
                
                print(f"Conversion complete.")
                output_dfs.append(df)
    
    print("All runs completed. Concatenating dataframes")

    results_df = pd.concat(output_dfs, axis=0, ignore_index=True)
    
    df_columns = ['cdssm', 'cdssm_seed', 'add_func', 'algorithm', 'fk', 'run', 'N', 'seed', 'cpu'] + [str(T) for T in config['Ts']]
    results_df = results_df[df_columns]
    return results_df

def generate_ground_truth(config_name: str, config: dict):
    
    add_func_dict = load_add_funcs()
    add_func_dict = {k: v for k, v in add_func_dict.items() if k in config['add_funcs'] and hasattr(v, 'ground_truth')}

    output_dfs = []
    for cdssm_name, seed in config['cdssm'].items():

        cdssm = build_cdssm(CDSSM_LIB[cdssm_name])
        np.random.seed(seed)
        x, y = cdssm.simulate(config['T'], num=config['simulation_num'])

        lgssm = cdssm.lgssm()
        df_dict = {add_func_name: [] for add_func_name in add_func_dict.keys()}
        
        print(f"Running RTS smoother for config: {config_name}, cdssm: {cdssm_name}")

        for t in config['Ts']:
            kalman = Kalman(ssm=lgssm, data=y[:t])
            kalman.filter()
            kalman.smoother()
            for add_func_name, add_func in add_func_dict.items():
                df_dict[add_func_name].append(add_func.ground_truth(kalman.smth))
        idx = [str(T) for T in config['Ts']]
        output_df = pd.DataFrame(df_dict, index=idx).T.reset_index()
        output_df['cdssm'] = cdssm_name
        output_dfs.append(output_df)

        print(f"Run complete.")

    print("RTS smoother runs complete. Concatenating dataframes")
    ground_truth_df = pd.concat(output_dfs, axis=0, ignore_index=True)
    ground_truth_df = ground_truth_df.rename(columns={"index": "add_func"})
    ground_truth_df = ground_truth_df[["cdssm", "add_func"] + idx]
    return ground_truth_df
                    
def store_results(results_df: pd.DataFrame, config_name: str, ground_truth=False):
    ground_truth_or_smc = 'ground_truth' if ground_truth else 'smc'
    local_or_remote = 'remote' if isremote() else 'local'
    storage_path = f'./results/{ground_truth_or_smc}/{local_or_remote}/r_logistic_int_config.json'
    if debug:        
        storage_path = f'./paper/logistic_sde_o_smoothing/results/{ground_truth_or_smc}/{local_or_remote}/r_{str(config_name)}'
    else:
        storage_path = f'./results/{ground_truth_or_smc}/{local_or_remote}/r_{str(config_name)}'
    print("Storing dataframe in json format")
    results_df.to_json(storage_path)
    print('Storage complete.')

def cdssm_main():
    config_name, config = load_config() if not debug else load_config_debug()
    results_df = generate_results_cdssm(config_name, config)
    store_results(results_df, config_name)
    
def ssm_main():
    config_name, config = load_config() if not debug else load_config_debug()
    ssm_df = generate_results_lgssm(config_name, config)
    store_results(ssm_df, config_name)

def ground_truth_main():
    config_name, config = load_config() if not debug else load_config_debug()
    ground_truth_df = generate_ground_truth(config_name=config_name, config=config)
    store_results(ground_truth_df, config_name, ground_truth=True)

def main():
    config_name, config = load_config() if not debug else load_config_debug()
    df_list = []
    if config['run_cdssm']:
        cdssm_df = generate_results_cdssm(config_name, config)
        df_list.append(cdssm_df)
    if config['run_lgssm']:
        ssm_df = generate_results_lgssm(config_name, config)
        df_list.append(ssm_df)
        ground_truth_df = generate_ground_truth(config_name=config_name, config=config)
        store_results(ground_truth_df, config_name, ground_truth=True)        
    results_df = pd.concat(df_list, axis=0, ignore_index=True)
    store_results(results_df, config_name)

if __name__ == '__main__':
    main()