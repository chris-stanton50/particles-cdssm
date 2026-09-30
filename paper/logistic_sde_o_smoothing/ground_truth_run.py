import numpy as np
import pandas as pd
from types import MethodType
from time import perf_counter

import particles_cdssm
from particles_cdssm.cdssm_lib import CDSSM_LIB
from particles_cdssm.tools import build_cdssm, isremote
from particles_cdssm.feynman_kac import BootstrapReparameterisedDA
from particles_cdssm.collectors import Online_smooth_MCMC
from settings.add_funcs import DS_Integral_ExpThetaX


# Parameters - ensure they match the final config
N = int(1e5)
T = int(1e4)
Ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 21, 27, 35, 46, 59, 77, 100, 129, 166, 215, 278, 359, 464, 599, 774, 1000, 291, 1668, 2154, 2782, 3593, 4641, 5994, 7742, 10000]


def results_to_df(results: list):
    idx = [t-1 for t in Ts]
    results = np.array(results)[idx]
    results_df = pd.Series(data=results, index=[str(t) for t in Ts], dtype=np.float64).to_frame().T
    results_df = pd.concat([pd.DataFrame({'cdssm': ['LAMPERTI_LOGISTIC_NB_LOW'], 'add_func': ['DS_Integral_ExpThetaX']}), results_df], axis=1)
    return results_df

def main():
    
    print('Building CD-SSM and simulating dataset...')
    
    # Build the cdssm and simulate from it 
    cdssm = build_cdssm(CDSSM_LIB["LAMPERTI_LOGISTIC_NB_LOW"])
    np.random.seed(55765)
    x, y = cdssm.simulate(T, num=1000)
    
    print('Simulation complete.')
    
    # Add the add_func method for the integral of the population w.r.t s to the cdssm
    cdssm.add_func = MethodType(DS_Integral_ExpThetaX.cdssm, cdssm)

    # Build the Feynman-Kac model
    fk = BootstrapReparameterisedDA(cdssm=cdssm, data=y)

    # Build the SMC object
    smc = particles_cdssm.CDSSM_SMC(
            fk=fk,
            N=N,
            resampling="systematic",
            ESSrmin=1.0,
            store_history=False,
            verbose=False,
            collect=[Online_smooth_MCMC(nsteps=1)],
            num=50
            )

    # Running the algorithm for T steps
    print('Running the online smoothing algorithm...')
    start_time = perf_counter()
    for i in range(T):
        smc.next()
        if i % 100 == 0:
            curr_time = perf_counter()
            print(f'Iteration {i} of {T}. Time elapsed: {round(curr_time - start_time, 2)}s')

    end_time = perf_counter()
    print(f'Online smoothing algorithm run complete. Total time: {round(end_time - start_time, 2)}')
    
    print('Storing the results...')
    # Covert results to dataframe with the right format
    results_df = results_to_df(smc.summaries.online_smooth_MCMC)    

    # Store the ground truth
    local_or_remote = 'remote' if isremote() else 'local'
    file_path = f'./results/ground_truth/{local_or_remote}/r_logistic_int_config.json'
    results_df.to_json(file_path)
    print('Storage complete. Results saved to ' + file_path)   

if __name__ == '__main__':
    main()