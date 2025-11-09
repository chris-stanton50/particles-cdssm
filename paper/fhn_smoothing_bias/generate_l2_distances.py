import numpy as np
import distances as dsts
from particles_cdssm.sdes import IntegratedFitzhughNagumo
from time import perf_counter

np.random.seed(23544)
n_replicates = 1000

def run():
    sde = IntegratedFitzhughNagumo()

    N_kde = 5000 # Number of samples to calculate the KDEs
    true_k = 1000 # Number of points imputed to obtain 'true' transition density.

    x_0 = np.array([-1.006165250497603, -6.922579807639913])

    np.random.seed(14534)
    true_sample = sde.simulate_ldl(N_kde, x_start=np.stack([x_0]*N_kde, axis=0), delta_t=0.05, num=true_k)[-1].T
    
    conf_l2_distances = np.empty((n_replicates,), dtype=np.float64)
    run_times = np.empty((n_replicates,), dtype=np.float64)
        
    for i in range(n_replicates):
        if i % 1 == 0:
            print(f'Generating {i}th ' + r'$\mathcal{L}_2$' + ' distance...')
        sample = sde.simulate_ldl(N_kde, x_start=np.stack([x_0]*N_kde, axis=0), delta_t=0.05, num=true_k)[-1].T
        time = perf_counter()
        l_2_distance = dsts.kdes_l2_distance(sample, true_sample)
        time = perf_counter() - time
        conf_l2_distances[i] = l_2_distance
        print(f'L_2 distance generated: {l_2_distance}: Run time: {time}')
        run_times[i] = time
    print(f'Saving ' + r'$\mathcal{L}_2$' + ' distances...')
    np.save("l_2_distances.npy", conf_l2_distances)
    print('L_2 distances saved to l_2 distances.npy. Run complete.')
    print(f'Run times summary: N: {run_times.size} Mean: {run_times.mean()} Std: {run_times.std()}')

if __name__ == '__main__':
    run()
    