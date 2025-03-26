from particles_cdssm.cdssm_lib import CDSSM_LIB, build_cdssm
from particles_cdssm.core import CDSSM_SMC
import particles_cdssm.feynman_kac as sfk

T = 5; N=10; num=10; M=5

# Build the cdssm
cdssm_str = 'ou'
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, True)

# Simulate synthetic data from the cdssm
x, y = cdssm.simulate(T)

# Generate all possible fk models for the given cdssm
all_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)
all_smth_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=True)

# Run backward_sampling_genealogy for all filtering fks
filt_algs = {}
for fk_name, fk in all_fks.items():
    filt_algs[fk_name] = CDSSM_SMC(fk=fk, N=100, store_history=True)
    
for fk_name, smc in filt_algs.items():
    smc.run()
    print(f'Running backward_sampling_genealogy for {fk_name}')
    samples = smc.hist.backward_sampling_genealogy(M)

# Run `backward_sampling_` + genealogy, ON2 and mcmc for all smoothing fks
smth_algs = {}
for fk_name, fk in all_smth_fks.items():
    smth_algs[fk_name] = CDSSM_SMC(fk=fk, N=100, num=10, store_history=True)    

for fk_name, smc in smth_algs.items():
    smc.run()
    print(f'Running backward_sampling_genealogy for {fk_name}')
    samples_g = smc.hist.backward_sampling_genealogy(M)
    print(f'Running backward_sampling_ON2 for {fk_name}')
    samples_o = smc.hist.backward_sampling_ON2(M)
    print(f'Running backward_sampling_mcmc for {fk_name}')
    samples_m = smc.hist.backward_sampling_mcmc(M)