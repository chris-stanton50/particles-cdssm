from sdes.cdssm_lib import CDSSM_LIB, build_cdssm
from sdes.core import CDSSM_SMC
import sdes.feynman_kac as sfk

T = 10; N=100; num=10

# Build the cdssm
cdssm_str = 'ou'
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, False)

# Simulate synthetic data from the cdssm
x, y = cdssm.simulate(T)

# Generate all possible fk models for the given cdssm
all_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)

# Run the CDSSM_SMC algorithm for each fk model
algs = {}
for fk_name, fk in all_fks.items():
    algs[fk_name] = CDSSM_SMC(fk=fk, N=N, num=10, store_history=True)
    
for fk_name, smc in algs.items():
    print(f'Running CDSSM_SMC for {fk_name}')
    smc.run()
