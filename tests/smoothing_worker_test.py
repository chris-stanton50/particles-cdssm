from particles_cdssm.cdssm_lib import CDSSM_LIB, build_cdssm
from particles_cdssm.core import CDSSM_SMC
import particles_cdssm.feynman_kac as sfk
from particles.utils import multiplexer
from particles_cdssm.core import CDSSM_SMC, smoothing_worker

Ns=[5, 10, 20]; T = 5; nums = [10, 20]
smth_methods = ['FFBS_ON2', 'FFBS_MCMC', 'genealogy']

# Build the cdssm
cdssm_str = 'ou'
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, smoothing=True)

# Simulate synthetic data from the cdssm
x, y = cdssm.simulate(T)

# Generate all possible fk models for the given cdssm
all_smth_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=True)
    
# Run smoothing worker with multiplexer for all the fk models
# Note: If different add_funcs are set instead of the default choices, then 
# the add_funcs need to be passed as `protected_args`  to the multiplexer function.
cdssm_output = multiplexer(f=smoothing_worker, nruns=3, nprocs=0, seeding=True, method=smth_methods, N=Ns, fk=all_smth_fks, num=nums, smc_cls=CDSSM_SMC)