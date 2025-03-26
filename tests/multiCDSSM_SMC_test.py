from particles.collectors import Moments
from sdes.core import summaries
from sdes.cdssm_lib import CDSSM_LIB, build_cdssm
from sdes.core import multiCDSSM_SMC
import sdes.feynman_kac as sfk

T=10; Ns = [10, 20, 30]; nums = [10, 20, 30]

# Build the cdssm
cdssm_str = 'ou'
cdssm_spec = CDSSM_LIB[cdssm_str]
cdssm = build_cdssm(cdssm_str, False)

# Simulate synthetic data from the cdssm
x, y = cdssm.simulate(T)

# Generate all possible fk models for the given cdssm
all_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)

# `collect` is a protected arg, because it is expected that we pass a list of 
# collectors to the `CDSSM_SMC` class
# If no out_func is specified, each indpendent pf run will store the full SMC object.
# An out_func can instead be passed that ensures that only the required output from the pf is stored.
out = multiCDSSM_SMC(nruns=2, nprocs=0, out_func=summaries, collect=[Moments], fk=all_fks, N=Ns, num=nums)