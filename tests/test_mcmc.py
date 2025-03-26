from sdes.cdssm_lib import build_cdssm
from sdes.mcmc import PIMH, CDSSM_PIMH, ICSMC, CDSSM_ICSMC
import particles.state_space_models as ssms
from particles.kalman import MVLinearGauss, LinearGauss
import sdes.feynman_kac as sfk

# Data params
T = 10

# MCMC Params
niter=100;  Nx=100; num=10

CDSSM_PIMH_params = {'niter': niter, 'Nx': Nx, 'num': num}
CDSSM_CSMC_params = {'niter': niter, 'Nx': Nx, 'num': num, 'backward_step': False}
CDSSM_CSMC_BS_params = {'niter': niter, 'Nx': Nx, 'num': num, 'backward_step': True}

cdssm_mcmc_alg_classes = [CDSSM_PIMH, CDSSM_ICSMC, CDSSM_ICSMC]
cdssm_mcmc_alg_params = [CDSSM_PIMH_params, CDSSM_CSMC_params, CDSSM_CSMC_BS_params]

cdssm = build_cdssm('mv_ou')
x, y = cdssm.simulate(T)
fk = sfk.BackwardReparameterisedDA(cdssm=cdssm, data=y)


print('Running CDSSM MCMC algorithms')
for alg_class, alg_params in zip(cdssm_mcmc_alg_classes, cdssm_mcmc_alg_params):
    alg_params['fk'] = fk
    alg = alg_class(**alg_params)
    disp_str = f'Running {alg.__class__.__name__} algorithm'
    if hasattr(alg, 'backward_step') and alg.backward_step:
        disp_str += ' with backward step'
    print(disp_str)
    alg.run()

"""
This bit tests the algorithms on standard SSMs and FKs, no data augmentation:
"""

# # Define algorithms and parameters
# PIMH_params = {'niter': niter, 'Nx': Nx}
# CSMC_params = {'niter': niter, 'Nx': Nx, 'backward_step': False}
# CSMC_BS_params = {'niter': niter, 'Nx': Nx, 'backward_step': True}

# ssm_mcmc_alg_classes = [PIMH, CSMC, CSMC]
# ssm_mcmc_alg_params = [PIMH_params, CSMC_params, CSMC_BS_params]
# example_ssms = [ssms.Gordon_etal(), ssms.BearingsOnly(), ssms.DiscreteCox(), ssms.ThetaLogistic(), LinearGauss()]

# example_fks = [None]*len(example_ssms)

# for i, ssm in enumerate(example_ssms):
#     x, y = ssm.simulate(T)
#     fk_cls = ssms.Bootstrap if ssm.__class__.__name__ != 'LinearGauss' else ssms.GuidedPF
#     fk = fk_cls(ssm=ssm, data=y)
#     example_fks[i] = fk

# # Add a final ssm that is the discrete time version of a CD-SSM:
# cdssm = build_cdssm('mv_ou')
# ssm = cdssm.discrete_ssm()
# x, y = ssm.simulate(T)
# fk = ssms.GuidedPF(ssm=ssm, data=y)

# example_fks.append(fk)
    
# # Test that implementations of PIMH and CSMC (excl. the parameter) run:
        
# print('Running SSM MCMC algorithms')
# for fk in example_fks:
#     disp_str_ext = f' for ssm {fk.ssm.__class__.__name__} with fk {fk.__class__.__name__}'
#     for alg_class, alg_params in zip(ssm_mcmc_alg_classes, ssm_mcmc_alg_params):
#         alg_params['fk'] = fk
#         alg = alg_class(**alg_params)
#         disp_str = f'Running {alg.__class__.__name__} algorithm'
#         if hasattr(alg, 'backward_step') and alg.backward_step:
#             disp_str += ' with backward step'
#         disp_str += disp_str_ext
#         print(disp_str)
#         alg.run()


