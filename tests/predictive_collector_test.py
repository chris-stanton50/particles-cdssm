import numpy as np
from particles.kalman import MVLinearGauss, LinearGauss

import particles.state_space_models as ssms
from particles.core import SMC

from particles_cdssm.core import CDSSM_SMC
from particles_cdssm.cdssm_lib import CDSSM_LIB, build_cdssm
import particles_cdssm.feynman_kac as sfk
import particles_cdssm.collectors as cols

# Define params for the length of the dataset/number of particles:
T=10; N=10; num=10

col_classes = [cols.PredictiveParticles, cols.ObservationPredictiveParticles, cols.PredictiveMoments, cols.ObservationPredictiveMoments, cols.NLPD, cols.LowVarianceNLPD, cols.AbsoluteError]
col_kwargs_list = [{'K': 1, 'method': 'simulated'}]*len(col_classes)

collectors = [col_class(**col_kwargs) for col_class, col_kwargs in zip(col_classes, col_kwargs_list)]

# Generate examples of fks based on all SSMs defined in the particles package, with default choice of parameters:
bootstrap_ssm_list = [ssms.StochVol(), ssms.StochVolLeverage(), ssms.Gordon_etal(), ssms.BearingsOnly(), ssms.DiscreteCox(), ssms.ThetaLogistic(), MVLinearGauss(covX=np.eye(2), covY=np.eye(2)), LinearGauss()]
guided_ssm_list = [MVLinearGauss(covX=np.eye(2), covY=np.eye(2)), LinearGauss()]

all_fks = {}

for ssm in bootstrap_ssm_list:
    x, y = ssm.simulate(T)
    all_fks[ssm.__class__.__name__] = ssms.Bootstrap(ssm=ssm, data=y)

for ssm in guided_ssm_list:
    x, y = ssm.simulate(T)
    all_fks[ssm.__class__.__name__] = ssms.GuidedPF(ssm=ssm, data=y)

for ssm_name, fk in all_fks.items():
    fk_name = fk.__class__.__name__
    for col in collectors:
        col_name = col.__class__.__name__
        smc = SMC(fk=fk, N=N, store_history=False, collect=[col])
        print(f"Running SMC for Model: {ssm_name} FK: {fk_name}, with collector {col_name}")
        smc.run()


# Build the cdssm
cdssm_strs = ['ou', 'mv_ou', 'iou']
# cdssm_strs = ['ou']

for cdssm_str in cdssm_strs:
    cdssm_spec = CDSSM_LIB[cdssm_str]
    cdssm = build_cdssm(cdssm_str, False)

    # Simulate synthetic data from the cdssm
    x, y = cdssm.simulate(T)

    # Generate all possible fk models for the given cdssm
    all_cdssm_fks = sfk.gen_all_fk_models(cdssm, y, smoothing=False)

    for fk_name, fk in all_cdssm_fks.items():
        for col in collectors:
            smc = CDSSM_SMC(fk=fk, N=N, store_history=False, collect=[col], num=num)
            print(f"Running CDSSM_SMC for Model:{cdssm_str} FK: {fk_name} with Collector:{col.__class__.__name__}")
            smc.run()


print("All tests completed successfully.")

