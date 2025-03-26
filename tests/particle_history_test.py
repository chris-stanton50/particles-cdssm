import numpy as np
import particles.state_space_models as ssms
from particles.kalman import MVLinearGauss, LinearGauss
from sdes.core import SMC
from sdes.cdssm_lib import CDSSM_LIB, build_cdssm

T = 5; N=10; M=5

# Generate examples of fks based on all SSMs defined in the particles package, with default choice of parameters:
bootstrap_ssm_list = [ssms.StochVol(), ssms.StochVolLeverage(), ssms.Gordon_etal(), ssms.BearingsOnly(), ssms.DiscreteCox(), ssms.ThetaLogistic()]
guided_ssm_list = [MVLinearGauss(covX=np.eye(2), covY=np.eye(2)), LinearGauss()]

all_fks = {}

for ssm in bootstrap_ssm_list:
    x, y = ssm.simulate(T)
    all_fks[ssm.__class__.__name__] = ssms.Bootstrap(ssm=ssm, data=y)

for ssm in guided_ssm_list:
    x, y = ssm.simulate(T)
    all_fks[ssm.__class__.__name__] = ssms.GuidedPF(ssm=ssm, data=y)

# Run backward_sampling_genealogy for all fks
filt_algs = {}
for fk_name, fk in all_fks.items():
    filt_algs[fk_name] = SMC(fk=fk, N=100, store_history=True)
    
for fk_name, smc in filt_algs.items():
    smc.run()
    print(f'Running backward_sampling_genealogy for {fk_name}')
    samples = smc.hist.backward_sampling_genealogy(M)