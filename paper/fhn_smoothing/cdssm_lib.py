"""
Contains the CDSSMs parameterisations used in the tests/experiments in this folder.
"""
import numpy as np
import particles_cdssm.sdes as sdes
from particles_cdssm.continuous_discrete_ssms import MvNormalCDSSM


#---------------------IntegratedOrnsteinUhlenbeck + MvNormalCDSSM---------------------------

IFHN = {
    'sde_cls': sdes.IntegratedFitzhughNagumo,
    'cdssm_cls': MvNormalCDSSM,
    'sde_params': {'epsilon': 0.1, # Parameters from Dvitlivsen and Samson (2024)
                      'gamma': 1.5,
                      'beta': 0.8,
                      'sigma_u': 0.3
                    }, 
    'cdssm_params': {
                    'x0': np.zeros((1, 2)),
                    's_ts': 0.05, # Change this parameter
                    'G': np.array([[1., 0.]]),
                    'covY': (0.01 ** 2) * np.eye(1) # This should be set to be quite low
                    },
    'fk_names': ['BootstrapDA', 'BwG_NDBr_OUP', 'BwR_NDBr_OUP'],
    'seed': 3563,
    'fk_names_map': {'BootstrapDA': 'Bootstrap', 'BwG_NDBr_OUP': 'Backward'}
    }

CDSSM_LIB = {'ifhn': IFHN}