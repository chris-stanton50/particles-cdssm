"""
Contains the CDSSMs parameterisations used in the tests/experiments in this folder.
"""
import numpy as np
import particles_cdssm.sdes as sdes
from particles_cdssm.continuous_discrete_ssms import MvNormalCDSSM

#------------------------------------------------------------------------------------------
# ----------------------- MvOrnsteinUhlenbeck + MvNormalCDSSM -----------------------------

MV_OU = {
    'sde_cls': sdes.MvOrnsteinUhlenbeck,
    'cdssm_cls': MvNormalCDSSM,
    'sde_params': {'dimX': 2,
                'rho': 1.0*np.ones((1, 2)),
                'mu': np.zeros((1, 2)),
                'phi': 1.0*np.eye(2),
                },
    'cdssm_params': {'x0': np.zeros((1, 2)),
                    's_ts': 1., 
                    'G': np.eye(2),
                    'covY': (0.1 ** 2) * np.eye(2)
                    },
    'high_noise_param': (1.0 ** 2) * np.eye(2),
    'fk_names': ['BootstrapDA', 'FwG_NDBrP', 'BwG_NDBr_OUP', 'BsR_DH', 'FwR_DH_NDBrP', 'BwR_NDBr_OUP'],
    'fk_names_map': {'BootstrapDA': 'Bootstrap-GT', 'BwG_NDBr_OUP': 'Backward-GT', 'FwG_NDBrP': 'Forward-GT', 'BsR_DH': 'Bootstrap-FFBS', 'FwR_DH_NDBrP': 'Forward-FFBS', 'BwR_NDBr_OUP': 'Backward-FFBS'},
    'seed': 34953,
    }
#---------------------IntegratedOrnsteinUhlenbeck + MvNormalCDSSM---------------------------

IOU = {
    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
    'cdssm_cls': MvNormalCDSSM,
    'sde_params': {'dimX': 2,
                    'rho': 1.0*np.ones((1, 1)),
                    'mu': np.zeros((1, 1)),
                    'phi': 1.0*np.ones((1, 1)),
                    },
    'cdssm_params': {
                    'x0': np.zeros((1, 2)),
                    's_ts': 1.,
                    'G': np.eye(2),
                    'covY': (0.1 ** 2) * np.eye(2)
                    },
    'high_noise_param': (1.0 ** 2) * np.eye(2),
    'fk_names': ['BootstrapDA', 'BwG_NDBr_OUP', 'BwR_NDBr_OUP'],
    'fk_names_map': {'BootstrapDA': 'Bootstrap-GT', 'BwG_NDBr_OUP': 'Backward-GT', 'BwR_NDBr_OUP': 'Backward-FFBS'},
    'seed': 34953,
    }

CDSSM_LIB = {'mv_ou': MV_OU,
             'iou': IOU,
            }