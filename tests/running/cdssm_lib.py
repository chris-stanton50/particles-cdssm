"""
Contains the CDSSMs parameterisations used in the tests/experiments in this folder.
"""
import numpy as np
import particles_cdssm.sdes as sdes
from particles_cdssm.continuous_discrete_ssms import NormalCDSSM, MvNormalCDSSM
import particles.distributions as dists

#----------------------------------------------------------------------------------------------
# ---------------------------- OrnsteinUhlenbeck + NormalCDSSM ------------------------------

OU = {
    'sde_cls': sdes.OrnsteinUhlenbeck,
    'cdssm_cls': NormalCDSSM,
    'sde_params': {'rho': 0.01, 'mu': 0., 'phi': 0.01},
    'cdssm_params': {'x0': dists.Normal(loc=0., scale=1.), 's_ts': 1., 'sigmaY': 1.0},
    'seed': 34953,
    }

#------------------------------------------------------------------------------------------
# ----------------------- MvOrnsteinUhlenbeck + MvNormalCDSSM -----------------------------

MV_OU = {
    'sde_cls': sdes.MvOrnsteinUhlenbeck,
    'cdssm_cls': MvNormalCDSSM,
    'sde_params': {'dimX': 2,
                'rho': np.array([0.01, 1.0]).reshape((1, 2)),
                'mu': np.zeros((1, 2)),
                'phi': np.array([[0.01, 0.], [0., 1.0]]),
                },
    'cdssm_params': {'x0': dists.MvNormal(loc=np.zeros((1, 2)), cov=np.eye(2)),
                    's_ts': 1.,
                    'G': np.eye(2),
                    'covY': np.array([[1., 0.],[0., 0.1]])
                    },
    'seed': 34953,
    }

#---------------------------------------------------------------------------------------------
#---------------------IntegratedOrnsteinUhlenbeck + MvNormalCDSSM---------------------------

IOU = {
    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
    'cdssm_cls': MvNormalCDSSM,
    'sde_params': {'dimX': 2,
                    'rho': 1.0*np.ones((1, 1)),
                    'mu': np.zeros((1, 1)),
                    'phi': 1.0*np.ones((1, 1)),
                    },
    'cdssm_params': {'dimX': 2,
                    'rho': 0.01*np.ones((1, 1)),
                    'mu': np.zeros((1, 1)),
                    'phi': 0.01*np.ones((1, 1)),
                    },
    'seed': 34953,
    }

CDSSM_LIB = {'ou': OU,
             'mv_ou': MV_OU,
             'iou': IOU,
            }