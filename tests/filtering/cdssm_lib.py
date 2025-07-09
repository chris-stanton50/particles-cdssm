"""
Contains the CDSSMs parameterisations used in the tests/experiments in this folder.
"""
import numpy as np
import particles_cdssm.sdes as sdes
from particles_cdssm.continuous_discrete_ssms import GaussianCDSSM, MvGaussianCDSSM

#----------------------------------------------------------------------------------------------
# ---------------------------- OrnsteinUhlenbeck + GaussianCDSSM ------------------------------

OU = {
    'sde_cls': sdes.OrnsteinUhlenbeck,
    'cdssm_cls': GaussianCDSSM,
    'sde_params': {'rho': 1.0, 'mu': 0., 'phi': 1.0},
    'cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 0.1},
    'seed': 34953,
    'n_bottom_fks': 6,
    'n_top_fks': 7
    }

BM = {
    'sde_cls': sdes.BrownianMotion,
    'cdssm_cls': GaussianCDSSM,
    'sde_params': {'m': 0., 's': 1.0},
    'cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 0.1},
    'seed': 34953,
    'n_bottom_fks': 6,
    'n_top_fks': 19,
    }

#------------------------------------------------------------------------------------------
# ----------------------- MvOrnsteinUhlenbeck + MvGaussianCDSSM -----------------------------

MV_OU = {
    'sde_cls': sdes.MvOrnsteinUhlenbeck,
    'cdssm_cls': MvGaussianCDSSM,
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
    'seed': 34953,
    'n_bottom_fks': 6,
    'n_top_fks': 9,
    }

MV_BM = {
    'sde_cls': sdes.MvBrownianMotion,
    'cdssm_cls': MvGaussianCDSSM,
    'sde_params': {'dimX': 2,
                'm': np.zeros((1, 2)),
                's': 1.0*np.eye(2),
                },
    'cdssm_params': {'x0': np.zeros((1, 2)),
                    's_ts': 1., 
                    'G': np.eye(2),
                    'covY': (0.1 ** 2) * np.eye(2)
                    },
    'seed': 34953,
    'n_bottom_fks': 6,
    'n_top_fks': 25,
    }

#---------------------------------------------------------------------------------------------
#---------------------IntegratedOrnsteinUhlenbeck + MvGaussianCDSSM---------------------------

IOU = {
    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
    'cdssm_cls': MvGaussianCDSSM,
    'sde_params': {'dimX': 2,
                    'rho': 1.0*np.ones((1, 1)),
                    'mu': np.zeros((1, 1)),
                    'phi': 1.0*np.ones((1, 1)),
                    },
    'cdssm_params': {
                    'x0': np.zeros((1, 2)),
                    's_ts': 1.,
                    'G': np.array([1., 0.]).reshape((1, 2)), # smooth component observed
                    'covY': (0.02 ** 2) * np.eye(1) # low noise regime
                    },
    'seed': 34953,
    'n_bottom_fks': 2,
    'n_top_fks': 17,
    }

IBM = {
    'sde_cls': sdes.IntegratedBrownianMotion,
    'cdssm_cls': MvGaussianCDSSM,
    'sde_params': {'dimX': 2,
                    'm': np.zeros((1, 1)),
                    's': 1.0*np.ones((1, 1))
                    },
    'cdssm_params': {
                    'x0': np.zeros((1, 2)),
                    's_ts': 1.,
                    'G': np.array([1., 0.]).reshape((1, 2)), # smooth component observed
                    'covY': (0.02 ** 2) * np.eye(1) # low noise regime
                    },
    'seed': 34953,
    'n_bottom_fks': 2,
    'n_top_fks': 17,
}

#---------------------TwiceIntegratedOrnsteinUhlenbeck + MvGaussianCDSSM---------------------------


I2OU = {
    'sde_cls': sdes.TwiceIntegratedOrnsteinUhlenbeck,
    'cdssm_cls': MvGaussianCDSSM,
    'sde_params': {'dimX': 3,
                    'rho': 1.0*np.ones((1, 1)),
                    'mu': np.zeros((1, 1)),
                    'phi': 1.0*np.ones((1, 1)),
                    },
    'cdssm_params': {
                    'x0': np.zeros((1, 3)),
                    's_ts': 1.,
                    'G': np.array([1., 0., 0.]).reshape((1, 3)), # smoothest component observed
                    'covY': (0.1 ** 2) * np.eye(1) # low noise regime
                    },
    'default_T': 5,
    'seed': 34953,
    'n_bottom_fks': 2,
    'n_top_fks': None,
    }

I2BM = {
    'sde_cls': sdes.TwiceIntegratedBrownianMotion,
    'cdssm_cls': MvGaussianCDSSM,
    'filtering_sde_params': {'dimX': 3,
                    'm': np.zeros((1, 1)),
                    's': 1.0*np.ones((1, 1)),
                    },
    'filtering_cdssm_params': {
                    'x0': np.zeros((1, 3)),
                    's_ts': 1.,
                    'G': np.array([1., 0., 0.]).reshape((1, 3)), # smoothest component observed
                    'covY': (0.02 ** 2) * np.eye(1) # low noise regime
                    },
    'default_T': 5,
    'seed': 34953,
    'n_bottom_fks': 2,
    'n_top_fks': None,
    }


CDSSM_LIB = {'ou': OU,
             'bm': BM,
             'mv_ou': MV_OU,
             'mv_bm': MV_BM,
             'iou': IOU,
             'ibm': IBM,
             'i2ou': I2OU,
             'i2bm': I2BM
            }