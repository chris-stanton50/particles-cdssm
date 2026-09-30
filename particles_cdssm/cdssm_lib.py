"""
Contains CDSSM parameterisations used in all numerical experiments.
"""
import numpy as np
import particles_cdssm.sdes as sdes
from particles_cdssm.continuous_discrete_ssms import (
    NormalCDSSM,
    MvNormalCDSSM,
    NegativeBinomialCDSSM,
    LampertiLogisticNegativeBinomialCDSSM,
)


CDSSM_LIB = {

# ----------------------- OrnsteinUhlenbeck + NormalCDSSM -----------------------------

    'OU_0.05':      {
                    'sde_cls': sdes.OrnsteinUhlenbeck,
                    'cdssm_cls': NormalCDSSM,
                    'sde_params': {'rho': 1.0,'mu': 0.0,'phi': 1.0,},
                    'cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 0.05}
                    },
    'OU_0.1':      {
                    'sde_cls': sdes.OrnsteinUhlenbeck,
                    'cdssm_cls': NormalCDSSM,
                    'sde_params': {'rho': 1.0,'mu': 0.0,'phi': 1.0,},
                    'cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 0.1}
                    },
    'OU_0.2':       {
                    'sde_cls': sdes.OrnsteinUhlenbeck,
                    'cdssm_cls': NormalCDSSM,
                    'sde_params': {'rho': 1.0,'mu': 0.0,'phi': 1.0,},
                    'cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 0.2}
                    },
    'OU_0.5':       {
                    'sde_cls': sdes.OrnsteinUhlenbeck,
                    'cdssm_cls': NormalCDSSM,
                    'sde_params': {'rho': 1.0,'mu': 0.0,'phi': 1.0,},
                    'cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 0.5}
                    },
    'OU_1.0':       {
                    'sde_cls': sdes.OrnsteinUhlenbeck,
                    'cdssm_cls': NormalCDSSM,
                    'sde_params': {'rho': 1.0,'mu': 0.0,'phi': 1.0,},
                    'cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 1.0}
                    },

# ----------------------- MvOrnsteinUhlenbeck + MvNormalCDSSM -----------------------------
    
    'MV_OU_0.05':   {
                    'sde_cls': sdes.MvOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2,'rho': 1.0*np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': 1.0*np.eye(2),},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.05 ** 2) * np.eye(2)}
                    },
    'MV_OU_0.1':    {
                    'sde_cls': sdes.MvOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2,'rho': 1.0*np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': 1.0*np.eye(2),},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.1 ** 2) * np.eye(2)}
                    },
    'MV_OU_0.2':    {
                    'sde_cls': sdes.MvOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2,'rho': 1.0*np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': 1.0*np.eye(2),},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.2 ** 2) * np.eye(2)}
                    },
    'MV_OU_0.5':    {
                    'sde_cls': sdes.MvOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2,'rho': 1.0*np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': 1.0*np.eye(2),},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.5 ** 2) * np.eye(2)}
                    },
    'MV_OU_1.0':    {
                    'sde_cls': sdes.MvOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2,'rho': 1.0*np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': 1.0*np.eye(2),},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (1.0 ** 2) * np.eye(2)}
                    },

#---------------------IntegratedOrnsteinUhlenbeck + MvNormalCDSSM---------------------------

    'IOU_0.05':     {
                    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2, 'rho': 1.0*np.ones((1, 1)), 'mu': np.zeros((1, 1)), 'phi': 1.0*np.ones((1, 1))},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.05 ** 2) * np.eye(2)}
                    },
    'IOU_0.1':      {
                    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2, 'rho': 1.0*np.ones((1, 1)), 'mu': np.zeros((1, 1)), 'phi': 1.0*np.ones((1, 1))},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.1 ** 2) * np.eye(2)}
                    },
    'IOU_0.2':      {
                    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2, 'rho': 1.0*np.ones((1, 1)), 'mu': np.zeros((1, 1)), 'phi': 1.0*np.ones((1, 1))},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.2 ** 2) * np.eye(2)}
                    },
    'IOU_0.5':      {
                    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2, 'rho': 1.0*np.ones((1, 1)), 'mu': np.zeros((1, 1)), 'phi': 1.0*np.ones((1, 1))},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (0.5 ** 2) * np.eye(2)}
                    },
    'IOU_1.0':      {
                    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'dimX': 2, 'rho': 1.0*np.ones((1, 1)), 'mu': np.zeros((1, 1)), 'phi': 1.0*np.ones((1, 1))},
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 1., 'G': np.eye(2), 'covY': (1.0 ** 2) * np.eye(2)}
                    },

#--------------------- Lotka-Volterra CD-SSM - Ryder et al (2017) ---------------------------

    'LV_Ryder':    {
                    'sde_cls': sdes.LotkaVolterra_Ryder,
                    'cdssm_cls': MvNormalCDSSM,
                    'sde_params': {'theta': np.array([0.5, 0.0025, 0.3])},
                    'cdssm_params': {'x0': np.array([[71., 79.]]), 's_ts': 10.,
                                     'G': np.eye(2), 'covY': np.eye(2)}
                    },

#--------------------- Logistic CD-SSM - Chopin et al (2023), Knape and de Valpine (2012) ---------------------------

    'LOGISTIC_NB_V_HIGH': {
        'sde_cls': sdes.LogisticGrowthDiffusion,
        'cdssm_cls': NegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {'x0': 600., 's_ts': 1., 'theta_4': 1.069},
    },

    'LOGISTIC_NB_HIGH': {
        'sde_cls': sdes.LogisticGrowthDiffusion,
        'cdssm_cls': NegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {'x0': 600., 's_ts': 1., 'theta_4': 4.303},
    },

    'LOGISTIC_NB_MED': {
        'sde_cls': sdes.LogisticGrowthDiffusion,
        'cdssm_cls': NegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {'x0': 600., 's_ts': 1., 'theta_4': 17.631},
    },

    'LOGISTIC_NB_LOW': {
        'sde_cls': sdes.LogisticGrowthDiffusion,
        'cdssm_cls': NegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {'x0': 600., 's_ts': 1., 'theta_4': 78.161},
    },

    'LAMPERTI_LOGISTIC_NB_V_HIGH': {
        'sde_cls': sdes.LampertiLogisticGrowthDiffusion,
        'cdssm_cls': LampertiLogisticNegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {
            'x0': np.log(600.) / 0.7,
            's_ts': 1.,
            'theta_4': 1.069,
        },
    },

    'LAMPERTI_LOGISTIC_NB_HIGH': {
        'sde_cls': sdes.LampertiLogisticGrowthDiffusion,
        'cdssm_cls': LampertiLogisticNegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {
            'x0': np.log(600.) / 0.7,
            's_ts': 1.,
            'theta_4': 4.303,
        },
    },

    'LAMPERTI_LOGISTIC_NB_MED': {
        'sde_cls': sdes.LampertiLogisticGrowthDiffusion,
        'cdssm_cls': LampertiLogisticNegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {
            'x0': np.log(600.) / 0.7,
            's_ts': 1.,
            'theta_4': 17.631,
        },
    },

    'LAMPERTI_LOGISTIC_NB_LOW': {
        'sde_cls': sdes.LampertiLogisticGrowthDiffusion,
        'cdssm_cls': LampertiLogisticNegativeBinomialCDSSM,
        'sde_params': {'theta_1': 1.8, 'theta_2': 0.003, 'theta_3': 0.7},
        'cdssm_params': {
            'x0': np.log(600.) / 0.7,
            's_ts': 1.,
            'theta_4': 78.161,
        },
    },


#---------------------IntegratedFitzhughNagumo + MvNormalCDSSM---------------------------

    'IFHN':         {
                    'sde_cls': sdes.IntegratedFitzhughNagumo,
                    'cdssm_cls': MvNormalCDSSM,                    
                    'sde_params': {'epsilon': 0.1, 'gamma': 1.5,'beta': 0.8, 'sigma_u': 0.3}, # Parameters from Dvitlivsen and Samson (2024) 
                    'cdssm_params': {'x0': np.zeros((1, 2)), 's_ts': 0.05, 'G': np.array([[1., 0.]]), # Vary the s_t parameter
                    'covY': (0.01 ** 2) * np.eye(1)} # Covariance should be low
                    }

}
