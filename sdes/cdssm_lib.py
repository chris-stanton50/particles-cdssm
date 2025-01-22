import numpy as np
import numpy.linalg as nla
import sdes.sdes as sdes
from sdes.continuous_discrete_ssms import GaussianCDSSM, TimeSwitchingGaussianCDSSM
import particles.state_space_models as ssms
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Note: Run times are based on the local: a trial run indicates that the remote is ~40% faster 
# than the local. (This is based on the MV_OU model for filtering.)
#------------------------------------------------------------------------------------------
# ------------------ TS_MvOrnsteinUhlenbeck + TimeSwitchingGaussianCDSSM ------------------
"""
(Used to test whether smoothing algorithms are working as intended. Signal changes at T=2
so that it has low variance, and observation noise changes at T=10 so that it is highly informative.
therefore, the smoothing distribution is very different from the filtering distribution for T < 10.
"""
TS_MV_OU = {
    'sde_cls': sdes.TS_MvOrnsteinUhlenbeck,
    'cdssm_cls': TimeSwitchingGaussianCDSSM,
    'sde_params': {'dimX': 2,
                    't_switch': 2,
                    'rho_1': 0.01*np.ones((1, 2)),
                    'mu_1': np.zeros((1, 2)),
                    'phi_1': nla.cholesky(np.array([[1., 0.9,], [0.9, 1.]])),
                    'rho_2': 0.01*np.ones((1, 2)),
                    'mu_2': np.zeros((1, 2)),
                    'phi_2': 0.01 * nla.cholesky(np.array([[1., 0.9,], [0.9, 1.]]))
                    },
    'cdssm_params': {'x0': np.zeros((1, 2)),
                    'delta_s': 1.,
                    'G_1': np.eye(2),
                    'G_2': np.eye(2), 
                    'covY_1': (0.4 ** 2) * np.eye(2),
                    'covY_2': 0.001 * (0.4 ** 2) * np.eye(2),
                    't_switchY': 10
                    },
    'default_T': 10,
    'seed': 34953,
    'fk_names': ['BsR_DH', 'BwR_OU_OUP', 'BwR_DH_OUP', 'BwR_DH_IOUP', 'FwR_DH_OUP', 'FwR_DH_DBrP', 'FwR_DH_NDBBrP']
    }

#------------------------------------------------------------------------------------------
# ----------------------- MvOrnsteinUhlenbeck + GaussianCDSSM -----------------------------
"""
(Used to test if guided filtering steps are working as intended. This model has a low noise regime,
so guiding particles is very important for this case.)
(For the smoothing, at T=100, performance of all offline smoothing methods are indistingusihable, 
even in the low noise regime.)
Filtering run time est: 32 mins
Smoothing run time est (excl. ON2): 45 mins
Online smothing run time est (excl. ON2): 45 mins
"""
MV_OU = {
    'sde_cls': sdes.MvOrnsteinUhlenbeck,
    'cdssm_cls': GaussianCDSSM,
    'sde_params': {'dimX': 2,
                'rho': 0.2*np.ones((1, 2)),
                'mu': np.zeros((1, 2)),
                'phi': 0.3*nla.cholesky(np.array([[1., 0.9,], [0.9, 1.]])),
                },
    'cdssm_params': {'x0': np.zeros((1, 2)),
                    'delta_s': 1., 
                    'G': np.eye(2),
                    'covY': (0.01 ** 2) * np.eye(2)
                    },
    'default_T': 10,
    'seed': 34953,
    'fk_names': ['BsR_DH', 'BwR_DH_OUP', 'BwR_DH_NDBBrP', 'FwR_DH_OUP', 'FwR_DH_DBrP', 'FwR_DH_NDBBrP']
    }

# -----------------------------------------------------------------------------------------
# ------------------IntegratedOrnsteinUhlenbeck + GaussianCDSSM ---------------------------
"""
Filtering run time est: 6h 40m
Smoothing run time est (excl. ON2): 9h 20m
Online smothing run time est (excl. ON2): 9h 20m
"""

IOU = {
    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
    'cdssm_cls': GaussianCDSSM,
    'sde_params': {'dimX': 4,
                    'rho': 0.2*np.ones((1, 2)),
                    'mu': np.zeros((1, 2)),
                    'phi': 0.3*nla.cholesky(np.array([[1., 0.9,], [0.9, 1.]])), # correlation between rough components
                    },
    'cdssm_params': {'x0': np.zeros((1, 4)),
                    'delta_s': 1., 
                    'G': np.concatenate([np.eye(2), np.zeros((2, 2))], axis=1), # only smooth component observed
                    'covY': (0.01 ** 2) * np.eye(2) # low noise regime
                    },
    'default_T': 100,
    'seed': 34953,
    # 'fk_names': ['BootstrapDA', 'BwG_OU_OUP', 'BwR_OU_OUP', 'BwG_NDBr_OUP', 'BwR_NDBr_OUP', 'BwG_OU_NDBBrP', 'BwR_OU_NDBBrP']
    'fk_names': ['BwR_OU_OUP', 'BwR_NDBr_OUP', 'BwR_OU_NDBBrP']
    }

#-----------------------------------------------------------------------------------------
#---------------------IntegratedBrownianMotion + GaussianCDSSM ---------------------------
"""
Filtering run time est: 7h

fk_names:
------------
In this example, the 'matching condition' implies that when constructing backward proposals,
unless we add some drift for no reason, we must 'always' use an exact diffusion bridge proposal.
"""
IBM = {
    'sde_cls': sdes.IntegratedBrownianMotion,
    'cdssm_cls': GaussianCDSSM,
    'sde_params': {'dimX': 2,
                    'm': np.zeros((1, 1)),
                    's': 0.3*np.ones((1, 1)), 
                    },
    'cdssm_params': {'x0': np.zeros((1, 2)),
                    'delta_s': 1.,
                    'G': np.eye(2), # both components observed
                    'covY': (0.01 ** 2) * np.eye(2) # low noise regime
                    },
    'default_T': 100,
    'seed': 34953,
    # 'fk_names': ['BootstrapDA', 'BwG_NDBr_DBrP', 'BwR_NDBr_DBrP', 'BwG_NDBr_NDBBrP', 'BwR_NDBr_NDBBrP']
    'fk_names': ['BwR_NDBr_DBrP', 'BwR_NDBr_NDBBrP']
    }

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


CDSSM_LIB = {'ts_mv_ou': TS_MV_OU,
             'mv_ou': MV_OU,
             'iou': IOU,
             'ibm': IBM}

def build_cdssm(cdssm_spec_name):
    """
    cdssm_spec_name: str
    """
    cdssm_spec = CDSSM_LIB[cdssm_spec_name]

    # Extract objects from the CDSSM Spec:
    sde_cls = cdssm_spec['sde_cls']
    cdssm_cls = cdssm_spec['cdssm_cls']
    sde_params = cdssm_spec['sde_params']
    cdssm_params = cdssm_spec['cdssm_params']
    T = cdssm_spec['default_T']

    # We define the underlying SDE:
    sde = sde_cls(**sde_params)

    # We define the CDSSM:
    cdssm = cdssm_cls(sde, **cdssm_params)
    return cdssm

# We may want to use this covariance matrix to create a mv_ou_4d cdssm spec:
# phi = 0.3 * nla.cholesky(np.array([[1., 0.9, 0.8, 0.5], 
#                                     [0.9, 1., 0.6, 0.4],
#                                     [0.8, 0.6, 1., 0.6],
#                                     [0.5, 0.4, 0.6, 1.]]))
