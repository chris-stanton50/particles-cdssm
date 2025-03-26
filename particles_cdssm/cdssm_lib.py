import numpy as np
import numpy.linalg as nla
import particles_cdssm.sdes as sdes
from particles_cdssm.continuous_discrete_ssms import GaussianCDSSM, MvGaussianCDSSM
import particles.state_space_models as ssms
import particles.distributions as dists
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


"""
We use this module to define the CDSSM specifications that we will use in our experiments.

- For filtering experiments, we are interested in evaluating the performance of guided particle
filters in the low noise regime. 

- For smoothing experiments, we are interested in evaluating the performance of smoothing methods
as the number of observations $T$ increases.
"""
# Note: Run times are based on the local: a trial run indicates that the remote is ~40% faster 
# than the local. (This is based on the MV_OU model for filtering.)
#----------------------------------------------------------------------------------------------
# ---------------------------- OrnsteinUhlenbeck + GaussianCDSSM ------------------------------

OU = {
    'sde_cls': sdes.OrnsteinUhlenbeck,
    'cdssm_cls': GaussianCDSSM,
    'filtering_sde_params': {'rho': 1.0, 'mu': 0., 'phi': 1.0},
    'smoothing_sde_params': {'rho': 0.01, 'mu': 0., 'phi': 0.01},
    'filtering_cdssm_params': {'x0': 0., 's_ts': 1., 'sigmaY': 0.1},
    'smoothing_cdssm_params': {'x0': dists.Normal(loc=0., scale=1.), 's_ts': 1., 'sigmaY': 1.0},
    'filtering_fk_names': ['BootstrapDA', 'BsR_DH', 'BwG_OU_OUP', 'BwR_OU_OUP', 'BwG_DH_OUP', 'BwR_DH_OUP', 'FwG_OUP', 'FwR_DH_OUP'],
    'smoothing_fk_names': ['BootstrapDA', 'BsR_DH', 'FwG_OUP', 'FwR_DH_OUP', 'BwG_DH_DBrP', 'BwR_DH_DBrP'],
    'default_T': 10,
    'seed': 34953,
    }

#------------------------------------------------------------------------------------------
# ----------------------- MvOrnsteinUhlenbeck + MvGaussianCDSSM -----------------------------
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
    'cdssm_cls': MvGaussianCDSSM,
    'filtering_sde_params': {'dimX': 2,
                'rho': 1.0*np.ones((1, 2)),
                'mu': np.zeros((1, 2)),
                'phi': 1.0*np.eye(2),
                },
    # 'smoothing_sde_params': {'dimX': 2,
    #             'rho': np.array([0.01, 1.0]).reshape((1, 2)),
    #             'mu': np.zeros((1, 2)),
    #             'phi': np.array([[0.001, 0.], [0., 1.0]]),
    #             },
    'smoothing_sde_params': {'dimX': 2,
                'rho': np.array([1.0, 1.0]).reshape((1, 2)),
                'mu': np.zeros((1, 2)),
                'phi': np.array([[1.0, 0.], [0., 1.0]]),
                },
    'filtering_cdssm_params': {'x0': np.zeros((1, 2)),
                    's_ts': 1., 
                    'G': np.eye(2),
                    'covY': 0.1 * np.eye(2)
                    },
    'smoothing_cdssm_params': {'x0': dists.MvNormal(loc=np.zeros((1, 2)), cov=1.*np.eye(2)),
                    's_ts': 1.,
                    'G': np.eye(2),
                    'covY': np.array([[1., 0.], [0., 0.1]])
                    },
    'filtering_fk_names': ['BootstrapDA', 'BsR_DH', 'BwG_OU_OUP', 'BwR_OU_OUP', 'BwG_DH_OUP', 'BwR_DH_OUP', 'FwG_OUP', 'FwR_DH_OUP'],
    'smoothing_fk_names': ['BootstrapDA', 'BsR_DH', 'BwG_DH_DBrP', 'BwR_DH_DBrP'],
    # 'smoothing_fk_names': ['BootstrapDA', 'BsR_DH'],
    # 'smoothing_fk_names': ['BootstrapDA', 'BsR_DH', 'FwG_OUP', 'FwR_DH_OUP', 'BwG_DH_DBrP', 'BwR_DH_DBrP'],
    'default_T': 10,
    'seed': 34953,
}

#---------------------------------------------------------------------------------------------
#---------------------IntegratedOrnsteinUhlenbeck + MvGaussianCDSSM---------------------------
"""
This SDE is a special case of the SDEs considered in Hairer et al (2011). The existence 
of a continuous-time likelihood for a hypoelliptic diffusion bridge is shown in Bierkens et al (2020).
Can be considered as the time evolution of a state of a mechanical system with friction under the 
influence of noise.

We add a mean term to the OU process for the velocity, so that the model is realistic.
We use particle filters to infer the true values of the latent process.

Filtering experiment: (Do not include in the paper) 
----------------------

We observe the position of the particle with low noise at discrete times. We want to infer at each time t, how fast the particle is moving.

We set higher noise in the velocity (rough) component. We observe the particle displacement with low noise.
We are interested in inferring the velocity at each time t.

- We initialise the displacement and the velocity with low noise.

Smoothing Experiment:
----------------------

We observe the position of the particle with high noise at discrete times. We know that the particle is moving at (approx.) a constant velocity.
We know, approximately, the velocity of the particle. We want to infer the position of the particle at time 0.

We set low noise in the velocity (rough) component. We initialise the displacement with high noise, and the velocity with low noise.
We observe just the displacement with high noise. We are interested in inferring the displacement at time 0.

We may need to play around with the parameters to get the experiment to work as intended.

(e.g decrease the mean reversion/noise in the velocity OU process, and increase the noise in the position OU process.)
Smoothing run time est: 7h
"""
IOU = {
    'sde_cls': sdes.IntegratedOrnsteinUhlenbeck,
    'cdssm_cls': MvGaussianCDSSM,
    'filtering_sde_params': {'dimX': 2,
                    'rho': 1.0*np.ones((1, 1)),
                    'mu': 1.0*np.ones((1, 1)),
                    'phi': 0.2*np.ones((1, 1)),
                    },
    'smoothing_sde_params': {'dimX': 2,
                    'rho': 1.0*np.ones((1, 1)),
                    'mu': 1.0*np.ones((1, 1)),
                    'phi': 0.1*np.ones((1, 1)),
                    },
    'filtering_cdssm_params': {
                    'x0': np.zeros((1, 2)),
                    's_ts': 1.,
                    'G': np.array([1., 0.]).reshape((1, 2)), # smooth component observed
                    'covY': (0.01 ** 2) * np.eye(1) # low noise regime
                    },
    'smoothing_cdssm_params': {'x0': dists.MvNormal(loc=np.array([0., 1.]).reshape((1, 2)), cov=0.1*np.eye(2)),
                    's_ts': 1.,
                    'G': np.array([1., 0.]).reshape((1, 2)), # smooth component observed
                    'covY': (0.01 ** 2) * np.eye(1) 
                    },
    'filtering_fk_names': ['BootstrapDA', 'BsR_DH', 'BwG_OU_OUP', 'BwR_OU_OUP', 'BwG_DH_OUP', 'BwR_DH_OUP'],
    'smoothing_fk_names': ['BootstrapDA', 'BwG_NDOU_OUP', 'BwR_NDOU_OUP'],
    'default_T': 5,
    'seed': 34953,
    }


#-----------------------------------------------------------------------------------------
#---------------------IntegratedFitzHughNagumo + GaussianCDSSM ---------------------------

CDSSM_LIB = {'ou': OU,
             'mv_ou': MV_OU,
             'iou': IOU,
            #  'ifhn': IFHN # This is an example of parameter inference.
             }

def build_cdssm(cdssm_spec_name, smoothing):
    """
    Inputs:
    -------
    cdssm_spec_name: str
    smoothing: (bool): True = smoothing, False = filtering

    Rewturns:
    --------
    cdssm: A cdssm object defined by the given CDSSM Spec.
    """
    cdssm_spec = CDSSM_LIB[cdssm_spec_name]

    # Extract objects from the CDSSM Spec:
    sde_cls = cdssm_spec['sde_cls']
    cdssm_cls = cdssm_spec['cdssm_cls']
    filt_smth_str = 'smoothing' if smoothing else 'filtering'
    
    sde_params = cdssm_spec[f'{filt_smth_str}_sde_params']
    cdssm_params = cdssm_spec[f'{filt_smth_str}_cdssm_params']
    T = cdssm_spec['default_T']

    # We define the underlying SDE:
    sde = sde_cls(**sde_params)

    # We define the CDSSM:
    cdssm = cdssm_cls(sde, **cdssm_params)
    return cdssm