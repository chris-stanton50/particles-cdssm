import numpy as np
import matplotlib.pyplot as plt
import particles_cdssm.sdes as sdes

import particles_cdssm
import particles_cdssm.feynman_kac as cdfk
import seaborn as sb
import particles_cdssm.plot as splt
from particles_cdssm.continuous_discrete_ssms import NormalCDSSM, MvNormalCDSSM


class LotkaVolterra(sdes.MvEllipticSDE):
    """
    The stochastic Lotka-Volterra model for predator-prey dynamics.
    
    dX_{1, t} = \alpha X_{1, t} - \beta X_{1, t}X_{2, t}dt + s_1 dW_{1, t}
    dX_{2, t} = \delta X_{1, t}X_{2, t} -\gamma X_{2, t}dt + s_2 dW_{2, t}

    X_1: Prey population,
    X_2: Predator population
    
    \alpha : prey growth rate,
    \beta: predation rate,
    \delta: predator reproduction rate per prey consumed,
    \gamma: predator mortality rate,
    s_1, s_2 :noise intensities for prey and predator.
    
    Note: Good initial values for the process are:
    X_1 = 50, X_2 = 10
    """
    dimX = 2
    default_params = {'alpha': 0.1,
                      'beta': 0.02,
                      'delta': 0.01,
                      'gamma': 0.5,
                      's': np.array([0.1, 0.1])}
    
    def b(self, t, x):
        x_1 = self.alpha * x[:, 0] - self.beta * x[:, 0] * x[:, 1] # (N, )
        x_2 = self.delta * x[:, 0]*x[:, 1] - self.gamma * x[:, 1] # (N, )
        return np.stack([x_1, x_2], axis=1)

    def sigma(self, t, x):
        diag_sigma = x*self.s # (N, 2)        
        return np.einsum('ni,ij->nij', diag_sigma, np.eye(2))
    
    def db(self, t, x):
        db_1_dx_1 = self.alpha - self.beta * x[:, 1]  # d/dx_1 of b_1 (N, )
        db_1_dx_2 = -self.beta * x[:, 0]  # d/dx_2 of b_1 (N, ) 
        db_2_dx_1 = self.delta * x[:, 1]  # d/dx_1 of b_2 (N, )
        db_2_dx_2 = self.delta * x[:, 0] - self.gamma  # d/dx_2 of b_2 (N, )
        # Jacobian as a 2D array (2, 2, N)
        db = np.array([[db_1_dx_1, db_1_dx_2], [db_2_dx_1, db_2_dx_2]])
        # Jacobian as a 2D array (N, 2, 2)
        db = np.einsum('ijk->kij', db)
        return db

model_sde = LotkaVolterra()

cdssm_params = {'x0': np.array([50., 10.]).reshape(1, -1), 'G': np.eye(2), 'covY': (1.0 ** 2) * np.eye(2), 's_ts': 10.}
cdssm = MvNormalCDSSM(model_sde, **cdssm_params)

x, y = cdssm.simulate(100)

bootstrap_fk = cdfk.BootstrapDA(cdssm, data=y)
guided_fk = cdfk.ForwardGuidedDA(cdssm, data=y)

# forward_proposal = guided_fk._build_forward_proposal(0, np.array([50., 10.]).reshape(1, -1))
# forward_proposal.b(0, np.array([[50., 10.]]))

# # Bootstrap
# alg = particles_cdssm.CDSSM_SMC(N=100, fk=bootstrap_fk, num=100)
# alg.run()

# Guided
alg = particles_cdssm.CDSSM_SMC(N=100, fk=guided_fk, num=100)
alg.run()