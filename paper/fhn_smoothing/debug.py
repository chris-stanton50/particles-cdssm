import numpy as np
from particles_cdssm.sdes import MvEllipticSDE, IntegratedSDE


class FitzHughNagumo(MvEllipticSDE):
    """
    Stochastic Fitzhugh-Nagumo model, as in Samson et al (2025).
    Not an elliptic SDE.
    
    https://www.sciencedirect.com/science/article/pii/S0167947324001798?via%3Dihub

    dV_t = 1/epsilon * (V_t - V_t^3 - U_t) dt
    dU_t = gamma * V_t - U_t + beta dt + sigma_u dB_t
    """
    dimX = 2
    default_params = {'epsilon': 0.1,
                      'gamma': 1.5,
                      'beta': 0.8,
                      'sigma_u': 0.3
                    }

    def b(self, t, x):
        x_1 = 1/self.epsilon * (x[:, 0] - (x[:, 0] ** 3) - x[:, 1]) # (N, )
        x_2 = self.gamma * x[:, 0] - x[:, 1] + self.beta # (N, )
        return np.stack([x_1, x_2], axis=1)

    def sigma(self, t, x):
        N = x.shape[0]
        return np.stack([np.array([[0.0, 0.0], [0.0, self.sigma_u]])]*N, axis=0)
    
class IntegratedFitzhughNagumo(IntegratedSDE):
    """
    Integrated version of the FitzHugh-Nagumo model:
    The recovery variable has been reparameterised so that the resulting diffusion process is 
    an integrated SDE:

    dX_{1, t} = X_{2,t} dt
    dX_{2, t} = (1-\epsilon - 3 X_{1,t}^2)X_{2,t} dt + [(1-\gamma)X_{1,t} - X_{1,t}^3  - (\beta)]dt - \frac{\sigma}{\epsilon} dB_t


    X_1: Action potential
    X_2: First derivative of action potential

    The parameters are:

    - epsilon: the stiffness of the particle
    - gamma: the damping coefficient
    - beta: the external force applied to the particle
    - sigma_u: the noise intensity
    """
    dimX = 2
    default_params = {'epsilon': 0.1,
                      'gamma': 1.5,
                      'beta': 0.8,
                      'sigma_u': 0.3
                    }    

    def b_rough(self, t, x):
        x_2 =  (1 - self.epsilon - 3*np.square(x[:, 0])) * x[:, 1] # (N, )
        x_2 += (1 - self.gamma) * x[:, 0] - np.power(x[:, 0], 3) - self.beta
        return x_2.reshape(-1, 1) # (N, 1)

    def sigma_rough(self, t, x):
        N = x.shape[0]
        return self.sigma_u/self.epsilon * np.ones((N, 1, 1))  # (N, 1, 1)

    # def db_rough(self, t, x):
    #     x_2 =  -2. * self.alpha * x[:, 1] # (N, )
    #     return x_2.reshape(-1, 1) # (N, 1)
    
    # def dsigma_rough(self, t, x):
    #     N = x.shape[0]
    #     return np.zeros((N, 1, 1, 1)) # (N, 1, 1, 1)

sde = IntegratedFitzhughNagumo()
samples = sde.simulate(10, t_start=0., t_end=0.025, x_start= np.array([[0.0, 0.0]]), num=1000)
