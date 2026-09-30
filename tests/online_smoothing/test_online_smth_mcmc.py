import particles
import numpy as np
import particles.state_space_models as ssms
from particles.kalman import MVLinearGauss, LinearGauss, Kalman
from particles_cdssm.collectors import Online_smooth_MCMC

import particles_cdssm.sdes as sdes
import particles_cdssm.continuous_discrete_ssms as cdssms
import particles_cdssm.feynman_kac as sfk
from particles_cdssm import CDSSM_SMC

"""
Possible extensions of these two tests:
- Iterate over:
    - Parameterisations of the SSMs
    - Length of the dataset T
    - Choice of additive functional 
"""

def lgssm_1d_test():    
    # Define a 1-D LG-SSM with an additive function that takes the sum of the means
    class LinearGauss_w_add_func(LinearGauss):
    
        def add_func(self, t, xp, x):
            return x
    
    ssm = LinearGauss_w_add_func(rho=0.9, sigmaX=1.0, sigmaY=0.5, sigma0=1.0)
    # Simulate data for the ssm
    T = 100
    x, y = ssm.simulate(T)
    
    # Extract true value of integral using RTS smoother
    kf = Kalman(ssm=ssm, data=y)
    kf.filter()
    kf.smoother()
    true_mean_sum = sum([kf.smth[i].mean[0] for i in range(T)])

    # Obtain estimate using MCMC-based online smoothing
    # Define a FK model and SMC algorithm
    fk = ssms.Bootstrap(ssm=ssm, data=y)
    # Create SMC object
    smc_kwargs =dict(fk=fk,
                        N=10000,
                        qmc=False,
                        resampling="systematic",
                        ESSrmin=1.0,
                        store_history=False,
                        verbose=False,
                        collect=[Online_smooth_MCMC(nsteps=3)],
                    ) 
    smc = particles.SMC(**smc_kwargs)
    # Run the algorithm
    smc.run()
    est_mean_sum = smc.summaries.online_smooth_MCMC[-1]
    
    # Test whether the values are close
    assert abs(true_mean_sum - est_mean_sum) < 1.0
    print(f'LGSSM 1-D test passed. Est mean: {est_mean_sum}. True mean: {true_mean_sum}')
    
def lgssm_test():
    class MVLinearGauss_w_add_func(MVLinearGauss):
    
        def add_func(self, t, xp, x):
            return x[:, 0]

    lgssm = MVLinearGauss_w_add_func(covX=np.eye(2), covY=(0.5**2) * np.eye(2))
    T = 100
    x, y = lgssm.simulate(T=T)

    # Extract true value of integral using RTS smoother
    kf = Kalman(ssm=lgssm, data=y)
    kf.filter()
    kf.smoother()
    true_mean_sum = sum([kf.smth[i].mean[0][0] for i in range(T)])

    # Obtain estimate using MCMC-based online smoothing
    # Define a FK model and SMC algorithm
    fk = ssms.Bootstrap(ssm=lgssm, data=y)
    smc_kwargs =dict(fk=fk,
                        N=30000,
                        qmc=False,
                        resampling="systematic",
                        ESSrmin=1.0,
                        store_history=False,
                        verbose=False,
                        collect=[Online_smooth_MCMC(nsteps=3)],
                    )
    smc = particles.SMC(**smc_kwargs)
    # Run the algorithm
    smc.run()
    est_mean_sum = smc.summaries.online_smooth_MCMC[-1]

    # Test whether the values are close
    assert abs(true_mean_sum - est_mean_sum) < 2.0
    print(f'LGSSM test passed. Est mean: {est_mean_sum}. True mean: {true_mean_sum}')


def cdssm_1d_test():

    # Define a 1-D CD-SSM with an additive function that takes the sum of the means 
    # at the ending point of the paths
    class NormalCDSSM_w_add_func(cdssms.NormalCDSSM):
        
        def add_func(self, t, xp, x):
            return x[x.dtype.names[-1]]
    
    # Construct an OU CD-SSM
    ou_params = {'rho': 0.2,'mu': 0.,'phi': 1.}
    ou_sde = sdes.OrnsteinUhlenbeck(**ou_params)

    cdssm_params = {'x0': 0., 's_ts': 1., 'sigmaY': 0.05}
    ou_cdssm = NormalCDSSM_w_add_func(ou_sde, **cdssm_params)
    
    # Simulate a synthetic dataset
    T = 100
    x, y = ou_cdssm.simulate(T=T)

    # Extract true value of the integral using RTS smoother
    lgssm = ou_cdssm.lgssm()
    kf = Kalman(ssm=lgssm, data=y)
    kf.filter()
    kf.smoother()
    true_mean_sum = sum([kf.smth[i].mean[0] for i in range(T)])

    # Obtain estimate using MCMC-based pathspace online smoothing
    # Define a FK model and SMC algorithm
    fk = sfk.BootstrapReparameterisedDA(cdssm=ou_cdssm, data=y)

    cdssm_smc_kwargs = dict(fk=fk,
            N=100,
            resampling="systematic",
            ESSrmin=1.0,
            store_history=False,
            verbose=False,
            collect=[Online_smooth_MCMC(nsteps=3)],
            num=50)
    smc = CDSSM_SMC(**cdssm_smc_kwargs)
    smc.run()
    est_mean_sum = smc.summaries.online_smooth_MCMC[-1]

    # Test whether the values are close
    assert abs(true_mean_sum - est_mean_sum) < 2.0
    print(f'CD-SSM in 1D test passed. Est mean: {est_mean_sum}. True mean: {true_mean_sum}')

def cdssm_test():
    # Define a CD-SSM with an additive function that takes the sum of the means 
    # at the ending point of the paths
    class MvNormalCDSSM_w_add_func(cdssms.MvNormalCDSSM):
        
        def add_func(self, t, xp, x):
            return x[x.dtype.names[-1]][:, 0]
        
    # Construct an OU CD-SSM
    mv_ou_params = {'dimX': 2, 'rho': 1.0*np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': 1.0*np.eye(2)}
    mv_ou_sde = sdes.MvOrnsteinUhlenbeck(**mv_ou_params)

    cdssm_params = {'x0': np.zeros((1, 2)),'s_ts': 1., 'G': np.eye(2), 'covY': (0.1 ** 2) * np.eye(2)}
    mv_ou_cdssm = MvNormalCDSSM_w_add_func(mv_ou_sde, **cdssm_params)
    
    # Simulate a synthetic dataset
    T = 100
    x, y = mv_ou_cdssm.simulate(T=T)

    # Extract true value of the integral using RTS smoother
    lgssm = mv_ou_cdssm.lgssm()
    kf = Kalman(ssm=lgssm, data=y)
    kf.filter()
    kf.smoother()
    true_mean_sum = sum([kf.smth[i].mean[0][0] for i in range(T)])

    # Obtain estimate using MCMC-based pathspace online smoothing
    # Define a FK model and SMC algorithm
    fk = sfk.BootstrapReparameterisedDA(cdssm=mv_ou_cdssm, data=y)

    cdssm_smc_kwargs = dict(fk=fk,
            N=100,
            resampling="systematic",
            ESSrmin=1.0,
            store_history=False,
            verbose=False,
            collect=[Online_smooth_MCMC(nsteps=3)],
            num=50)
    smc = CDSSM_SMC(**cdssm_smc_kwargs)
    smc.run()
    est_mean_sum = smc.summaries.online_smooth_MCMC[-1]

    # Test whether the values are close
    assert abs(true_mean_sum - est_mean_sum) < 2.0
    print(f'CD-SSM test passed. Est mean: {est_mean_sum}. True mean: {true_mean_sum}')
    
if __name__ == '__main__':
    lgssm_1d_test()
    lgssm_test()
    cdssm_1d_test()
    cdssm_test()