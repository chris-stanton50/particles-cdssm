# Particle Based Inference for Continuous-Discrete State Space Models (CD-SSMs) - `particles_cdssm`

A Python package to implement Sequential Monte Carlo methods for inference in [Continuous-Discrete State Space Models.](https://arxiv.org/abs/2407.15666v1#) Built on top of the [particles](https://github.com/nchopin/particles/tree/master) package.    


```
@article{stanton2024particle,
  title={Particle Based Inference for Continuous-Discrete State Space Models},
  author={Stanton, Christopher and Beskos, Alexandros},
  journal={arXiv preprint arXiv:2407.15666},
  year={2024}
}
```

# Difference between a State Space Model and a Continuous Discrete State Space model:

Example of both for Stochastic Volatility:

### Stochastic Volatility State Space Model


$$X_0 \sim N\left(\mu, \frac{\sigma^2}{1-\rho^2}\right)$$
$$X_t|X_{t-1}=x_{t-1} \sim N\left( \mu + \rho (x_{t-1}-\mu), \sigma^2\right), \quad t\geq 1$$
$$Y_t|X_t=x_t \sim N\left(0, e^{x_t}\right) \quad t\geq 0.$$


### Stochastic Volatility Continuous-Discrete State Space Model


$$\textrm{Initial Distribution:} \quad X(0) \sim \mathcal{N}(\mu, \frac{\phi^2}{2\rho})$$
$$\textrm{Continuous Latent State process}: \quad dX(s) = \rho(\mu - X(s))ds + \phi dB(s) ,\quad s \in [0, S]$$
$$\textrm{Observed at Discrete Times:} \quad Y_t | X(t) = x(t) \sim \mathcal{N}(0, e^{x(t)}) ,\quad t\in \{0, 1, 2, \dots, T-1\}.$$

The continuous-discrete state space model allows one to define volatility continuously throughout the trading day, instead of only having one value of volatility for each trading day. 

# Difference between this package and the `particles` package:


In the particles package, we could run a particle filter on a state space model as follows:

```python
import particles
import particles.state_space_models as ssms

class StochVol(ssms.StateSpaceModel):
    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.mu, scale=self.sigma / np.sqrt(1. - self.rho**2))
    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=0., scale=np.exp(0.5 * x))
   
sv_ssm = StochVol(mu=-1., rho=.95, sigma=.1)  # actual model

x, y = sv_ssm.simulate(100)

fk_model = ssms.Bootstrap(ssm=sv_ssm, data=y)
alg = particles.SMC(fk=fk_model, N=100)

alg.run()
```

In the particles-cdssm package, we can run a particle filter for a continuous-discrete state space model as follows:

```python
import particles.distributions as dists

import particles_cdssm
import particles_cdssm.sdes as sdes
import particles_cdssm.continuous_discrete_ssms as cdssms
import particles_cdssm.feynman_kac as cdfk


rho=1.0; mu=-1.0; phi=0.2  # parameters of the continuous process

# Initial distribution
initial_dist = dists.Normal(loc=mu, scale=phi/np.sqrt(2*rho))  

# Define the Ornstein Uhlenbeck SDE class
class OrnsteinUhlenbeck(sdes.SDE):
    
    default_params = {'rho': 1.0, 'mu': -1.0, 'phi': 0.1} 
    
    def b(self, t, x):  # Drift function
        return self.rho * (self.mu - x)
    
    def sigma(self, t, x):  # Diffusion function
        return self.phi

ou_sde = OrnsteinUhlenbeck(rho=rho, mu=mu, phi=phi)  # Ornstein-Uhlenbeck SDE    

# Define the observation density as a method of CDSSM:

class StochVolCDSSM(cdssms.CDSSM):
    
    def PY(self, t, xp, x):
        return dists.Normal(loc=0., scale=np.exp(0.5 * x))

sv_cdssm = StochVolCDSSM(ou_sde, x0=initial_dist) # Stochastic volatility CDSSM:

x, y = sv_cdssm.simulate(100)
cd_fk_model = cdfk.BootstrapDA(cdssm = sv_cdssm, data=y)

alg = particles_cdssm.CDSSM_SMC(fk=cd_fk_model, N=100)
alg.run()
```

## Some interesting applications for CD-SSMs:

- Population Modelling (e.g Malthus/Verhulst)
- Biosciences (e.g Lotka-Volterra)
- Neuroscience (e.g Fitzhugh-Nagumo/Duffing-Van-Der-Pol)
- Finance (Black-Scholes-Merton/Heston/Cox-Ingersoll-Ross)
 
## Features ##

- *Particle Filters*
    - Bootstrap Particle Filter
    - Guided Particle Filter (via Forward and Backward proposals)

- *Forward Filtering Backward Sampling*
    - FFBS (Standard $\mathcal{O}(N^2)$ version)
    - FFBS-MCMC (Linear cost $\mathcal{O}(N)$ version)

Bayesian parameter estimation via Particle MCMC and online smoothing algorithms are under development for next release. That's all for now: check out the docs if this interests you!

<!-- 
# Future Development
    - Particle MCMC-based smoothers
    - iCMSC
    - iCSMC-MCMC

## Joint Offline Smoothing ($X_{1:t}, \theta | Y_{1:t}=y_{1:t}$)

- Particle MCMC
    - Particle Marginal Metropolis Hastings (PMMH)
    - Particle Gibbs (PG)
    - Particle Gibbs with Backward Step (PGBS) -->

<!-- ## Online Smoothing for Additive Functionals ($X_{1:t} | Y_{1:t}=y_{1:t}$)

- Forward Additive- $\mathcal{O}(N^2)$
- Forward Additive-MCMC -->


<!-- ## Joint Online Smoothing ($X_{1:t}, \theta | Y_{1:t}=y_{1:t}$)

- $SMC^2$ -->


# Contact

This package is under active development - for any issues, bugs or frustrations, please do submit an issue, or alternatively you can e-mail me (christopher.stanton.20@ucl.ac.uk).

# Setup Instructions

### venv

Reproducibility for this repo is managed through a virtual environment. After the cloning the repository, build the virtual environment locally through the following commands:

(Ensuring that `python` refers to Python 3.11.10 (if not, on Mac brew install python3.11, then use `python3.11` instead)), inside the directory of the cloned repository: 

- Make a folder for the venv: `mkdir venv` (the folder venv is already in the `.gitignore` file)
- Create the venv `python -m venv ./venv/particles_cdssm`
- Activate the venv by sourcing the activate script: `source ./venv/diffusions/bin/activate`
- Upgrade pip in your virtual environment `pip install --upgrade pip`
- Install all dependencies in the venv `pip install -r requirements.txt`

All of the above steps can be run by sourcing the `build_venv.sh` script.

To delete the created virtual environment, simply decactivate it then recursively delete the folder in which the packages were created.

`rm -rf venv`

### Add repo to the Python path

Currently, the repo has not been made into a package using a wheel file. So, after cloning the repository, the location of the repository needs to be added to the python path to import modules from the project:
To do this, run the following shell command in the terminal, or add the following line to the `.zshrc` (Mac) or `.bashrc` (Linux) file:

 `export PYTHONPATH=$PYTHONPATH:~/location/of/cloned_repository/particles-cdssm/`

You are now ready to use the package. Ensure that you have the created `particles-cdssm` virtual environment activated when trying to use the package. It can be activated with the following command from the home directory:

`source ./venv/particles-cdssm/bin/activate`

You could alias this for faster activation with the command:

`alias activate_particles_cdssm="source ~/path/from/home/to_repository/particles-cdssm/venv/diffusions/bin/activate"`