# Particle Based Inference for Continuous-Discrete State Space Models (CD-SSMs) - `particles_cdssm`

A Python package to implement Sequential Monte Carlo methods for inference in Continuous-Discrete State Space Models. Built on top of the `particles` package, can also be used as an alternative to packages for simulation of SDE paths using numerical integrators (such as `sdeint`).    

This project was initally developed to implement the methods outlined the paper [**Particle Based Inference for Continuous-Discrete State Space Models**](https://arxiv.org/abs/2407.15666v1#). As the project progressed, the code was extended to have an API to make the developed methods more accessible to an end user - the end result is this `particles_cdssm` package. Please cite the following work if using these methods for your research:

```
@article{stanton2024particle,
  title={Particle Based Inference for Continuous-Discrete State Space Models},
  author={Stanton, Christopher and Beskos, Alexandros},
  journal={arXiv preprint arXiv:2407.15666},
  year={2024}
}
```

## What does this package do?

This package provides implementations of numerical methods (namely, Sequential Monte Carlo methods) to conduct Bayesian filtering, smoothing and forecasting on a class of models that we call 'Continuous-Discrete State Space models' (CD-SSMs). Loosely speaking, a CD-SSM is a model in which a latent continuous-time process is observed at discrete points in time, with noise. In particular, the continuous-time process that is not observed is the solution of a **Stochastic Differential Equation** (SDE), also known as a diffusion processes.

## What are the numerical methods implemented to achieve this?

We use Sequential Monte Carlo methods for inference. This class of methods is now around 25 years old and has established itself as a state-of-the-art approach for inference in State Space Models. For those unfamiliar with this class of methods, we refer to the book-length treatment [An Introduction to Sequential Monte Carlo](https://link.springer.com/book/10.1007/978-3-030-47845-2). A package to implement these methods accompanies the book: [particles](https://github.com/nchopin/particles).

## Why would I want to use SMC methods for inference in CD-SSMs?

There are many real-world applications for which it is natural to use a stochastic dynamical system as the model: here are a few examples:

- Population Modelling (e.g Malthus/Verhulst)
- Biosciences (e.g Lotka-Volterra)
- Neuroscience (e.g Fitzhugh-Nagumo/Duffing-Van-Der-Pol)
- Finance (Black-Scholes-Merton/Heston/Cox-Ingersoll-Ross)

Furthermore, for these applications, K-step ahead forecasting is often the main goal of the modelling exercise. For example, we might want to predict the population in K years time, or we may want to predict a stock price the next day. The online nature of particle filters makes them ideal for cross-validating forecasting performance of these models. It is much less efficient computationally to attempt this using offline inference methods such as MCMC. Furthermore, generated predictions are inherently probabilistic, so we get uncertainty quantification for free.

## Can I not just use the particles package instead of this one?

There are challenges associated with using SMC methods to conduct inference when the latent states come from a diffusion observed at discrete times: the most problematic one is that the transition denity between the latent states is intractable for all but the simplest of models. This package implements SMC methods that are able to successfully able to overcome this and other issues, using an approach originally formulated in the context of Markov Chain Monte Carlo (MCMC) methods known as **Data Augmentation**. 

For further details on the issues involved with using SMC for CD-SSMs and how it is possible to implement particle-based methods despite these issues, see the publication that motivated the development of this package [Particle Based Inference for Continuous-Discrete State Space Models](https://arxiv.org/abs/2407.15666v1#) and the references therein.
 
## What SMC algorithms are implemented in this package?

Below is the full list of SMC methods that one can implement for inference in the class of CD-SSMs. Sequential Monte Carlo methods can be used for a broad range of inferential objectives.

## Online Filtering ($X_t | Y_{1:t}=y_{1:t}$)

- Particle Filters
    - Bootstrap Particle Filter
    - Guided Particle Filter
    - Auxiliary Particle Filter

## Offline Smoothing ($X_{1:t} | Y_{1:t}=y_{1:t}$)

- Particle-based smoothers 
    - FFBS
    - FFBS-MCMC 
- Particle MCMC-based smoothers
    - iCMSC
    - iCSMC-MCMC

## Online Smoothing for Additive Functionals ($X_{1:t} | Y_{1:t}=y_{1:t}$)

- Forward Additive- $\mathcal{O}(N^2)$
- Forward Additive-MCMC

## Joint Offline Smoothing ($X_{1:t}, \theta | Y_{1:t}=y_{1:t}$)

- Particle MCMC
    - Particle Marginal Metropolis Hastings (PMMH)
    - Particle Gibbs (PG)
    - Particle Gibbs with Backward Step (PGBS)

## Joint Online Smoothing ($X_{1:t}, \theta | Y_{1:t}=y_{1:t}$)

- $SMC^2$

That's all for now! Tutorials to introduce you to the API and more formal documentation are on the way! A user that is already familiar with the `particles` package will hopefully find it intuitive.

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