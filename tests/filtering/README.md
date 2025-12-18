# Filtering Test

The folder contains a test that ensures that the continuous-discrete particle filters developed within this package are all running correctly:

## Running the test:

To run the test, run the shell script via command:

`source run_filtering_tests.sh`

The running of the test may take some time, around 30 minutes.

## Run an individual test

To run the test on a single CD-SSM and visualise the results of the test, run (e.g for `mv_ou`):

`python filtering_test.py 10 mv_ou`

Then run the notebook `filtering_test_results.ipynb` to review the results.

## Test details

The test considers 6 different CD-SSMs: 2 from each of the 3 types of SDE: `SDE`, `MvEllipticSDE`, `IntegratedSDE`:

- `ou`: 1-D Ornstein Uhlenbeck
- `mv_ou` 2-D (MV) Ornstein Unlenbeck
- `iou` 2-D Integrated Ornstein Uhlenbeck
- `bm` 1-D Brownian Motion
- `mv_bm` 2-D (MV) Brownian Motion
- `ibm` 2-D Integrated Brownian Motion

All SDEs are assumed to be observed at equidistant points in time with Gaussian distributed additive noise.

For each of the CD-SSMs: the following steps happen:

- Synthetic data is generated of length $T=10$
- The Kalman filter is run to recover the true value of $p(y_{1:10})$ for the model.
    - (Note that since the SDEs are linear, the transition density is available, and is also a linear, Gaussian update)
- All Feynman-Kac constructions that is possible to build with the tools developed in this package is constructed for the given CD-SSM, using different choices of forward and backward proposal, and the different Feynman-Kac formalisms in `feynman_kac.py`: recall that these formalisms are:
    - `BootstrapDA`
    - `BootstrapReparmaeterisedDA`
    - `ForwardGuidedDA`
    - `ForwardReparameterisedDA`
    - `BackwardGuidedDA`
    - `BackwardReparmaeterisedDA`

- The particle filter is run 96 times on each Feynman-Kac model, and the estimator $\hat{p}(y_{1:10})$ is stored for each run.
- The MAE of the estimator: $E[|\hat{p}(y_{1:T}) - p(y_{1:T})]$ is calculated with confidence intervals for each of the Feynman-Kac models.

The test to evaluate the performance is then to check the ordering of the performance of the different Feynman-Kac constructions: we adopt a low noise regime for all CD-SSMs, thus we expect the Bootstrap particle filter to be the worst performing. Using a form of guided proposal will improve the performance. Then, by using a better choice of Linear SDE in the choice of proposal, we further improve performance. The expected best and worst performing Feynman-Kac models are recorded for each of the 6 CD-SSMs in `cdssm_lib.py`. 

