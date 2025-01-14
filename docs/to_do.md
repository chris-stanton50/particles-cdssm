Focus now on what needs to be done to finish the paper!

## High Priority

To do tomorrow morning: 

- Run the results for the filtering hypoelliptic case: 2D integrated bm.
- Get some results ready to show Alex, but don't make this the focus of the meeting.
- Only then do the rest of the stuff.

### Run methods

- Re-run the results for the `MvEllipticSDEs` `MvOrnsteinUhlenbeck` and `TS_MvOrnsteinUhlenbeck`. 
    -  `MvOrnsteinUhlenbeck`: N=100, T=100, n_runs=100 -f -s - os (without $N^2$ smoothing)
    - `TS_MvOrnsteinUhlenbeck`: N=100, T=10, n_runs=100 -f -s -os (without $N^2$ smoothing)
- Run the filtering for both a 2D/4D linear hypoelliptic example. 

### Check results

- Check that the results for 2 Elliptic cases are consistent with those obtained under the previous API.
- Check that the results from the hypoelliptic filtering are working.

Once all of the above is done, commit the changes and merge the changes from hypoelliptic into the master branch.


- *Note*: whe
- In the elliptic case, run an example of a long time series with high observation noise. In this example, offline smoothing methods will perform well, and when extended to the parameter, the PGBS will outperform PG and standard PMMH.   
- Debug an example of a hypoelliptic offline smoothing thing. 
- Check that the filtering results are consistent for the backward guided and backward reparameterised Feynman-Kac models in the elliptic case.
- Offline smoothing: check that the elliptic and hypoelliptic case run smoothly. 
- Write a script to evaluate the curse of dimensionality for particle filters in high dimensions.
- Implement the standard and integrated forms of the Fitzhugh-Nagumo model. We can use numerical simulation to check whether the derivation is correct. 
- Implement some simulation schemes for hypoelliptic diffusions (e.g the locally Gaussian scheme).

## Medium Priority

- The two files `sdes.py` and `feynman_kac.py` need to be cleaned up a bit after all of the changes that you have made. It may also be valuable to then clean up the `continuous_discrete_ssms.py`, the `state_space_models` file and the `numerical_schemes.py` file. 
- Create a `sdelib.py` file, that contains implementations of standard non-linear SDEs that appear in the literature.
- Idea: Create a `linearsdelib.py` file, that contains implementations of the standard linear SDEs that one should use when interacting with the library API. Think a bit about whether this is the best approach to take. 
- Extend the `MvLinearGauss` class from the particles package, so that it can deal with the case of non-zero drift in the latent state. Extend this change to the `MvDiscreteLinearGauss` class, so that one can use Kalman filter/RTS smoothers to benchmark performance of time varying SDEs and Hypoelliptic SDEs.     

- The API for univariate SDEs is a complete mess!

There are methods left over from when you tried to set things up so that you could evaluate the score function in the OrnsteinUhlenbeck class.
Also the mechanism to `generate_vec_params` and all the `_vec` methods is stupid an unintuitive. Do something about this!

Make some other changes to the API of the `sdes.py ` file. For example, have an `SDEBase` class, and have `SDE` inherit from it. `MvSDE` can also inherit from `SDEBase`.
`LinearSDE` can then inherit from `SDE`. It might be good to construct a `LinearSDEBase` class too, that `MvLinearSDE` and `LinearSDE` can inherit from.

But before this can be done, the API for the univariate case needs to be sorted out.

- Implement the `db` method of the `ForwardProposal` and `MvForwardProposal` classes

This function is only used when constructing a `VanDerMeulenSchauer` auxiliary bridge to transform the particles, with an OU process as the choice of auxiliary bridge. Finish the implementation of this at somepoint, but for now it is not a big priority.

## Low Priority


- You have lazily added cacheing for `_a`, `_b` and `_v` methods for hypoelliptic SDEs to get immediate speedups. Formalise the cacheing a bit more, extend it to Elliptic SDEs. Could also extend to the `A`, `B` and `C` methods for all linear SDEs. These are used throughout for EM simulation and weight evaluation in Van-Der-Meulen-Schauer bridges.

- Improve the documentation for the `CheckSDE` and `BuildLinearSDE` classes within the `auxiliary_bridges.py` file.

- Currently, restrictions have been applied so that only `IntegratedEndPointProposal` can only be used when the given model is an `IntrgratedSDE`. This is not strictly necessary: 
    one could remove this constraint in the future, and it wouldn't be an issue. Have a bit of a think about this!
- Test whether the multivariate SDEs API works when either dimX=1 or dimY=1. If it does work, then we can consider getting rid of the univariate API alogether, except for educational purposes. 
- The API for Time-Switching SDEs currently does not work for univariate SDEs. Change the interface for univariate SDEs (particularly the use of the 'vec' functions in forward proposals/auxiliary bridges) so that 
- For equidistant simulation of linear SDEs (of Brownian or OU form), edit the simulation code to:
    - Store the quantities _a, _b, _v when doing exact simulation. (Currently exact simulation from Linear SDEs is not used in the algorithms, so this won't speed up computation)
- Later on, remove the sdes.distributions file. This is not needed, as the `VaryingCovNormal` class already exists in the current version of the particles package.
