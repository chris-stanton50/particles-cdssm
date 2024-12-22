# To do

- Go through the `feynman_kac.py` file and update the way in which Backward Proposals are constructed, so that they use end point proposals.

## High Priority

- Debug Implementations of **Elliptic** and **Hypoelliptic** diffusions in higher dimensions.
- Implement some simulation schemes for hypoelliptic diffusions.

## Medium Priority

- Improve the API for choosing end point proposals in the backward proposal case.

## Low Priority

- Test whether the multivariate SDEs API works when either dimX=1 or dimY=1. If it does work, then we can consider getting rid of the univariate API alogether, except for educational purposes. 
- The API for Time-Switching SDEs currently does not work for univariate SDEs. Change the interface for univariate SDEs (particularly the use of the 'vec' functions in forward proposals/auxiliary bridges) so that 
- For equidistant simulation of linear SDEs (of Brownian or OU form), edit the simulation code to:
    - Store the quantities _a, _b, _v when doing exact simulation. (Currently exact simulation from Linear SDEs is not used in the algorithms, so this won't speed up computation)
- Later on, remove the sdes.distributions file. This is not needed, as the `VaryingCovNormal` class already exists in the current version of the particles package.
