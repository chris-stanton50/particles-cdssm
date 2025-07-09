# Linear SDEs 

These tests were created to ensure that the transition density of the linear SDE: the part of the package that is most likely to result in errors in the path proposals (and hence all of the particle-based methods with guided proposals) are implemented correctly.

These tests are build with Python's `unittest` module.

To run the tests:

`python -m unittest test_linear_sdes.py` 


The following types of SDEs are working correctly:

- All elliptic Linear SDEs: univariate or multivariate.
- All integrated SDEs, with the exception of the following case:
    - `IntegratedOrnsteinUhlenbeck`, when $d_w \geq 2$ (or equiv: $d_x \geq 4$) where the diffusion coefficient $\phi$ introduces correlation in the noise in the rough component.

Tests still need to be developed in the following case:

- The exception in the integrated SDE case above
- All TwiceIntegratedLinear SDEs.