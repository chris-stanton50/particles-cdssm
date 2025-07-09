# Tests for particles-cdssm:

We document here the current state of the tests for the package:

The tests are split into the following folders:

- Filtering: Tests that evaluate whether filtering for CD-SSMs is implemented correctly.
- Linear SDEs: Unittests to check whether transition density of the Linear SDEs implemented within the package are correct. This is a key ingredient in all of the path proposals implemented within the package.


## In preparation for future release:

- Running: Contains python files that represent each of the functions outlined in the API of the package. Can only used to show that each fo the functions runs, does not evaluate whether the output is correct.
- Smoothing: Tests that evaluate whether backward sampling methods that change ancestors (e.g FFBS) are working correctly.
- Smoothing MCMC: Tests that evaluate whether MCMC based smoothing algorithms (that do not extend to the parameter) are working correctly.