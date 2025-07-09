"""
Core module of particles_cdssm. Contains the following classes:

- `SMC` : New version of the SMC class from the particles package, with additional method
    `backward_sampling_genealogy` for sampling from the smoothing distribution. Added as 
    a method to the `ParticleHistory` object, see the `particles_cdssm.smoothing` module.

- `CDSSM_SMC` : SMC Class for CD-SSMs. Use this class to run SMC on a CD-SSM model, by 
    passing the Feynman-Kac formalism for the CD-SSM.

Also implemented are tools enabling the use of basic parallelisation of `CDSSM_SMC` algorithms,
as also implemented in the particles package in the case of SMC algorithms. 

See the documentation for CDSSM_SMC for further details. 
"""

import time
import functools
import numpy as np

import particles
from particles.collectors import default_collector_cls, Moments
from particles.utils import multiplexer

from particles_cdssm.feynman_kac import CDSSM_FeynmanKac
from particles_cdssm.smoothing import generate_hist_obj

class SMC(particles.SMC):
    """
    New version of `SMC` with additional methods for backward sampling.
    New method is passed to the `ParticleHistory` object, see the `particles_cdssm.smoothing` module.
    """
    def __init__(
        self,
        fk=None,
        N=100,
        qmc=False,
        resampling="systematic",
        ESSrmin=0.5,
        store_history=False,
        verbose=False,
        collect=None,
    ):
        particles.SMC.__init__(
            self,
            fk=fk,
            N=N,
            qmc=qmc,
            resampling=resampling,
            ESSrmin=ESSrmin,
            store_history=store_history,
            verbose=verbose,
            collect=collect,
        )
        self.hist = generate_hist_obj(store_history, self)
        
class CDSSM_SMC(SMC):
    """
    Version of SMC for CD-SSMs.
    """
    def __init__(
        self,
        fk=None,
        N=100,
        resampling="systematic",
        ESSrmin=0.5,
        store_history=False,
        verbose=False,
        collect=None,
        num=10
    ):
        super().__init__(fk=fk,
                        N=N,
                        qmc=False,
                        resampling=resampling,
                        ESSrmin=ESSrmin,
                        store_history=store_history,
                        verbose=verbose,
                        collect=collect,
                        )
        self.fk.num = num # Pass the number of imputed points to the fk object.
        if not isinstance(fk, CDSSM_FeynmanKac):
            raise ValueError('fk must be an instance of CDSSM_FeynmanKac')
        self.hist = generate_hist_obj(store_history, self)


# -----------Functions for default behaviour of  `cdssm_smc` -----------

class _picklable_f:

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, **kwargs):
        pf = CDSSM_SMC(**kwargs)
        pf.run()
        return self.fun(pf)
            
@_picklable_f
def _identity(x):
    return x

# -----------Functions for print_summary collector, to be applied to an out_func -----------

def get_col(pf):
    cols = pf.summaries._collectors
    for col in cols:
        if col.__class__ not in default_collector_cls + [Moments]:
            return col
    return None

def print_summary(out_func):
    """
    Wrapper on the out_func, that prints the name of the `fk` object used for the algorithm,
    and the cpu time of the algorithm.
    If a (non-default) collector is used, it also prints the name of the collector.
    """
    @functools.wraps(out_func)
    def dec_out_func(pf):
        name = pf.fk.__class__.__name__ if not isinstance(pf.fk, CDSSM_FeynmanKac) else pf.fk.sname 
        col = get_col(pf)
        collector = f'with {col.summary_name} collector ' if col else ''
        print(f'Running {pf.__class__.__name__} {name} {collector}with {pf.N} particles took {round(pf.cpu_time,ndigits=4)} seconds')
        return out_func(pf)
    return dec_out_func

# ----------- Examples of an `out_func`that one can pass to run multiCDSSM_SMC/multiSMC -----------

# Example of an out_func that returns the summaries object of the particle filter
# This could be preferred to storing the whole pf object for each run, as this would
# store the particles, weights and ancestors at time T.
@print_summary
def summaries(pf):
    pf.summaries.cpu_time = pf.cpu_time # Add the cpu time to the summaries object
    return pf.summaries

@print_summary
def logLt(pf):
    return pf.logLt

def multiCDSSM_SMC(nruns=10, nprocs=0, out_func=None, collect=None, **args):
    """
    Version of particles.multiSMC that is applicable to Feynman-Kac measures
    that have been generated from CD-SSMs. 
    Runs the `CDSSM_SMC` algorithm in parallel.
        
    Run CDSSM_SMC algorithms in parallel, for different combinations of parameters.

    `multiCDSSM_SMC` relies on the `multiplexer` utility, and obeys the same logic.
    A basic usage is::

        results = multiCDSSM_SMC(fk=my_fk_model, N=100, nruns=20, nprocs=0)

    This runs the same SMC algorithm 20 times, using all available CPU cores.
    The output, ``results``, is a list of 20 dictionaries; a given dict corresponds
    to a single run, and contains the following (key, value) pairs:

        + ``'run'``: a run identifier (a number between 0 and nruns-1)

        + ``'output'``: the corresponding CDSSM_SMC object (once method run was completed)

    Since a `CDSSM_SMC` object may take a lot of space in memory (especially when
    the option ``store_history`` is set to True), it is possible to require
    `multiCDSSM_SMC` to store only some chosen summary of the CDSSM_SMC runs, using option
    `out_func`. For instance, if we only want to store the estimate
    of the log-likelihood of the model obtained from each particle filter::

        of = lambda pf: pf.logLt
        results = multiCDSSM_SMC(fk=my_fk_model, N=100, nruns=20, out_func=of)

    It is also possible to vary the parameters. Say::

        results = multiCDSSM_SMC(fk=my_fk_model, N=[100, 500, 1000])

    will run the same SMC algorithm 30 times: 10 times for N=100, 10 times for
    N=500, and 10 times for N=1000. The number 10 comes from the fact that we
    did not specify nruns, and its default value is 10. The 30 dictionaries
    obtained in results will then contain an extra (key, value) pair that will
    give the value of N for which the run was performed.

    It is possible to vary several arguments. Each time a list must be
    provided. The end result will amount to take a *cartesian product* of the
    arguments::

        results = multiCDSSM_SMC(fk=my_fk_model, N=[100, 1000], resampling=['multinomial',
                           'residual'], nruns=20)

    In that case we run our algorithm 80 times: 20 times with N=100 and
    resampling set to multinomial, 20 times with N=100 and resampling set to
    residual and so on.

    Finally, if one uses a dictionary instead of a list, e.g.::

        results = multiCDSSM_SMC(fk={'backward': fk_backward, 'forward': fk_forward}, N=100)

    then, in the output dictionaries, the values of the parameters will be replaced
    by corresponding keys; e.g. in the example above, {'fk': 'fk_forward'}. This is
    convenient in cases such like this where the parameter value is some non-standard
    object.

    Parameters
    ----------
    * nruns : int, optional
        number of runs (default is 10)
    * nprocs : int, optional
        number of processors to use; if negative, number of cores not to use.
        Default value is 1 (no multiprocessing)
    * out_func : callable, optional
        function to transform the output of each CDSSM_SMC run. (If not given, output
        will be the complete CDSSM_SMC object).
    * collect : list of collectors, or 'off'
        this particular argument of class SMC may be a list, hence it is "protected"
        from Cartesianisation
    * args : dict
        arguments passed to CDSSM_SMC class (except collect)

    Returns
    -------
    A list of dicts

    See also
    --------
    `particles.utils.multiplexer`: for more details on the syntax.
    """
    f = _identity if out_func is None else _picklable_f(out_func)
    return multiplexer(
        f=f,
        nruns=nruns,
        nprocs=nprocs,
        seeding=True,
        protected_args={"collect": collect},
        **args
    )
  
#-----Example additive functions for input to smoothing_worker. Can be used for both SMC and CDSSM_SMC-----
def use_end_point(phi):
    def phi_dec(t, x, xf):
        if x is not None:
            x = x if x.dtype in [np.float32, np.float64] else x[x.dtype.names[-1]]
        xf = xf if xf.dtype in [np.float32, np.float64] else xf[xf.dtype.names[-1]]
        out = phi(t, x, xf)
        return out
    return phi_dec

@use_end_point
def phi_x(t, x, xf): # 1st moment of the end point (N, ), (N, ) -> (N, )
    return xf

@use_end_point
def phi_x_x(t, x, xf): # 2nd moment of the end point (N, dimX), (N, dimX) -> (N, dimX)
    return xf * xf

@use_end_point
def phi_x_xf(t, x, xf): # 2nd moment of the end point (N, ), (N, ) -> (N, )
    return np.zeros_like(xf) if x is None else x * xf

@use_end_point
def phi_x_3(t, x, xf): # 3rd moment of the end point (N, dimX), (N, dimX) -> (N, dimX)
    return xf ** 3

@use_end_point
def phi_x_4(t, x, xf): # 4nd moment of the end point (N, dimX), (N, dimX) -> (N, dimX)
    return xf ** 4

def gen_quantile_add_func(q):
    def quantile(t, x, xf):
        np.where(xf < 0, 1, 0)
        return np.quantile(xf, q, axis=0)

quantile_add_funcs = {}
default_add_funcs = {'phi_x': phi_x, 'phi_x_x': phi_x_x, 'phi_x_xf': phi_x_xf} # Simple choices that can be applied to any SDE regardless of dimension.
#-----------------------------------------------------------------------------------------

def smoothing_worker(
    method=None, N=100, fk=None, num=10, smc_cls=CDSSM_SMC, add_funcs=default_add_funcs, quantiles=None):
    """Modified version of 'smoothing_worker' from particles.smoothing.
    Removed two-filter smoothing, enabled evaluation of multiple additive functions.

    This worker may be used in conjunction with particles.utils.multiplexer in order to
    run in parallel off-line smoothing algorithms.
    
    When using the `multiplexer` function to run offline smoothing algorithms: ensure the following:
    
    - `add_funcs` is set within the `protected_args` as its standard input is a dictionary, 
        and one does not want to cartesianise over the choices of additive functions.
    - `smc_cls is set to either `CDSSM_SMC` or `SMC`. We should not cartesianise over the 
        choice of SMC class.
        
    Parameters
    ----------
    method : string
        Input a string for the choice of smoothing method. The choices are:
        ['genealogy', 'FFBS_purereject', 'FFBS_hybrid', FFBS_MCMC', 'FFBS_ON2']
        For 'FFBS_MCMC', can append an integer (e.g `FFBS_MCMC_5`) to specify the number of MCMC steps.
        If running CDSSM_SMC algorithms, only the following choices are available:
        ['genealogy', 'FFBS_MCMC', 'FFBS_ON2']
        For 'FFBS_MCMC', can append an integer (e.g `FFBS_MCMC_5`) to specify the number of MCMC steps.
    N : int
        number of particles
    fk : Feynman-Kac object
        The Feynman-Kac model used for the smoothing algorithm
    num : int 
        Number of imputed points to use if running CDSSM_SMC algorithms
    smc_cls: particles.SMC object
        The smc class to use: set to either CDSSM_SMC or SMC
    add_funcs : dict of functions, each with with signature (t, x, xf)
        dictionary of additive functions, at time t, for particles x=x_t and xf=x_{t+1}
        Each function should be defined to return either an (N, ) array or an (N, D) array.
    quantiles: None or list/array-like of quantiles to compute
        

    Returns
    -------
    out : dict
    
    The output dictionary contains
    * the output of the smoothing algorithm for each additive function in add_funcs
    * cpu_time: - CPU time to run the smoothing algorithm
    """
    T = fk.T; out = {}
    fk_str = fk.__class__.__name__ if smc_cls is SMC else fk.sname
    if smc_cls is CDSSM_SMC:
        pf = CDSSM_SMC(fk=fk, N=N, num=num, store_history=True)
    else:
        pf = SMC(fk=fk, N=N, store_history=True)
    tic = time.perf_counter()
    pf.run()
    if method == "genealogy":
        z = pf.hist.backward_sampling_genealogy(N)
    elif method.startswith("FFBS"):
        split = method.split("_")
        submethod = split[1]
        if submethod == "ON2":
            z = pf.hist.backward_sampling_ON2(N)
        elif submethod == "hybrid":
            z = pf.hist.backward_sampling_reject(N)
        elif submethod == "purereject":
            z = pf.hist.backward_sampling_reject(N, max_trials=N * 10 ** 9)
        elif submethod == "MCMC":
            nsteps = int(split[2]) if len(split) > 2 else 1
            z = pf.hist.backward_sampling_mcmc(N, nsteps=nsteps)
        else:
            print("smoothing_worker: no such method")
    else:
        print("smoothing_worker: no such method")
    cpu_time = time.perf_counter() - tic
    print(f"Method {method} with fk_model {fk_str} took {round(cpu_time, 2)}s for N={N}, T={fk.T}")
    for add_func_name, add_func in add_funcs.items():
        out_0 = np.mean(add_func(0, None, z[0]), axis=0)
        scalar_out_func = isinstance(out_0, float)
        out_1_T = [np.mean(add_func(t, z[t-1], z[t]), axis=0) for t in range(1, T)]
        out_add_func = [out_0] + out_1_T
        out_add_func = np.array(out_add_func) if scalar_out_func else np.stack(out_add_func, axis=1).T # (T, dimX)/ (T, )
        out[add_func_name] = out_add_func
    if quantiles is not None: # Currently implemented for end points only
        if smc_cls is CDSSM_SMC:
            z_arr = np.stack([z_i[z_i.dtype.names[-1]] for z_i in z]) # (T, N) / (T, N, dimX)
        else:
            z_arr = np.stack([z_i for z_i in z]) # (T, N) / (T, N, dimX)
        out['quantiles'] = np.quantile(z_arr, quantiles, axis=1) # (nQ, T) or (nQ, T, dimX)
        out['quantile_index'] = quantiles
    out['cpu_time'] = cpu_time
    return out