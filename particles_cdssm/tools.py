"""
For now, use this module as a miscellaneous dump for various useful functions.
These functions are predominantly used in notebooks to speed up the presentation
of some results.
"""

import numpy as np
import numpy.linalg as nla
import inspect
import time
import collections

# import matplotlib.pyplot as plt
# import arviz as az



MeanAndCov = collections.namedtuple("MeanAndCov", "mean cov")

def log_abs_det(A):
    """
    Compute the log of the absolute determinant of a matrix A.
    """
    return np.log(np.abs(nla.det(A)))

def univ_container(N=100):
    names = [str(round(i, 1)) for i in np.arange(1, 11, dtype=np.float64)/10.]
    dtype = [(name, 'float64') for name in names]
    return np.empty(N, dtype=dtype)

def mv_container(N=100, dim=2):
    names = [str(round(i, 1)) for i in np.arange(1, 11, dtype=np.float64)/10.]
    dtype = [(name, 'float64', dim) for name in names]
    return np.empty(N, dtype=dtype)
    
def get_methods(instance):
    return [name for name, member in inspect.getmembers(instance, predicate=inspect.ismethod)]

def get_attrs(instance):
    return [name for name, member in inspect.getmembers(instance, predicate=inspect.isdatadescriptor)]

def get_properties(instance):
    return [name for name, attr in inspect.getmembers(type(instance))
                  if isinstance(attr, property)]

def match_first_dim(func):
    def new_func(*args):
        new_args = list(args)
        arr_args = [arg for arg in args if type(arg) is np.ndarray]
        arr_args_idx = [i for i, arg in enumerate(args) if type(arg) is np.ndarray]
        N = max([arr_arg.shape[0] for arr_arg in arr_args])
        for i, arg in enumerate(arr_args):
            idx = arr_args_idx[i]
            if arg.shape[0] != N and arg.shape[0] == 1:
                arg = np.concatenate([arg]*N)
            elif arg.shape[0] != N and arg.shape[0] != 1:
                arg = np.stack([arg]*N)
            else:
                pass
            new_args[idx] = arg
        return func(*new_args)
    return new_func

def method_match_first_dim(method):
    """
    Note: The logic of this decorator will break down if the number of particles
    is set to the dimension of either x or y. 
    """
    def new_func(self, *args):
        new_args = list(args)
        arr_args = [arg for arg in args if type(arg) is np.ndarray]
        arr_args_idx = [i for i, arg in enumerate(args) if type(arg) is np.ndarray]
        N = max([arr_arg.shape[0] for arr_arg in arr_args])
        for i, arg in enumerate(arr_args):
            idx = arr_args_idx[i]
            if arg.shape[0] != N and arg.shape[0] == 1 and arg.ndim == 2:
                arg = np.concatenate([arg]*N)
            elif arg.shape[0] != N and arg.ndim > 2:
                arg = np.stack([arg]*N)
            else:
                pass
            new_args[idx] = arg
        return method(self, *args)
    return new_func

def filter_step_var_cov(G, varY, pred, yt):
    """
    Parameters
    ----------
    G:  float
        mean of Y_t | X_t is G * X_t
    varY: float 
        variance of Y_t | X_t
    pred: MeanAndCov object
        predictive distribution at time t
        The mean is an (N, ) array, and the variance is an (N, ) array.
    yt: float
        The observation at time t

    Returns
    ----------

    pred: MeanAndCov object
        filtering distribution at time t    
    """
    pred_mean = pred.mean; pred_var = pred.cov
    opt_prop_mean = pred_mean + (G*pred_var)/((G*pred_var) + varY) * (yt - G*pred_mean)
    opt_prop_var = pred_var * (1 - (G*pred_var)/((G*pred_var) + varY))
    return MeanAndCov(mean=opt_prop_mean, cov=opt_prop_var)

def mv_filter_step_var_cov(G, CovY, pred, yt):
    """
    Version of the function `filter_step_as_array' in the particles.kalman module that 
    can take as input a different covariance matrix for each particle.

    For standard (discrete-discrete) state space models, it is usually the case that the 
    covariance matrix does not depend on the previous particle. This is how the standard
    LGSSM is implemented in the MVLinearGauss class of the particles package. 
    
    For continuous-distrete state space models, if the diffusion coefficient of the model 
    sde is state-dependent, then its transition density (and thus any choice of proxy) will
    depend on the covariance matrix. Thus, this implementation is necessary to deal with these 
    cases.

    We input the mean and covariance of X_t | X_{t-1} in the `pred` input, and the observation 
    density Y_t | X_t through inputs G and CovY. We also give the observation y_t.
    
    We return the distribution of X_t | Y_t = y_t in the form of a MeanAndCov object. 

    Parameters
    ----------
    G:  (dy, dx) numpy array
        mean of Y_t | X_t is G * X_t
    covY: (dy, dy) numpy array
        covariance of Y_t | X_t
    pred: MeanAndCov object
        predictive distribution at time t
        The mean is an (N, dx) array, and the covariance is an (N, dx, dx) array.
    yt: (dy, ) numpy array: The observation at time t

    Returns
    ----------

    pred: MeanAndCov object
        filtering distribution at time t    
    """    
    N = pred.mean.shape[0]; G = np.stack([G]*N); CovY = np.stack([CovY]*N); yt = np.concatenate([yt]*N) # (N, dimY, dimX), (N, dimY, dimY), (N, dimY)
    jt_mu_x = pred.mean; jt_cov_x = pred.cov; # (N, dimX), (N, dimX, dimX)
    jt_mu_y = np.einsum('ijk,ik->ij', G, jt_mu_x) # (N, dimY, dimX), (N, dimX) -> (N, dimY)
    jt_cov_xy = np.einsum('ijk,ilk->ijl', jt_cov_x, G) # (N, dimX, dimX), (N, dimY, dimX) -> (N, dimX, dimY)
    jt_cov_yx = np.einsum('ijk->ikj', jt_cov_xy) # (N, dimY, dimX)
    jt_cov_y =  np.einsum('ijk,ikl->ijl', G, jt_cov_xy) + CovY # (N, dimY, dimX), (N, dimX, dimY) -> (N, dimY, dimY)
    opt_prop_loc = jt_mu_x + np.einsum('ijk,ik->ij', jt_cov_xy, nla.solve(jt_cov_y, yt - jt_mu_y)) # (N, dimX)
    opt_prop_cov = nla.solve(jt_cov_y, jt_cov_yx) # (N, dimY, dimX)
    opt_prop_cov = jt_cov_x - np.einsum('ijk,ikl->ijl', jt_cov_xy, opt_prop_cov) # (N, dimX, dimY) (N, dimY, dimX) -> (N, dimX, dimX)
    return MeanAndCov(mean=opt_prop_loc, cov=opt_prop_cov)
    
# @match_first_dim
def mv_grad_log_linear_gaussian(x_s: np.ndarray, x_t: np.ndarray, A: np.ndarray, b: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    The gradient of the log of a linear Gaussian transition density w.r.t x_s
    X_t | X_s = x_s \sim \mathcal{N}(A x_s + b, S)

    $$ \nabla_{x_s} \log(p_{s, t}(x_t|x_s)) = [A^T S^{-1} (x_t - b) - A^T S^{-1} A x_s]$$

    $$ \nabla_{x_s} \log(p_{s, t}(x_t|x_s)) = A^T S^{-1} (x_t - A x_s - b)$$
        
    Standard dimensions of the inputs:

    x_s (N, dimX)
    x_t (N, dimY)
    A (N, dimY, dimX)
    b (N, dimY)
    S (N, dimY, dimY)
    
    Dimension of output: 
    (N, dimX)
    """
    dimX = x_s.shape[1]; dimY = x_t.shape[1]
    N = max(x_s.shape[0], x_t.shape[0])
    A_trans = np.einsum('ijk->ikj', A)
    A_x_s = np.einsum('ijk,ik->ij', A, x_s)
    grad_log_lg = np.einsum('ijk,ik->ij', A_trans, nla.solve(S, x_t - A_x_s - b))
    # grad_log_lg = np.einsum('ijk,ik->ij', A_trans, nla.solve(S, x_t - b)) - np.einsum('ijk,ik->ij', A_trans, nla.solve(S, A_x_s))
    return grad_log_lg

def mv_grad_grad_log_linear_gaussian(A: np.ndarray, b: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    The hessian of the log of a linear Gaussian transition density w.r.t x_s 
    X_t | X_s = x_s \sim \mathcal{N}(A x_s + b, S)

    $$ \nabla_{x_s} \nabla_{x_s}^T \log(p_{s, t}(x_t|x_s)) = -A^T S^{-1} A $$
    
    Standard dimensions of the inputs:

    A (N, dimY, dimX)
    b (N, dimY)
    S (N, dimY, dimY)
    
    Dimension of output: 

    (N, dimX)
    """    
    return -1.*np.einsum('ijk,ijl->ikl', A, nla.solve(S, A)) # (N, dimX, dimX)
 
def grad_log_linear_gaussian(x_s: np.ndarray, x_t: np.ndarray, a, b, s):
    """
    The gradient of the log of a linear Gaussian transition density 
    X_t | X_s = x_s \sim \mathcal{N}(a x_s + b, s)

    Standard dimensions of the inputs:

    x_s (N, )
    x_t (N, )
    a float
    b float
    s float
    """
    return (a* (x_t - a*x_s - b))/s

def vec_grad_log_linear_gaussian(x_s: np.ndarray, x_t: np.ndarray, a, b, s): 
    """
    Vectorised implementation of the gradient of the log transition density to use
    when the gradient of the log transition density of a linear SDE is needed for the 
    evaluation of path integrals. For this implementation, the inputs to the function will be of
    the following dimension:

    x_s (N, num+1)
    x_t (N, )
    a (N, num+1)
    b (N, num+1)
    s (N, num+1)

    The matrix transpose .T is applies to x_s during the calculations, to ensure that the broadcasting
    is done correctly.
    """
    return ((x_t - (a*x_s).T).T - b)/s

def grad_grad_log_linear_gaussian(a, b, s):
    """
    The second derivative of the log of a linear Gaussian transition density 
    X_t | X_s = x_s \sim \mathcal{N}(a x_s + b, s)
    a: float
    s: float
    """
    return (-1.*(a * a))/s

def sims_to_array(x, S_ts, x0=None):
    """
    Converts list of structured arrays to a single structured array.
    These objects are typically generated from 2 sources:

    - Generation of synthetic data (cdssm.simulate method)
    - The list within a particles history object (F/B Guided, not F/B Reparameterised)
    
    Inputs
    ----------
    x: List of structured arrays of length l
        x[i] is a structured array of shape (N, ) 
        contains num fields each of dimension (1, ) or (1, dimX) 
    S_ts: (l+1, ) Discrete observation times that the simulations correspond to
    s_init: (Optional) float: The initial time point of the simulation.
    x0: (Optional) (1, dimX): If included, this initial point is prepended to the 
        returned unstructured array.

    Returns
    ----------
    X: Unstructured array of shape (N, l*num (+1), dimX)
    ts: Discretisation points: (l*num (+1), )
    """
    names = x[0].dtype.names; N = x[0].shape[0]
    X = []; ts = []
    for i in range(len(x)):
        names = x[i].dtype.names; num = len(names)
        x_i = np.stack([x[i][name] for name in names], axis=1) # (N, num) / (N, num, dimX)
        ts_i = S_ts[i] + np.arange(1, num+1, dtype=np.float64) * (S_ts[i+1] - S_ts[i])/num # (num, )
        X.append(x_i); ts.append(ts_i)
    X = np.concatenate(X, axis=1) # (N, l*num, dimX)
    ts = np.concatenate(ts, axis=0) # (l*num)
    if x0 is not None:
        x0_arr = np.stack([x0]*N, axis=0) # (N, 1) / (N, 1, dimX)
        X = np.concatenate([x0_arr, X], axis=1) # (N, l*num + 1) / (N, l*num + 1, dimX)
        ts = np.concatenate([np.array([0.]), ts], axis=0) # (l*num+1,)
    return X, ts
    
def struct_array_to_array(struct_X):
    """
    Utility function to convert structured array consisting of paths from the proposal SDE to 
    a unstructured numpy array. For use in the context of 1D SDEs.

    Inputs
    ----------
    struct_X: Structured array, containing the sample paths generated from an SDE object.
    
    Returns
    ----------
    X: Unstructured array, of dimension (num, N)

    where num is the number of imputed points, and N is the number of particles
    """
    X = np.array([struct_X[name] for name in struct_X.dtype.names])
    return X

def start_points_paths_to_array(x_start, X):
    """
    Utility function to convert structured array of paths from the proposal SDE 
    and an unstructured vector of start points into an unstrucutured numpy array
    that contains the start points follow by the paths. For use in the context of 
    1D SDEs.

    Inputs
    ----------
    x_start: np.array of shape (N, ) where N is the number of particles
    X: Structured array, containing the sample paths.

    Returns
    --------
    X_array: An unstructured numpy array of shape (N, num+1)

    Where N is the number of particles, and num is the number of imputed points. 
    """
    N = len(x_start)
    x_start = x_start.reshape(1, N)
    X_array = struct_array_to_array(X)
    X_array = np.concatenate([x_start, X_array]).T
    return X_array

def vectorise_param(param, num_plus_1):
    if type(param) is float:
        return param
    else:
        return np.stack([param]*num_plus_1).T

def mv_state_container_size(X: np.ndarray, x_start):
    X_shape_idx = 1 if X.shape == () else X.shape[1]
    x_start_shape_idx = 1 if x_start.shape == () else x_start.shape[1]
    size = max(X_shape_idx, x_start_shape_idx)
    return size

def init_kwargs_dict(cls, locals):
    signature = inspect.signature(cls.__init__)
    params = signature.parameters
    kwarg_names = [name for name, param in params.items() if param.default != inspect.Parameter.empty]
    init_kwargs = {k: locals[k] for k in locals if k in kwarg_names}
    return init_kwargs

def struct_arr_to_arr(struct_arr):
    """
    Input: struct_arr: numpy structured array of shape (M, )
    Returns: arr: numpy array of shape (M, num)
    """
    names = struct_arr.dtype.names # num names
    arr = np.stack([struct_arr[name] for name in names], axis=1)
    # List of length (num, ) of numpy arrays of shape (M, )
    return arr

def struct_arrs_to_arr(struct_arrs):
    """
    Input: struct_arrs: list of length T of numpy structured arrays of shape (M, )
            each structured array has the same dtype and consists of num fields: the imputed points.
    Returns: arr: numpy array of shape (M, num*T): representing the full paths.
    """
    arrs = [struct_arr_to_arr(struct_arr) for struct_arr in struct_arrs]
    arr = np.concatenate(arrs, axis=1)
    return arr

def create_diagonal_matrices(arr):
    """
    Create a batch of diagonal matrices from a batch of vectors.

    Parameters:
    arr (np.ndarray): Input array of shape (N, d)

    Returns:
    np.ndarray: Output array of shape (N, d, d) with arr as diagonals
    """
    return np.einsum('ni,ij->nij', arr, np.eye(arr.shape[1]))

# Generate a random symmetric positive definite matrix
def generate_spd_matrix(d):
    while True:
        # Generate a random matrix
        A = np.random.rand(d, d)
        
        # Make it symmetric
        spd_matrix = np.dot(A, A.T)
        
        # Add d * I to shift eigenvalues up, which often helps but isn't strictly necessary
        spd_matrix += d * np.eye(d)
        
        # Check if the matrix is positive definite by confirming all eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(spd_matrix)
        if np.all(eigenvalues > 0):
            break
    
    return spd_matrix

def isNonNegDefinite(A):
    if not A.ndim == 2:
        return False
    if A.shape[0] != A.shape[1]:
        return False
    eigenvalues = np.linalg.eigvalsh(A)
    if np.all(eigenvalues >= 0):
        return True
    else:
        return False
    
def timed_func(func):
    def timed(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return timed

# def univariate_simulation_test(sde, nums, dist_kwargs):
#     fig, ax = plt.subplots()
#     for num in nums:
#         dist_kwargs['num'] = num
#         rvs = sde.simulate(size=1, **dist_kwargs)
#         t_s = [dist_kwargs['t_start']] + [float(t) for t in rvs.dtype.names]
#         X_ts = struct_array_to_array(rvs); X_ts = np.concatenate([np.array([dist_kwargs['init_x']]), X_ts])
#         ax.plot(t_s, X_ts, label=f'n_points = {num}')
#     # Configure axis settings
#     ax.legend(); ax.set_xlabel('t'); ax.set_ylabel('X_t'); ax.grid(visible=True)
#     return fig, ax

# def to_idata(alg, name=None):
#     """
#     Converts an X_MCMC object to an ArviZ InferenceData object. 
    
#     Also accessible as a method to the X_MCMC class.
    
#     Inputs:
#     ----------
    
#     alg: X_MCMC object
#         The MCMC algorithm to convert.
#     name: str (optional): The name of the algorithm to convert into an idata object.

#     Returns:
#     ----------
#     idata: ArviZ InferenceData object
#         The converted MCMC algorithm.
#     """
#     chain = alg.chain
#     iscontinuousdiscete = hasattr(alg.fk, 'cdssm')
#     T = len(alg.fk.data)

#     # Preprocess x chain into a (T, niter) / (T, niter, dimX) array
#     # CDSSM case
#     x = chain.x
#     if iscontinuousdiscete:
#         dx = 1 if x[x.dtype.names[-1]].ndim == 2 else x[x.dtype.names[-1]].shape[2]
#         if alg.fk.cdssm.isobservedat0:
#             obs_times = [alg.fk.cdssm.S(t) for t in range(T)]
#             init_x = chain.init_x
#             x_arr = np.concatenate([init_x['0.0'][:, np.newaxis], x[x.dtype.names[-1]]], axis=1)        
#         else:
#             obs_times = [alg.fk.cdssm.S(t) for t in range(1, T+1)] 
#             x_arr = chain.x[x.dtype.names[-1]] # Only store end points for now, consider changing this later
#         x_arr = x_arr[:, :, 0] if x_arr.ndim == 3 and dx == 1 else x_arr
#     # SSM case 
#     else:
#         dx = 1 if x.ndim == 2 else x.shape[2]
#         obs_times = np.arange(T)
#         x_arr = x
        
#     # Preprocess observations into (T, ) / (T, dimY) array
#     y = alg.fk.data
#     dy = 1 if y[0].ndim == 1 else y[0].shape[1]
#     y_arr = np.concatenate(alg.fk.data, axis=0)
#     y_arr = y_arr.ravel() if dy == 1 else y_arr

#     x_arr = x_arr[np.newaxis]
    
#     # lib_attrs

#     attrs = {
#     'inference_algorithm': alg.__class__.__name__,
#     'inference_library': 'particles_cdssm', 
#     'inference_library_version': '0.1.0', 
#     'fk_name': alg.fk.__class__.__name__,
#     'Nx': alg.Nx,
#     'niter': alg.niter,
#     'T': alg.fk.T,
#     'cpu_time': alg.cpu_time,
#     }
#     if 'ICSMC' in alg.__class__.__name__:
#         attrs['backward_step'] = str(alg.backward_step)
        
#     if iscontinuousdiscete:
#         cdssm_name = alg.fk.cdssm.__class__.__name__
#         model_sde_name = alg.fk.cdssm.model_sde.__class__.__name__
#         cdssm_attrs = {
#             'name': cdssm_name + '_' + model_sde_name,
#             'cdssm_name': cdssm_name,
#             'fk_sname': alg.fk.sname,
#             'num': alg.num,
#             'model_sde': model_sde_name
#             }
#         attrs.update(cdssm_attrs)
#     else:
#         ssm_name = alg.fk.ssm.__class__.__name__
#         fk_name = alg.fk.__class__.__name__
#         ssm_attrs = {
#             'name': ssm_name,
#             'ssm_name': ssm_name,
#             'fk_sname': fk_name
#             }
#         attrs.update(ssm_attrs)

#     if name is not None:
#         attrs['name'] = name

#     # Build InferenceData posterior
#     idata_post = az.from_dict(
#         posterior={"x": x_arr},
#         coords = {"time": obs_times},
#         dims={"x": ["time"] if dx == 1 else ["time", "dimX"]},
#         posterior_attrs = attrs,
#     )

#     # Build InferenceData observations
#     idata_obs = az.from_dict(
#         observed_data={"y": y_arr},
#         coords = {"time": obs_times},
#         dims={"y": ["time"] if dy == 1 else ["time", "dimY"]},
#     )

#     idata = az.InferenceData(posterior=idata_post.posterior, observed_data=idata_obs.observed_data, attrs=attrs)
#     idata.observed_data.attrs = idata_post.attrs.copy()
#     return idata

def build_cdssm(cdssm_spec):
    """
    Inputs:
    -------
    cdssm_spec_name: dict containing the CDSSM Spec

    Returns:
    --------
    cdssm: A cdssm object defined by the given CDSSM Spec.
    """

    # Extract objects from the CDSSM Spec:
    sde_cls = cdssm_spec['sde_cls']
    cdssm_cls = cdssm_spec['cdssm_cls']
    
    sde_params = cdssm_spec['sde_params']
    cdssm_params = cdssm_spec['cdssm_params']

    # We define the underlying SDE:
    sde = sde_cls(**sde_params)

    # We define the CDSSM:
    cdssm = cdssm_cls(sde, **cdssm_params)
    return cdssm