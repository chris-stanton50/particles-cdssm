import numpy as np
import scipy.stats as stats

def inner_prod(samples_1: np.ndarray, samples_2: np.ndarray, H_1: np.ndarray , H_2: np.ndarray) -> float:
    """
    Calculates the inner product of 2 KDEs from imput samples and kernel covariance:
    
    samples_1: (d, N)
    samples_2: (d, N)
    H_1: (d, d) kernel covariance 1
    H_2: (d, d) kernel covariance 2
    """
    N = samples_1.shape[1]
    M = samples_2.shape[1]
    D = samples_1.shape[0]
    K = np.empty((N*M, 2), dtype=np.float64)
    for i in range(D):
        X, Y = np.meshgrid(samples_1[i], samples_2[i], indexing='ij')
        Z = X - Y
        K[:, i] = Z.ravel()
    p_22 = stats.multivariate_normal(mean=np.array([0., 0.]), cov=(H_1 + H_2)).pdf(K).mean()
    return p_22

def kdes_l2_distance(samples_1: np.ndarray, samples_2: np.ndarray, h_1=None, h_2=None) -> float:
    """
    L_2 distance between 2 Gaussian KDEs. Analytically tractable.
    
    samples_1: (d, N) 
    samples_2: (d, N)
    h_1: float
    h_2: float
    """
    if not h_1:
        h_1 = stats.gaussian_kde(samples_1).factor
    if not h_2:
        h_2 = stats.gaussian_kde(samples_2).factor
    H_1 = np.cov(samples_1) * (h_1 ** 2)
    H_2 = np.cov(samples_2) * (h_2 ** 2)
    p_11 = inner_prod(samples_1, samples_1, H_1, H_1)
    p_22 = inner_prod(samples_2, samples_2, H_2, H_2)
    p_12 = inner_prod(samples_1, samples_2, H_1, H_2)
    return np.sqrt(p_11 + p_22 - 2.*p_12)
    
def kde_l2_distance(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, h=None) -> float:
    """    
    L_2 distance between 1 Gaussian KDE and a Gaussian density with parameters (mean, cov). 
    Analytically tractable.
    
    samples: (d, N): array of samples from the distribution
    mean: (d, ) array of the mean of the Gaussian target
    cov: (d, d) array of the covariance of the Gaussian target
    """
    if not h:
        h = stats.gaussian_kde(samples).factor
    H = np.cov(samples) * (h ** 2)
    p_11 = stats.multivariate_normal(mean=np.array([0., 0.]), cov=2.*cov).pdf(np.array([0., 0.]))
    p_12 = stats.multivariate_normal(mean=mean, cov=(cov+H)).pdf(samples.T).mean()
    N = samples.shape[1]
    K = np.empty((N**2, 2), dtype=np.float64)
    for i in range(2):
        X, Y = np.meshgrid(samples[i], samples[i], indexing='ij')
        Z = X - Y
        K[:, i] = Z.ravel()
    p_22 = stats.multivariate_normal(mean=np.array([0., 0.]), cov=2.*H).pdf(K).mean()
    sq_l_2_distance = p_11 - 2.*p_12 + p_22
    return np.sqrt(sq_l_2_distance)

def left_kde_kl_divergence(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, M=100, h=None) -> float:
    """    
    KL(P|Q) divergence between 1 Gaussian KDE (Q) and a MV Gaussian density (P) with parameters (mean, cov).
    
    Inputs:
    ---------        
    samples: (d, N): array of samples from the distribution
    mean: (d, ) array of the mean of the Gaussian target
    cov: (d, d) array of the covariance of the Gaussian target
    M (int): Monte Carlo sample size
    h (float): kernel bandwidth (set to scott's bandwidth if None)
    
    Returns:
    ---------
    ks_ests (np.ndarray) (M, ) array of 1 sample MC estimates of the KL divergence.
    """
    kde = stats.gaussian_kde(samples, bw_method=h)
    mvn = stats.multivariate_normal(mean=mean, cov=cov)
    mc_samples = mvn.rvs(size=M)
    kl_ests = mvn.logpdf(mc_samples) - kde.logpdf(mc_samples.T) # (M, )
    return kl_ests
    
def right_kde_kl_divergence(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, M=100, h=None) -> float:
    """    
    KL(Q|P) divergence between 1 Gaussian KDE (Q) and a MV Gaussian density (P) with parameters (mean, cov).
    
    Inputs:
    ---------        
    samples: (d, N): array of samples from the distribution
    mean: (d, ) array of the mean of the Gaussian target
    cov: (d, d) array of the covariance of the Gaussian target
    M (int): Monte Carlo sample size
    h (float): kernel bandwidth (set to scott's bandwidth if None)
    
    Returns:
    ---------
    ks_ests (np.ndarray) (M, ) array of 1 sample MC estimates of the KL divergence.
    """
    kde = stats.gaussian_kde(samples, bw_method=h)
    mvn = stats.multivariate_normal(mean=mean, cov=cov)
    mc_samples = kde.resample(size=M)
    kl_ests = kde.logpdf(mc_samples) - mvn.logpdf(mc_samples.T) # (M, )
    return kl_ests

def kdes_kl_divergence(samples_1: np.ndarray, samples_2: np.ndarray, M=100, h_1=None, h_2=None) -> float:
    """    
    KL(Q|P) divergence between 2 Gaussian KDEs.

    samples_1, h_1 corresponds to distribution Q.

    Inputs:
    ---------        
    samples_1: (d, N): array of samples from the distribution
    samples_2: (d, N) array of the mean of the Gaussian target
    M: int: Number of Monte Carlo samples 
    h_1: float
    h_2: float
    
    Returns:
    ---------
    ks_ests (np.ndarray) (M, ) array of 1 sample MC estimates of the KL divergence.
    """
    kde_1 = stats.gaussian_kde(samples_1, bw_method=h_1)
    kde_2 = stats.gaussian_kde(samples_2, bw_method=h_2)
    mc_samples = kde_1.resample(size=M)
    kl_ests = kde_1.logpdf(mc_samples) - kde_2.logpdf(mc_samples) # (M, )
    return kl_ests