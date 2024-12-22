import numpy as np
import numpy.linalg as nla
import scipy.stats as stats
from sdes.sdes import SDE, MvSDE, MvEllipticSDE, HypoellipticSDE, BrownianMotion, OrnsteinUhlenbeck, MvIndepBrownianMotion, MvBrownianMotion, MvIndepOrnsteinUhlenbeck, MvOrnsteinUhlenbeck, TimeSwitchingSDE
from sdes.sdes import HypoellipticSDE, IntegratedIndepBrownianMotion, IntegratedBrownianMotion, IntegratedIndepOrnsteinUhlenbeck, IntegratedOrnsteinUhlenbeck
from sdes.sdes import TwiceIntegratedIndepBrownianMotion, TwiceIntegratedBrownianMotion, TwiceIntegratedIndepOrnsteinUhlenbeck, TwiceIntegratedOrnsteinUhlenbeck
from sdes.path_integrals import log_girsanov, log_delyon_hu, log_drift_delyon_hu, log_van_der_meulen_schauer, mv_log_girsanov, mv_log_delyon_hu, mv_log_van_der_meulen_schauer
from sdes.tools import log_abs_det, filter_step_var_cov, MeanAndCov
from particles.distributions import ProbDist, VaryingCovNormal


class EndPointProposal(ProbDist):
    pass

class LinearEndPointProposal(EndPointProposal):
    pass

class MvLinearEndPointProposal(VaryingCovNormal, LinearEndPointProposal):
    """
    End point proposals for backward guided/reparameterised Feynman Kac models 
    that for each input particle, construct a Linear SDE based on Taylor expansion
    of the drift and diffusion coefficients about the input particles.
    """

    def __init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY):
        MvForwardProposal.__init__(self, sde, x_start, t_start, t_end, y, LY, sigmaY)
        pred = MeanAndCov(self.pred_loc, self.pred_cov)        
        opt_prop_loc, opt_prop_cov = filter_step_var_cov(LY, sigmaY @ sigmaY.T, pred, y) 
        VaryingCovNormal.__init__(self, loc=opt_prop_loc, cov=opt_prop_cov)
    
    @property        
    def pred_loc(self):
        s = self.t_start; t = self.t_end; x_s = self.x_start
        A = self.LinearSDE._a(s, t); b = self._b(s, t)
        mu_x = np.einsum('ijk,ik->ij', A, x_s) + b # (N, dimX, dimX), (N, dimX) -> (N, dimX)
        return mu_x
    
    @property
    def pred_cov(self):
        return self.LinearSDE._v(self.t_start, self.t_end)

class MvNaiveEndPointProposal(MvLinearEndPointProposal, EllipticBrownianProposal):
    any_cov = False
    drift = False
    
class MvEulerMaruyamaEndPointProposal(MvLinearEndPointProposal, EllipticBrownianProposal):   
    any_cov = True
    full_cov = True
    drift = True

class MvOUEndPointProposal(MvLinearEndPointProposal, EllipticOUProposal):
    full_cov = True
    
class MvLocalLinearEndPointProposal(MvLinearEndPointProposal, EllipticOUProposal):
    any_cov = True
    full_cov = True
