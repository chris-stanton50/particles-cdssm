import numpy as np

class AddFuncStorageBase(object):
    pass

class AddFuncStorage(AddFuncStorageBase):
    """Add Func storage for one-dimensional CD-SSMs"""
    pass

class MvAddFuncStorage(AddFuncStorageBase):
    """Add Func storage for multivariate CD-SSMs"""
    pass


# Univariate CD-SSMs + SSMs

class FirstMom(AddFuncStorage):
    
    @staticmethod
    def cdssm(self, t, xp, x):
        return x[x.dtype.names[-1]]
    
    @staticmethod
    def ssm(self, t, xp, x):
        return x

    def ground_truth(smth: list):
        return sum(mc.mean[0] for mc in smth)

class SecondMom(AddFuncStorage):

    @staticmethod
    def cdssm(self, t, xp, x):
        return np.square(x[x.dtype.names[-1]])

    @staticmethod
    def ssm(self, t, xp, x):
        return np.square(x) 

    def ground_truth(smth: list):
        return sum(mc.cov[0][0] + mc.mean[0]**2 for mc in smth)
    
class DS_Integral_X(AddFuncStorage):

    @staticmethod
    def cdssm(self, t, xp, x):
        if self.isobservedat0 and t == 0:
            return np.zeros(x.shape[0], dtype=np.float64)
        delta_t = float(x.dtype.names[1]) - float(x.dtype.names[0])
        integral = xp[xp.dtype.names[-1]] * delta_t
        for name in x.dtype.names[:-1]:
            integral += x[name] * delta_t
        return integral
    
class DS_Integral_ExpThetaX(AddFuncStorage):

    @staticmethod
    def cdssm(self, t, xp, x):
        if self.isobservedat0 and t == 0:
            return np.zeros(x.shape[0], dtype=np.float64)
        delta_t = float(x.dtype.names[1]) - float(x.dtype.names[0])
        integral = xp[xp.dtype.names[-1]] * delta_t
        for name in x.dtype.names[:-1]:
            # Inverse Lamperti transform for logistic CD-SSM
            integral += np.exp(0.7*x[name]) * delta_t
        return integral
    
# Multivariate CD-SSMs + SSMs

class MvFirstMomFirstCpnt(MvAddFuncStorage):

    @staticmethod
    def cdssm(self, t, xp, x):
        return x[x.dtype.names[-1]][:, 0]

    @staticmethod
    def ssm(self, t, xp, x):
        return x[:, 0]

    @staticmethod    
    def ground_truth(smth: list):
        return sum(mc.mean[0][0] for mc in smth)


class MvFirstMomLastCpnt(MvAddFuncStorage):

    @staticmethod
    def cdssm(self, t, xp, x):
        return x[x.dtype.names[-1]][:, -1]
    
    @staticmethod
    def ssm(self, t, xp, x):
        return x[:, -1]

    @staticmethod
    def ground_truth(smth: list):
        return sum(mc.mean[0][-1] for mc in smth)

class MvSecondMomFirstCpnt(MvAddFuncStorage):

    @staticmethod
    def cdssm(self, t, xp, x):
        return np.square(x[x.dtype.names[-1]][:, 0])
    
    @staticmethod
    def ssm(self, t, xp, x):
        return np.square(x[:, 0])

    @staticmethod
    def ground_truth(smth: list):
        return sum(mc.cov[0][0] + mc.mean[0][0]**2 for mc in smth)
    
class MvSecondMomLastCpnt(MvAddFuncStorage):    

    @staticmethod
    def cdssm(self, t, xp, x):
        return np.square(x[x.dtype.names[-1]][:, -1])

    @staticmethod
    def ssm(self, t, xp, x):
        return np.square(x[:, -1])

    @staticmethod
    def ground_truth(smth: list):
        return sum(mc.cov[1][1] + mc.mean[0][1]**2 for mc in smth)

class MvSecondMomFirstLastCpnt(MvAddFuncStorage):    

    @staticmethod
    def cdssm(self, t, xp, x):
        first_cpnt = x[x.dtype.names[-1]][:, 0]
        last_cpnt = x[x.dtype.names[-1]][:, -1]
        return first_cpnt * last_cpnt

    @staticmethod
    def ssm(self, t, xp, x):
        first_cpnt = x[:, 0]
        last_cpnt = x[:, -1]
        return first_cpnt * last_cpnt

    @staticmethod
    def ground_truth(smth: list):
        return sum(mc.cov[1][0] + mc.mean[0][0]*mc.mean[0][1] for mc in smth)

class DS_Integral_X_FirstCpnt(MvAddFuncStorage):
    
    @staticmethod
    def cdssm(self, t, xp, x):
        if self.isobservedat0:
            return np.zeros(x.shape[0], dtype=np.float64)
        delta_t = float(x.dtype.names[1]) - float(x.dtype.names[0])
        integral = xp[xp.dtype.names[-1]][:, 0] * delta_t
        for name in x.dtype.names[:-1]:
            integral += x[name] * delta_t
        return integral
    
class DS_Integral_X_LastCpnt(MvAddFuncStorage):

    @staticmethod
    def cdssm(self, t, xp, x):
        if self.isobservedat0:
            return np.zeros(x.shape[0], dtype=np.float64)
        delta_t = float(x.dtype.names[1]) - float(x.dtype.names[0])
        integral = xp[xp.dtype.names[-1]][:, -1] * delta_t
        for name in x.dtype.names[:-1]:
            integral += x[name] * delta_t
        return integral