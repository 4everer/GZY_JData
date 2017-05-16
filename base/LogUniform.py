# define log uniform distribution, useful when use RandomizedSearchCV to search hyperparameters
from scipy.stats import rv_continuous
from scipy.stats import uniform
class LogUniform(rv_continuous):
    'Log Uniform distribution'
    def _pdf(self,x,p,q):
        if x>=np.exp(p) and x<=np.exp(q):
            return 1.0/((q-p)*x)
        else:
            return 0
    def _rvs(self,p,q): # important to explicitly define the _RVS, otherwise extremely slow
        size=self._size
        #print args
        loc=p
        scale=q-p
        return np.exp(uniform.rvs(loc=loc,scale=scale,size=size)) 
    def _argcheck(self,p,q):
        return True