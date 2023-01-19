import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
​
class ChoiceModels(object):
    
    def __init__(self, path, file):
        self.data = pd.read_csv(os.path.join(path, file), sep='\s+', header=0)
    
    '''
    def log_likelihood_binary(self, para):
       xb = np.matmul(X, para.T)
        xb = np.exp(xb)
        np.sum()
        res = np.sum(self.y)*xb/(1+xb)
        return res
        
        pass
    
    def gradient(self, para):
        g = np.zeros(self.k)
        for i in range(self.k):
            x = para.copy()
            dx = abs(x[i] * 0.01)
            x[i] = x[i] + dx
            dy = self.objfun(x) - self.objfun(para)
            g[i] = dy/dx
        return g    
​
    
    def optimization(self,para):
         v = opt.minimize(self.objfun, x0=para,jac=self.gradient, method='Newton-CG', \
                          options={'maxiter': 1000, 'disp': True})  
         return [v.fun, v.x]
    '''
    
