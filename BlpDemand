import os
import pandas as pd
import numpy as np
import scipy.io
from scipy.optimize import minimize
from scipy.stats import norm
from itertools import combinations
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

class BlpDemand(object):
    
    def __init__(self, ndraws=500, tol_fp=1e-12):
        '''
        set parameters
        '''
        self.ndraws = ndraws # number of random draws in monte-carlo integration
        self.tol_fp = tol_fp # convergence tolerance level in nested fixed-point iteration

    def generate_blp_draws(self, n_markets, n_draws, n_characteristics, use_halton=True):
        """
        Generate random consumer preference draws for BLP estimation.
        
        Parameters:
        - n_markets: Number of markets (T)
        - n_draws: Number of simulation draws per market (R)
        - n_characteristics: Number of random coefficients in utility (K)
        - use_halton: If True, use Halton sequences; otherwise, use normal draws.
        
        Returns:
        - draws: A numpy array of shape (n_markets, n_draws, n_characteristics)
        """
        if use_halton:
            from scipy.stats.qmc import Halton
            sampler = Halton(d=n_characteristics, scramble=True)
            draws = sampler.random(n=n_markets * n_draws)
            draws = norm.ppf(draws)  # Transform to standard normal
        else:
            draws = np.random.randn(n_markets, n_draws, n_characteristics)
        
        return draws.reshape(n_markets, n_draws, n_characteristics)

    def _initialize_data(self, path_data, file_data, path_str_data, file_str_data):
        data = scipy.io.loadmat(os.path.join(path_data, file_data))
        model_name = scipy.io.loadmat(os.path.join(path_str_data, file_str_data))['model_name']
        v_list = ['outshr', 'const', 'mpd', 'mpg', 'air', 'space', 'hpwt', 'price', 'trend']
        df = data['share']
        for item in v_list:
            df = np.concatenate([df, data[item]], axis=1)
        df = pd.DataFrame(df, columns = ['share'] + v_list)
        df['model_name'] = model_name
        df['maker'] = df['model_name'].transform(lambda x: x[0:2])
        df = df.sort_values(by=['trend', 'maker'], ascending=[True, True]) # group products by market and by firm
        return df

    def construct_blp_ivs(self, df):
        # demand side instruments for price: sum of attributes of other products
        # from the same firm and of products from rival firms
        z_list = ['const'] # creat instruments of price from this list
        IV_list = ['const', 'mpd', 'air', 'space', 'hpwt'] # the first part of IV are exogenous regressors
        for var in z_list:
            name_own = var + "_" + "z" + "_" + "own"
            IV_list.append(name_own)
            name_rival = var + "_" + "z" + "_" + "rival"
            IV_list.append(name_rival)
            
            df[name_own] = df.groupby(['trend', 'maker'])[var].transform(lambda x: x.sum())
            df[name_own] = df[name_own] - self.df[var]
            
            df['junk']= df.groupby(['trend'])[var].transform(lambda x: x.sum()) - self.df[var]
            df[name_rival] = df['junk'] - df[name_own]
        
        return df[IV_list].to_numpy() ## the matrix of demand-side instruments

    def _construct_variables(self, df):
        '''
        construct data 
        '''
        self.nmarkets = int(df['trend'].max()) + 1 # a market is a year
        self.y_fixed = np.log(df['share']/df['outshr']) # Dependent variable in demand models without random coefficients
                            
        '''
        demand side specification
        '''
        attributes = ['const', 'hpwt', 'air', 'mpd', 'space', 'price'] # demand side variables with same order as in the BLP paper
        self.attributes_random = ['const', 'hpwt', 'air', 'mpd', 'space'] # variables with random coefficients
        self.Xmat = df[attributes].to_numpy() # X-matrix in demand model
        
        '''
        construct demand-side instruments
        '''
        self.Zmat_D =  self.construct_blp_ivs(df) # matrix of demand side instruments
        self.weight_mat_D = np.linalg.inv(np.matmul(np.transpose(self.Zmat_D), self.Zmat_D)) # initial weighting matrix in GMM estimation
        self.project_mat = self._initialize_2sls_D() # inv(X'PX)(X'P) in which P = Z*inv(Z'Z)*Z'  
        
        '''
        Take standard normal draws for approximating integrals in market share and mark-up computations.
        Better to use halton draws
        '''
        self.draws = self.generate_blp_draws(self.nmarkets, self.ndraws,len(self.attributes_random))
    
    def ols(self):
        '''
        replicate the first column of table 3
        '''
        y = np.log(self.df['share']/self.df['outshr'])
        return sm.OLS(self.y_fixed, self.Xmat).fit()
        
    def iv(self):
        '''
        replicate the second column of table 3
        '''
        #return np.matmul(self.project_mat, self.y_fixed)
        return IV2SLS(self.y_fixed, self.Xmat, self.Zmat_D).fit()
    
    def share_conditional(self, udic):
        """
        Compute conditional market shares for given utility parameters.
        
        Parameters:
        - udic: Dictionary containing 'delta' (mean utility), 'xv' (coefficients),
                and 'rdraw' (random draws for heterogeneity).
        
        Returns:
        - Market share vector for all products in the market.
        """
        # Compute exponentiated utilities
        v = np.exp(udic['delta'] + np.sum(udic['rdraw'] * udic['xv'], axis=1))

        # Compute choice probabilities (logit denominator)
        return v / (1 + np.sum(v, axis=0))  # Ensure summation across products
    
    def market_share(self, mid, delta, xv):
        draws = self.draws[mid]
        inputs = [{'delta': delta, 'xv':xv, 'rdraw':draws[r]} for r in range(self.ndraws)]
        out = map(self.share_conditional, inputs)
        return (1/self.ndraws) * np.sum(list(out), axis=0) 
       
    def fixed_point(self, pack):
        mid = pack['mid']
        df = pack['df']
        sigmas = pack['sigmas']
        s0 = df['share'].to_numpy()
        xv = sigmas * df[self.attributes_random].to_numpy()
        check = 1.0
        delta_ini = np.zeros(len(s0))
        while check > self.tol_fp:
            delta_new = delta_ini + (np.log(s0) - np.log(self.market_share(mid, delta_ini, xv)))
            check = np.max(abs(delta_new - delta_ini))
            delta_ini = delta_new
        
        return {'delta': delta_new, 'markup': mrkup} 
    
    def mean_utility(self,sigmas):
        """
        sigmas: an 1_D array with the shape (len(self.attributes_random), ), which contains
        the standard errors of random coefficients
        """
        df = self.df.copy()
        v_list = ['share', 'maker'] + self.attributes_random
        
        '''
        # step 1: solve mean utility (delta_j) from the fixed-point iteration
        '''
        df_list = [{'mid': int(mid), 'df': d[v_list], 'sigmas': sigmas} for mid, d in df.groupby(['trend'])]
        out = list(map(self.fixed_point, df_list))
                    
        '''
        step 2: uncover mean part of coefficients (beta_bar) from delta_j, which is equivalent to 
        running an IV estimation using delta_j as the dependent variable
        '''
        delta_j = tuple([i['delta'] for i in out])
        delta_j = np.concatenate(delta_j, axis=0) # an array with the shape(2217,)
        beta_bar = np.matmul(self.project_mat, delta_j) # uncover mean coefficients in demand model
        
        '''
        step 3: uncover ommited product attributes (xi_j) from delta_j and beta_bar
        '''
        xi_j = delta_j - np.matmul(self.Xmat, beta_bar)
        
        return {'beta_bar': beta_bar, 'xi_j': xi_j, 'beta_bar': beta_bar} 
    
    def GMM_obj(self, sigmas):
        
        res = self.mean_utility(sigmas)
        '''
        step 1: Demand-side moments: interact xi_j with instruments,which include exogenous regressors (veihicles' own
        exogenous attributes) and instruments for price (sum of attributes of competing products)
        '''
        xi_j = res['xi_j']
        moments_D = np.matmul(np.transpose(self.Zmat_D), xi_j) # an array with the shape (m, ), where m is the number of IVs
        
        '''
        step 2: compute the GMM objective function under the assumption that errors in demand and supply equations are independent
        '''
        f_D = np.matmul(moments_D, self.weight_mat_D)
        f_D = np.matmul(f_D, moments_D)
        return (1/len(self.df))* f_D
    
    def optimization(self, objfun, para):
        '''
        Parameters
        ----------
        objfun : a user defined objective function of para
            
        para : a 1-D array with the shape (k,), where k is the number of parameters.
        Returns
        -------
        dict
            A dictionary containing estimation results
        '''
        v = opt.minimize(objfun, x0=para, jac=None, method='BFGS', 
                          options={'maxiter': 1000, 'disp': True})  
        return {'obj':v.fun, "Coefficients": v.x}

if __name__ == "__main__":
    blp = BlpDemand()
    df = blp._initialize_data("/kaggle/input/blp-data/", "BLP_data.mat", "/kaggle/input/blp-str-data/", "BLP_data_str.mat")
    print("Variable list:", df.columns)
    
