# General imports.
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
import math
from sklearn.preprocessing import StandardScaler


class DataGeneration():
    
    
    def generate_one_of_each_fixed(self,
                                   n: int = 1000, 
                                   noise_distribution: str = "bernoulli",
                                   x_causes_y: bool = True,
                                   m_structure: bool = False,
                                   butterfly_structure: bool = False,
                                   indirect_mediators: bool = False,
                                   function: str = "linear",
                                   coefficient_range: tuple = (1, 2),
                                   verbose: bool = False):

        '''
        This function generates a 10-node DAG with one variable representing each case.
        All variables are discrete random variables with finite support. 
        
        Parameters to noise distributions and SEM coefficients are all fixed for
        reproducibility purposes.

        Parameters:
        -----------
        n: int = 1000
            Total observations to sample.
        noise_distribution: str = "laplace"
            Distributional form of the noise (i.e. noise) term of the structural equation
            Y = f(X) + noise.
        x_causes_y: bool = True
            Boolean flag indicating whether X is a cause of Y. If set to false, there is
            no causal relation between X and Y. Note that Z is always a cause of both X 
            and Y by definition.
        verbose: bool = False
            Flag for displaying dataframes.

        Return:
        -------
        df_vars, df_noise
            df_vars contains variable data.
            df_noise contains all noise values.
        '''

        # Construct noise terms of structural equation.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        # Per http://proceedings.mlr.press/v9/peters10a/peters10a.pdf, we offer the following 
        # noise distributions: binomial, geometric, hypergeometric, negative binomial, and Poisson.
        total_vars = 10
        if m_structure:
            total_vars += 3
        if butterfly_structure:
            total_vars += 3
        if indirect_mediators:
            total_vars += 1
        noise = []
        if noise_distribution == "binomial":
            for var in range(total_vars):
                noise.append(np.random.binomial(n = 2, p = 0.45, size = n).reshape(-1, 1))
        elif noise_distribution == "bernoulli":
            for var in range(total_vars):
                noise.append(np.random.binomial(n = 1, p = 0.6, size = n).reshape(-1, 1))
        elif noise_distribution == "geometric":
            for var in range(total_vars):
                noise.append(np.random.geometric(p = 0.95, size = n).reshape(-1, 1))
        elif noise_distribution == "hypergeometric":
            for var in range(total_vars):
                noise.append(np.random.hypergeometric(ngood = 1, 
                                                      nbad = 1, 
                                                      nsample = 1, 
                                                      size = n).reshape(-1, 1))
        elif noise_distribution == "negative binomial":
            for var in range(total_vars):
                noise.append(np.random.negative_binomial(n = 2, p = 0.8, size = n).reshape(-1, 1))
        elif noise_distribution == "poisson":
            for var in range(total_vars):
                noise.append(np.random.poisson(lam = 0.1, size = n).reshape(-1, 1))
        elif noise_distribution == "mixed":  
            for i in range(total_vars):
                if i % 2 == 0:
                    # Hypergeometric.
                    noise.append(np.random.hypergeometric(ngood = 1, 
                                                          nbad = 1, 
                                                          nsample = 1, 
                                                          size = n).reshape(-1, 1))   
                else:
                    # Bernoulli.
                    noise.append(np.random.binomial(n = 1, p = 0.6, size = n).reshape(-1, 1))
        else:
            raise NameError("noise_distribution must take a value in ['binomial', 'bernoulli', 'negative binomial', 'geometric', 'hypergeometric', 'poisson', 'mixed']")

        # Define causal mechanisms.
        # Example from http://proceedings.mlr.press/v9/peters10a/peters10a.pdf
        # fun = lambda x: np.around(0.5 * x**2)
        #fun = lambda x: np.around(np.sqrt(x))
        #fun = lambda x: x
        
        # Define causal mechanisms.
        if function == "linear":
            if butterfly_structure:
                if noise_distribution == "bernoulli":
                    c = 1.9
                else:
                    c = 2.8
            elif m_structure:
                c = 1.5
            elif x_causes_y:
                c = 0.3
            else:
                c = 0.45
            fun = lambda x: np.around(c * x) # Originally used 0.3.
        elif function == "quadratic":
            if noise_distribution == "bernoulli":
                c = -1.4
            else:
                c = 0.4
            fun = lambda x: np.around(c * x**2) # low = 0.299, high = 0.51
        elif function == "square root":
            fun = lambda x: np.around(2.0 * np.sqrt(abs(x))) # low = 0.95, high = 1.1
        elif function == "cube root":
            if not butterfly_structure:
                if noise_distribution == "bernoulli":
                    c = 1.2
                else:
                    c = 0.7
            else:
                c = 3.4
            fun = lambda x: np.around(c * np.cbrt(abs(x))) # low = 0.95, high = 1.1
        else:
            raise NameError("Valid `function` values are in ['linear', 'square root', 'cube root', 'quadratic'].")
        
        # Define coefficient generator.
        coeff = lambda : np.random.randint(low = coefficient_range[0], high = coefficient_range[1], size = 1)
        
        # Define variables.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        # Causes X    : Z1, Z5.
        # Causes Y    : X, Z1, Z3, Z4.
        # Caused by X : Z2, Z3, Z7.
        # Caused by Y : Z2, Z6.
        
        if m_structure and butterfly_structure:
            indirect_mediator_noise = 16
        elif m_structure or butterfly_structure:
            indirect_mediator_noise = 13
        else:
            indirect_mediator_noise = 10
            
        if x_causes_y:
            xy_coeff = 1
        else:
            xy_coeff = 0
        if function not in ["cube root", "square root"]:
            M1 = np.zeros_like(noise[0])
            M2 = np.zeros_like(noise[0])
            M3 = np.zeros_like(noise[0])
            B1 = np.zeros_like(noise[0])
            B2 = np.zeros_like(noise[0])
            B3 = np.zeros_like(noise[0])
            Z3_2 = np.zeros_like(noise[0])
            
            Z1 = noise[2]
            Z4 = noise[5]
            Z5 = noise[6]
            Z8 = noise[9]
            if m_structure:
                M1 = noise[10]
                M2 = noise[11]
                M3 = coeff()*fun(M1 + M2) + noise[12]
            if butterfly_structure:
                if m_structure:
                    B1 = noise[13]
                    B2 = noise[14]
                    B3 = coeff()*fun(B1 + B2) + noise[15]
                else:
                    B1 = noise[10]
                    B2 = noise[11]
                    B3 = coeff()*fun(B1 + B2) + noise[12]
            X  = coeff()*fun(Z1 + Z5 + M1 + B1 + B3) + noise[0]
            Z3 = coeff()*fun(X) + noise[4]
            if indirect_mediators:
                Z3_2 = coeff()*fun(Z3) + noise[indirect_mediator_noise]
                Y  = coeff()*fun(xy_coeff*X + Z1 + Z3_2 + Z4 + M2 + B2 + B3) + noise[1]
            else:
                Y  = coeff()*fun(xy_coeff*X + Z1 + Z3 + Z4 + M2 + B2 + B3) + noise[1]
            Z2 = coeff()*fun(X + Y) + noise[3]
            Z6 = coeff()*fun(Y) + noise[7]
            Z7 = coeff()*fun(X) + noise[8]
        else:
            Z1 = noise[2]
            Z4 = noise[5]
            Z5 = noise[6]
            Z8 = noise[9]
            X = coeff()*fun(Z1) + coeff()*fun(Z5) + noise[0]
            if m_structure:
                M1 = noise[10]
                M2 = noise[11]
                M3 = coeff()*fun(M1) + coeff()*fun(M2) + noise[12]
                X = X + coeff()*fun(M1)
            if butterfly_structure:
                if m_structure:
                    B1 = noise[13]
                    B2 = noise[14]
                    B3 = coeff()*fun(B1) + coeff()*fun(B2) + noise[15]
                else:
                    B1 = noise[10]
                    B2 = noise[11]
                    B3 = coeff()*fun(B1) + coeff()*fun(B2) + noise[12]
                X = X + coeff()*fun(B1) + coeff()*fun(B3)
            Z3 = coeff()*fun(X) + noise[4]
            if indirect_mediators:
                Z3_2 = coeff()*fun(Z3) + noise[indirect_mediator_noise]
                Y = xy_coeff*fun(X) + coeff()*fun(Z1) + coeff()*fun(Z3_2) + coeff()*fun(Z4) + noise[1]
            else:
                Y = xy_coeff*fun(X) + coeff()*fun(Z1) + coeff()*fun(Z3) + coeff()*fun(Z4) + noise[1]
            if m_structure:
                Y = Y + coeff()*fun(M2)
            if butterfly_structure:
                Y = Y + coeff()*fun(B2) + coeff()*fun(B3)
            Z2 = coeff()*fun(X) + coeff()*fun(Y) + noise[3]
            Z6 = coeff()*fun(Y) + noise[7]
            Z7 = coeff()*fun(X) + noise[8]
            
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z2": Z2.reshape(-1), 
                                "Z3": Z3.reshape(-1), 
                                "Z4": Z4.reshape(-1), 
                                "Z5": Z5.reshape(-1),
                                "Z6": Z6.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1)}).astype(int)
        df_noise = pd.DataFrame({"X": noise[0].reshape(-1), 
                                 "Y": noise[1].reshape(-1), 
                                 "Z1": noise[2].reshape(-1),
                                 "Z2": noise[3].reshape(-1), 
                                 "Z3": noise[4].reshape(-1), 
                                 "Z4": noise[5].reshape(-1), 
                                 "Z5": noise[6].reshape(-1),
                                 "Z6": noise[7].reshape(-1), 
                                 "Z7": noise[8].reshape(-1), 
                                 "Z8": noise[9].reshape(-1)}).astype(int)
        
        if m_structure:
            df_vars["M1"]  = M1.reshape(-1).astype(int)
            df_vars["M2"]  = M2.reshape(-1).astype(int)
            df_vars["M3"]  = M3.reshape(-1).astype(int)
            #df_noise["M1"] = M1.reshape(-1).astype(int)
            #df_noise["M2"] = M2.reshape(-1).astype(int)
            #df_noise["M3"] = M3.reshape(-1).astype(int)

        if butterfly_structure:
            df_vars["B1"]  = B1.reshape(-1).astype(int)
            df_vars["B2"]  = B2.reshape(-1).astype(int)
            df_vars["B3"]  = B3.reshape(-1).astype(int)
            #df_noise["B1"] = B1.reshape(-1).astype(int)
            #df_noise["B2"] = B2.reshape(-1).astype(int)
            #df_noise["B3"] = B3.reshape(-1).astype(int)
            
        if indirect_mediators:
            df_vars["Z3.2"]  = Z3_2.reshape(-1).astype(int)
            #df_noise["Z3.2"]  = Z3_2.reshape(-1).astype(int)

        if verbose:
            print("VARIABLES:")
            display(df_vars.head())
            print("NOISE:")
            display(df_noise.head())

        return df_vars, df_noise
    
    
    def generate_one_of_each_continuous(self,
                                         n: int = 1000, 
                                         noise_distribution: str = "laplace",
                                         x_causes_y: bool = True,
                                         m_structure: bool = False,
                                         butterfly_structure: bool = False,
                                         parents: bool = False,
                                         children: bool = False,
                                         function: str = "polynomial",
                                         degree: int = 1,
                                         coefficient_range: tuple = (1.25, 2.5),
                                         standardize_noise: bool = True,
                                         standardize_polynomial: bool = True,
                                         noise_scalar: float = 1.0,
                                         verbose: bool = False):

        '''
        This function generates a 10-node DAG with variables representing each case.
        All variables are continuous random variables. 

        Parameters:
        -----------
        n: int = 1000
            Total observations to sample.
        noise_distribution: str = "laplace"
            Distributional form of the noise (i.e. noise) term of the structural equation
            Y = f(X) + noise.
        x_causes_y: bool = True
            Boolean flag indicating whether X is a cause of Y. If set to false, there is
            no causal relation between X and Y. Note that Z is always a cause of both X 
            and Y by definition.
        verbose: bool = False
            Flag for displaying dataframes.

        Return:
        -------
        df_vars, df_noise
            df_vars contains variable data.
            df_noise contains all noise values.
        '''

        # Construct noise terms of structural equation.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        total_vars = 10
        if m_structure:
            total_vars += 3
        if butterfly_structure:
            total_vars += 3
        noise = []
        if noise_distribution == "laplace":
            for var in range(total_vars):
                noise.append(np.random.laplace(loc = 0.0, scale = 1.0, size = n))
        elif noise_distribution == "uniform":
            for var in range(total_vars):
                noise.append(np.random.uniform(low = -1.0, high = 1.0, size = n))
        elif noise_distribution == "gaussian":
            for var in range(total_vars):
                noise.append(np.random.normal(loc = 0.0, scale = 1.0, size = n))
        elif noise_distribution == "exponential":
            for var in range(total_vars):
                noise.append(np.random.exponential(scale = 1.0, size = n))
        elif noise_distribution == "mixed":
            for i in range(total_vars):
                if i % 2 == 0:
                    noise.append(np.random.laplace(loc = 0.0, scale = 1.0, size = n))
                else:
                    noise.append(np.random.exponential(scale = 1.0, size = n))
        else:
            raise NameError("noise_distribution must take a value in ['laplace', 'uniform', 'gaussian', 'exponential', 'mixed']")
            
        # Make noise zero mean and unit variance.
        if standardize_noise:
            for i in range(total_vars):
                scaler = StandardScaler()
                noise[i] = scaler.fit_transform(noise[i].reshape(-1, 1))
        else:
            for i in range(total_vars):
                noise[i] = noise[i].reshape(-1, 1)

        # Define causal mechanisms.
        if standardize_polynomial:
            if degree == 1:
                if noise_distribution == "gaussian":
                    c = 0.8
                elif noise_distribution == "uniform":
                    c = 1.2
                elif noise_distribution == "exponential":
                    c = 0.25
                else:
                    c = -1.3
            elif degree == 2:
                c = 0.2
            else:
                c = 1.0
            #polynomial = lambda x: self.scale(c * x**degree)
            polynomial = lambda x: self.scale(x**degree)
        else:
            polynomial = lambda x: x**degree
        log = lambda x: 1.5 * np.log((abs(x) + 1e-6))
        sigmoid = lambda x: 2.5 * np.array([1 / (1 + math.exp(-y)) for y in x]).reshape(-1,1)
        sqrt = lambda x: 1.3 * np.sqrt(abs(x))
        cbrt = lambda x: np.cbrt(abs(x))
        if function == "polynomial":
            fun = polynomial
        elif function == "square root":
            fun = sqrt
        elif function == "cube root":
            fun = cbrt
        elif function == "log":
            fun = log
        elif function == "sigmoid":
            fun = sigmoid
        else:
            raise NameError("Valid `function` values are 'polynomial', 'square root', 'cube root', 'log', and 'sigmoid'.")
        
        # Define coefficient generator.
        coeff = lambda : np.random.uniform(coefficient_range[0], coefficient_range[1]) #* random.choice([-1, 1])
        
        # Define variables.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8
        # Causes X    : Z1, Z5.
        # Causes Y    : X, Z1, Z3, Z4.
        # Caused by X : Z2, Z3, Z7.
        # Caused by Y : Z2, Z6.
        if x_causes_y:
            xy_coeff = np.random.uniform(coefficient_range[0], coefficient_range[1])
            self.xy_coeff = xy_coeff
        else:
            xy_coeff = 0
        #print("True direct effect of X on Y:", xy_coeff)
        Z1 = noise[2]
        Z4 = noise[5]
        Z5 = noise[6]
        Z8 = noise[9]
        X = coeff()*fun(Z1) + coeff()*fun(Z5) + noise[0]
        if m_structure:
            M1 = noise[10]
            M2 = noise[11]
            M3 = coeff()*fun(M1) + coeff()*fun(M2) + noise[12]
            X = X + coeff()*fun(M1)
        if butterfly_structure:
            if m_structure:
                B1 = noise[13]
                B2 = noise[14]
                B3 = coeff()*fun(B1) + coeff()*fun(B2) + noise[15]
            else:
                B1 = noise[10]
                B2 = noise[11]
                B3 = coeff()*fun(B1) + coeff()*fun(B2) + noise[12]
            X = X + coeff()*fun(B1) + coeff()*fun(B3)
        Z3 = coeff()*fun(X) + noise[4]
        Y = xy_coeff*fun(X) + coeff()*fun(Z1) + coeff()*fun(Z3) + coeff()*fun(Z4) + noise[1]
        if m_structure:
            Y = Y + coeff()*fun(M2)
        if butterfly_structure:
            Y = Y + coeff()*fun(B2) + coeff()*fun(B3)
        Z2 = coeff()*fun(X) + coeff()*fun(Y) + noise[3]
        Z6 = coeff()*fun(Y) + noise[7]
        Z7 = coeff()*fun(X) + noise[8]
            
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z2": Z2.reshape(-1), 
                                "Z3": Z3.reshape(-1), 
                                "Z4": Z4.reshape(-1), 
                                "Z5": Z5.reshape(-1),
                                "Z6": Z6.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1)})
        df_noise = pd.DataFrame({"X": noise[0].reshape(-1), 
                                 "Y": noise[1].reshape(-1), 
                                 "Z1": noise[2].reshape(-1),
                                 "Z2": noise[3].reshape(-1), 
                                 "Z3": noise[4].reshape(-1), 
                                 "Z4": noise[5].reshape(-1), 
                                 "Z5": noise[6].reshape(-1),
                                 "Z6": noise[7].reshape(-1), 
                                 "Z7": noise[8].reshape(-1), 
                                 "Z8": noise[9].reshape(-1)})
        var_names = ["Z" + str(i) for i in range(1, 9)]
        var_names = ["X", "Y"] + var_names
        
        if m_structure:
            df_vars["M1"]  = M1.reshape(-1)
            df_vars["M2"]  = M2.reshape(-1)
            df_vars["M3"]  = M3.reshape(-1)
            df_noise["M1"] = M1.reshape(-1)
            df_noise["M2"] = M2.reshape(-1)
            df_noise["M3"] = M3.reshape(-1)

        if butterfly_structure:
            df_vars["B1"]  = B1.reshape(-1)
            df_vars["B2"]  = B2.reshape(-1)
            df_vars["B3"]  = B3.reshape(-1)
            df_noise["B1"] = B1.reshape(-1)
            df_noise["B2"] = B2.reshape(-1)
            df_noise["B3"] = B3.reshape(-1)
        
        if verbose:
            print("VARIABLES:")
            display(df_vars.head())
            print("NOISE:")
            display(df_noise.head())

        return df_vars, df_noise
    
    
    def generate_one_of_each_discrete(self,
                                      n: int = 1000, 
                                      noise_distribution: str = "binomial",
                                      x_causes_y: bool = True,
                                      m_structure: bool = False,
                                      butterfly_structure: bool = False,
                                      parents: bool = False,
                                      children: bool = False,
                                      binary_exposure: bool = False,
                                      function: str = "linear",
                                      coefficient_range: tuple = (1, 3),
                                      verbose: bool = False):

        '''
        This function generates a 10-node DAG with variables representing each case.
        All variables are discrete random variables with finite support. 

        Parameters:
        -----------
        n: int = 1000
            Total observations to sample.
        noise_distribution: str = "laplace"
            Distributional form of the noise (i.e. noise) term of the structural equation
            Y = f(X) + noise.
        x_causes_y: bool = True
            Boolean flag indicating whether X is a cause of Y. If set to false, there is
            no causal relation between X and Y. Note that Z is always a cause of both X 
            and Y by definition.
        verbose: bool = False
            Flag for displaying dataframes.

        Return:
        -------
        df_vars, df_noise
            df_vars contains variable data.
            df_noise contains all noise values.
        '''

        # Construct noise terms of structural equation.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        # Per http://proceedings.mlr.press/v9/peters10a/peters10a.pdf, we offer the following 
        # noise distributions: binomial, geometric, hypergeometric, negative binomial, and Poisson.
        total_vars = 10
        if m_structure:
            total_vars += 3
        if butterfly_structure:
            total_vars += 3
        if parents:
            total_vars += 8
        if children:
            total_vars += 8
        noise = []
        if noise_distribution == "binomial":
            for var in range(total_vars):
                p = np.random.uniform(low = 0.45, high = 0.55, size = 1)
                noise.append(np.random.binomial(n = 2, p = p, size = n).reshape(-1, 1))
        elif noise_distribution == "bernoulli":
            for var in range(total_vars):
                p = np.random.uniform(low = 0.4, high = 0.6, size = 1)
                noise.append(np.random.binomial(n = 1, p = p, size = n).reshape(-1, 1))
        elif noise_distribution == "geometric":
            for var in range(total_vars):
                #p = np.random.uniform(low = 0.75, high = 0.95, size = 1)
                noise.append(np.random.geometric(p = 0.95, size = n).reshape(-1, 1))
        elif noise_distribution == "hypergeometric":
            for var in range(total_vars):
                ngood = np.random.randint(low = 1, high = 2, size = 1)
                nbad = np.random.randint(low = 1, high = 2, size = 1)
                noise.append(np.random.hypergeometric(ngood = ngood, 
                                                      nbad = nbad, 
                                                      nsample = max(ngood + nbad - 2, 1), 
                                                      size = n).reshape(-1, 1))
        elif noise_distribution == "negative binomial":
            for var in range(total_vars):
                p = np.random.uniform(low = 0.75, high = 0.95, size = 1)
                noise.append(np.random.negative_binomial(n = 2, p = p, size = n).reshape(-1, 1))
        elif noise_distribution == "poisson":
            for var in range(total_vars):
                #lam = np.random.uniform(low = 0.1, high = 0.5, size = 1)
                noise.append(np.random.poisson(lam = 0.1, size = n).reshape(-1, 1))
        elif noise_distribution == "mixed":  
            for i in range(total_vars):
                if i % 2 == 0:
                    ngood = np.random.randint(low = 1, high = 2, size = 1)
                    nbad = np.random.randint(low = 1, high = 2, size = 1)
                    noise.append(np.random.hypergeometric(ngood = ngood, 
                                                          nbad = nbad, 
                                                          nsample = max(ngood + nbad - 2, 1), 
                                                          size = n).reshape(-1, 1))   
                else:
                    p = np.random.uniform(low = 0.45, high = 0.55, size = 1)
                    noise.append(np.random.binomial(n = 2, p = p, size = n).reshape(-1, 1))
        else:
            raise NameError("noise_distribution must take a value in ['binomial', 'bernoulli', 'negative binomial', 'geometric', 'hypergeometric', 'poisson', 'mixed']")

        # Define causal mechanisms.
        # Example from http://proceedings.mlr.press/v9/peters10a/peters10a.pdf
        # fun = lambda x: np.around(0.5 * x**2)
        #fun = lambda x: np.around(np.sqrt(x))
        #fun = lambda x: x
        
        # Define causal mechanisms.
        rand_1 = lambda : np.random.uniform(low = 0.299, high = 0.51, size = 1)
        #rand_1 = lambda : np.random.uniform(low = 0.299, high = 0.91, size = 1)
        rand_2 = lambda : np.random.uniform(low = 0.95, high = 1.1, size = 1)
        if function == "linear":
            fun = lambda x: np.around(rand_1() * x) # Originally used 0.3.
        elif function == "quadratic":
            fun = lambda x: np.around(rand_1() * x**2)
        elif function == "square root":
            fun = lambda x: np.around(rand_2() * np.sqrt(abs(x)))
        elif function == "cube root":
            fun = lambda x: np.around(rand_2() * np.cbrt(abs(x)))
        else:
            raise NameError("Valid `function` values are in ['linear', 'square root', 'cube root', 'quadratic'].")
        
        # Define coefficient generator.
        coeff = lambda : np.random.randint(low = coefficient_range[0], high = coefficient_range[1], size = 1) #* random.choice([-1, 1])
        #coeff = lambda : np.random.uniform(coefficient_range[0], coefficient_range[1])
        
        # Define variables.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        # Causes X    : Z1, Z5.
        # Causes Y    : X, Z1, Z3, Z4.
        # Caused by X : Z2, Z3, Z7.
        # Caused by Y : Z2, Z6.
        if x_causes_y:
            #xy_coeff = np.random.randint(low = coefficient_range[0], high = coefficient_range[1], size = 1)
            xy_coeff = 1
        else:
            xy_coeff = 0
        if function != "cube root":
            M1 = np.zeros_like(noise[0])
            M2 = np.zeros_like(noise[0])
            M3 = np.zeros_like(noise[0])
            B1 = np.zeros_like(noise[0])
            B2 = np.zeros_like(noise[0])
            B3 = np.zeros_like(noise[0])
            
            Z1 = noise[2]
            Z4 = noise[5]
            Z5 = noise[6]
            Z8 = noise[9]
            if m_structure:
                M1 = noise[10]
                M2 = noise[11]
                M3 = coeff()*fun(M1 + M2) + noise[12]
            if butterfly_structure:
                if m_structure:
                    B1 = noise[13]
                    B2 = noise[14]
                    B3 = coeff()*fun(B1 + B2) + noise[15]
                else:
                    B1 = noise[10]
                    B2 = noise[11]
                    B3 = coeff()*fun(B1 + B2) + noise[12]
            X  = coeff()*fun(Z1 + Z5 + M1 + B1 + B3) + noise[0]
            Z3 = coeff()*fun(X) + noise[4]
            Y  = coeff()*fun(xy_coeff*X + Z1 + Z3 + Z4 + M2 + B2 + B3) + noise[1]
            Z2 = coeff()*fun(X + Y) + noise[3]
            Z6 = coeff()*fun(Y) + noise[7]
            Z7 = coeff()*fun(X) + noise[8]
        else:
            Z1 = noise[2]
            Z4 = noise[5]
            Z5 = noise[6]
            Z8 = noise[9]
            X = coeff()*fun(Z1) + coeff()*fun(Z5) + noise[0]
            if m_structure:
                M1 = noise[10]
                M2 = noise[11]
                M3 = coeff()*fun(M1) + coeff()*fun(M2) + noise[12]
                X = X + coeff()*fun(M1)
            if butterfly_structure:
                if m_structure:
                    B1 = noise[13]
                    B2 = noise[14]
                    B3 = coeff()*fun(B1) + coeff()*fun(B2) + noise[15]
                else:
                    B1 = noise[10]
                    B2 = noise[11]
                    B3 = coeff()*fun(B1) + coeff()*fun(B2) + noise[12]
                X = X + coeff()*fun(B1) + coeff()*fun(B3)
            Z3 = coeff()*fun(X) + noise[4]
            Y = xy_coeff*fun(X) + coeff()*fun(Z1) + coeff()*fun(Z3) + coeff()*fun(Z4) + noise[1]
            if m_structure:
                Y = Y + coeff()*fun(M2)
            if butterfly_structure:
                Y = Y + coeff()*fun(B2) + coeff()*fun(B3)
            Z2 = coeff()*fun(X) + coeff()*fun(Y) + noise[3]
            Z6 = coeff()*fun(Y) + noise[7]
            Z7 = coeff()*fun(X) + noise[8]
        if children:
            idx = 10
            if m_structure:
                idx += 3
            if butterfly_structure:
                idx += 3
            Z1_child = coeff()*fun(Z1) + noise[idx]
            Z2_child = coeff()*fun(Z2) + noise[idx + 1]
            Z3_child = coeff()*fun(Z3) + noise[idx + 2]
            Z4_child = coeff()*fun(Z4) + noise[idx + 3]
            Z5_child = coeff()*fun(Z5) + noise[idx + 4]
            Z6_child = coeff()*fun(Z6) + noise[idx + 5]
            Z7_child = coeff()*fun(Z7) + noise[idx + 6]
            Z8_child = coeff()*fun(Z8) + noise[idx + 7]
            
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z2": Z2.reshape(-1), 
                                "Z3": Z3.reshape(-1), 
                                "Z4": Z4.reshape(-1), 
                                "Z5": Z5.reshape(-1),
                                "Z6": Z6.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1)}).astype(int)
        df_noise = pd.DataFrame({"X": noise[0].reshape(-1), 
                                 "Y": noise[1].reshape(-1), 
                                 "Z1": noise[2].reshape(-1),
                                 "Z2": noise[3].reshape(-1), 
                                 "Z3": noise[4].reshape(-1), 
                                 "Z4": noise[5].reshape(-1), 
                                 "Z5": noise[6].reshape(-1),
                                 "Z6": noise[7].reshape(-1), 
                                 "Z7": noise[8].reshape(-1), 
                                 "Z8": noise[9].reshape(-1)}).astype(int)

        if m_structure:
            df_vars["M1"]  = M1.reshape(-1).astype(int)
            df_vars["M2"]  = M2.reshape(-1).astype(int)
            df_vars["M3"]  = M3.reshape(-1).astype(int)
            df_noise["M1"] = M1.reshape(-1).astype(int)
            df_noise["M2"] = M2.reshape(-1).astype(int)
            df_noise["M3"] = M3.reshape(-1).astype(int)

        if butterfly_structure:
            df_vars["B1"]  = B1.reshape(-1).astype(int)
            df_vars["B2"]  = B2.reshape(-1).astype(int)
            df_vars["B3"]  = B3.reshape(-1).astype(int)
            df_noise["B1"] = B1.reshape(-1).astype(int)
            df_noise["B2"] = B2.reshape(-1).astype(int)
            df_noise["B3"] = B3.reshape(-1).astype(int)
            
        if children:
            df_vars["Z1 child"]  = Z1_child.reshape(-1).astype(int)
            df_vars["Z2 child"]  = Z2_child.reshape(-1).astype(int)
            df_vars["Z3 child"]  = Z3_child.reshape(-1).astype(int)
            df_vars["Z4 child"]  = Z4_child.reshape(-1).astype(int)
            df_vars["Z5 child"]  = Z5_child.reshape(-1).astype(int)
            df_vars["Z6 child"]  = Z6_child.reshape(-1).astype(int)
            df_vars["Z7 child"]  = Z7_child.reshape(-1).astype(int)
            df_vars["Z8 child"]  = Z8_child.reshape(-1).astype(int)
        
        if verbose:
            print("VARIABLES:")
            display(df_vars.head())
            print("NOISE:")
            display(df_noise.head())
            
        return df_vars, df_noise
    
    
    def generate_complex_2(self,
                           n: int = 5000, 
                           noise_distribution: str = "bernoulli",
                           x_causes_y: bool = True,
                           function: str = "linear",
                           coefficient_range: tuple = (1, 2),
                           verbose: bool = False):

        '''
        This function generates a complex DAG as a synthetic benchmark.
        All variables are discrete random variables with finite support. 
        
        Parameters to noise distributions and SEM coefficients are all fixed for
        reproducibility purposes.

        Parameters:
        -----------
        n: int = 5000
            Total observations to sample.
        noise_distribution: str = "laplace"
            Distributional form of the noise (i.e. noise) term of the structural equation
            Y = f(X) + noise.
        x_causes_y: bool = True
            Boolean flag indicating whether X is a cause of Y. If set to false, there is
            no causal relation between X and Y. Note that Z is always a cause of both X 
            and Y by definition.
        verbose: bool = False
            Flag for displaying dataframes.

        Return:
        -------
        df_vars
            df_vars contains variable data.
        df_adj
            directed adjacency matrix as dataframe.
        '''

        # Construct noise terms of structural equation.
        total_vars = 23
        noise = []
        if noise_distribution == "bernoulli":
            for var in range(total_vars):
                noise.append(np.random.binomial(n = 1, p = 0.6, size = n).reshape(-1, 1))
        elif noise_distribution == "hypergeometric":
            for var in range(total_vars):
                noise.append(np.random.hypergeometric(ngood = 1, 
                                                      nbad = 1, 
                                                      nsample = 1, 
                                                      size = n).reshape(-1, 1))
        elif noise_distribution == "mixed":  
            for i in range(total_vars):
                if i % 2 == 0:
                    # Hypergeometric.
                    noise.append(np.random.hypergeometric(ngood = 1, 
                                                          nbad = 1, 
                                                          nsample = 1, 
                                                          size = n).reshape(-1, 1))   
                else:
                    # Bernoulli.
                    noise.append(np.random.binomial(n = 1, p = 0.6, size = n).reshape(-1, 1))
        else:
            raise NameError("noise_distribution must take a value in ['bernoulli', 'hypergeometric', 'mixed']")

        # Define causal mechanisms.
        if function == "linear":
            fun = lambda x: np.around(1.9 * x) # Originally used 0.3.
        elif function == "quadratic":
            fun = lambda x: np.around(0.4 * x**2) # 0.4
        elif function == "square root":
            fun = lambda x: np.around(2.0 * np.sqrt(abs(x))) # low = 0.95, high = 1.1
        elif function == "cube root":
            fun = lambda x: np.around(1.2 * np.cbrt(abs(x))) # low = 0.95, high = 1.1
        else:
            raise NameError("Valid `function` values are in ['linear', 'square root', 'cube root', 'quadratic'].")
        
        # Define coefficient generator.
        coeff = lambda : np.random.randint(low = coefficient_range[0], high = coefficient_range[1], size = 1)
        
        # Define variables.
        if x_causes_y:
            xy_coeff = 1
        else:
            xy_coeff = 0  
        X_idx = 0
        Y_idx = 1
        Z1_idx = 2
        Z2_idx = 3
        Z3_idx = 4
        Z3_2_idx = 5
        Z4_idx = 6
        Z5_idx = 7
        Z6_idx = 8
        Z7_idx = 9
        Z8_idx = 10
        M1_idx = 11
        M2_idx = 12
        M3_idx = 13
        B1_idx = 14
        B2_idx = 15
        B3_idx = 16
        C1_idx = 17
        C2_idx = 18
        C3_idx = 19
        C4_idx = 20
        C5_idx = 21
        C6_idx = 22  
        
        # Pretreatment variables on simple paths and isolated vars.
        Z1 = noise[Z1_idx]
        Z4 = noise[Z4_idx]
        Z5 = noise[Z5_idx]
        Z8 = noise[Z8_idx]
        
        # Complex backdoor path.
        C1 = noise[C1_idx]
        C2 = noise[C2_idx]
        C3 = coeff()*fun(C1 + C2) + noise[C3_idx]
        C4 = coeff()*fun(C1 + C3 + Z5) + noise[C4_idx]
        C5 = coeff()*fun(C2 + C3 + Z4) + noise[C5_idx]
        C6 = coeff()*fun(C1 + C2) + noise[C6_idx]  
        
        # M-structure.
        M1 = noise[M1_idx]
        M2 = noise[M2_idx]
        M3 = coeff()*fun(M1 + M2) + noise[M3_idx]

        # Butterfly structure.
        B1 = noise[B1_idx]
        B2 = noise[B2_idx]
        B3 = coeff()*fun(B1 + B2) + noise[B3_idx]
        
        # X, Y, and post-treatment variables.
        X  = (coeff()*fun(Z1 + Z5 + M1 + B1 + B3 + C4) + noise[X_idx])
        Z3 = coeff()*fun(X) + noise[Z3_idx]
        Z3_2 = coeff()*fun(Z3) + noise[Z3_2_idx]
        Y  = (coeff()*fun(xy_coeff*X + Z1 + Z3_2 + Z4 + M2 + B2 + B3 + C5) + noise[Y_idx])
        Z2 = coeff()*fun(X + Y) + noise[Z2_idx]
        Z6 = coeff()*fun(Y) + noise[Z6_idx]
        Z7 = coeff()*fun(X) + noise[Z7_idx]
        
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z2": Z2.reshape(-1), 
                                "Z3.0": Z3.reshape(-1), 
                                "Z3.1": Z3_2.reshape(-1),
                                "Z4": Z4.reshape(-1), 
                                "Z5": Z5.reshape(-1),
                                "Z6": Z6.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1),
                                "Z5 (M1)": M1.reshape(-1),
                                "Z4 (M2)": M2.reshape(-1),
                                "Z2 (M3)": M3.reshape(-1),
                                "Z1 (B1)": B1.reshape(-1), 
                                "Z1 (B2)": B2.reshape(-1), 
                                "Z1 (B3)": B3.reshape(-1),
                                "Z1 (C1)": C1.reshape(-1),
                                "Z1 (C2)": C2.reshape(-1),
                                "Z1 (C3)": C3.reshape(-1),
                                "Z1 (C4)": C4.reshape(-1),
                                "Z1 (C6)": C5.reshape(-1),
                                "Z2 (C6)": C6.reshape(-1),
                                }).astype(int)
 
        # Construct adjacency matrix.
        # [i,j] means i causes j.
        '''
        X causes: Y, Z2, Z3, Z7
        Y causes: Z2, Z6
        Z1 causes: X, Y
        Z3 causes: Z3_2
        Z3_2 causes: Y
        Z4 causes: Y, C5
        Z5 causes: X, C4
        M1 causes: X, M3
        M2 causes: Y, M3
        B1 causes: X, B3
        B2 causes: Y, B3
        B3 causes: X, Y
        C1 causes: C3, C4, C6
        C2 causes: C3, C5, C6
        C3 causes: C4, C5
        C4 causes: X
        C5 causes: Y
        '''
        c = xy_coeff
        adjacency = np.zeros((total_vars, total_vars), dtype = int)
        
        adjacency[X_idx, Y_idx] = 1
        adjacency[X_idx, Z2_idx] = 1
        adjacency[X_idx, Z3_idx] = 1
        adjacency[X_idx, Z7_idx] = 1
        
        adjacency[Y_idx, Z2_idx] = 1
        adjacency[Y_idx, Z6_idx] = 1
        
        adjacency[Z1_idx, X_idx] = 1
        adjacency[Z1_idx, Y_idx] = 1
        
        adjacency[Z3_idx, Z3_2_idx] = 1
        adjacency[Z3_2_idx, Y_idx] = 1
        
        adjacency[Z4_idx, Y_idx] = 1
        adjacency[Z4_idx, C5_idx] = 1
        
        adjacency[Z5_idx, X_idx] = 1
        adjacency[Z5_idx, C4_idx] = 1
        
        adjacency[M1_idx, X_idx] = 1
        adjacency[M1_idx, M3_idx] = 1
        adjacency[M2_idx, Y_idx] = 1
        adjacency[M2_idx, M3_idx] = 1
        
        adjacency[B1_idx, X_idx] = 1
        adjacency[B1_idx, B3_idx] = 1
        adjacency[B2_idx, Y_idx] = 1
        adjacency[B2_idx, B3_idx] = 1
        adjacency[B3_idx, X_idx] = 1
        adjacency[B3_idx, Y_idx] = 1
        
        adjacency[C1_idx, C3_idx] = 1
        adjacency[C1_idx, C4_idx] = 1
        adjacency[C1_idx, C6_idx] = 1
        adjacency[C2_idx, C3_idx] = 1
        adjacency[C2_idx, C5_idx] = 1
        adjacency[C2_idx, C6_idx] = 1
        adjacency[C3_idx, C4_idx] = 1
        adjacency[C3_idx, C5_idx] = 1
        adjacency[C4_idx, X_idx] = 1
        adjacency[C5_idx, Y_idx] = 1

        df_adj = pd.DataFrame(adjacency, 
                              columns = df_vars.columns,
                              index = df_vars.columns)

        if verbose:
            print("VARIABLES:")
            display(df_vars.head())
            print("ADJACENCY:")
            display(df_adj)

        return df_vars, df_adj
    
    
    def generate_complex(self,
                         n: int = 5000, 
                         noise_distribution: str = "bernoulli",
                         x_causes_y: bool = True,
                         function: str = "linear",
                         coefficient_range: tuple = (1, 2),
                         verbose: bool = False):

        '''
        This function generates a complex DAG as a synthetic benchmark.
        All variables are discrete random variables with finite support. 
        
        Parameters to noise distributions and SEM coefficients are all fixed for
        reproducibility purposes.

        Parameters:
        -----------
        n: int = 5000
            Total observations to sample.
        noise_distribution: str = "laplace"
            Distributional form of the noise (i.e. noise) term of the structural equation
            Y = f(X) + noise.
        x_causes_y: bool = True
            Boolean flag indicating whether X is a cause of Y. If set to false, there is
            no causal relation between X and Y. Note that Z is always a cause of both X 
            and Y by definition.
        verbose: bool = False
            Flag for displaying dataframes.

        Return:
        -------
        df_vars
            df_vars contains variable data.
        df_adj
            directed adjacency matrix as dataframe.
        '''

        # Construct noise terms of structural equation.
        total_vars = 16
        noise = []
        if noise_distribution == "bernoulli":
            for var in range(total_vars):
                noise.append(np.random.binomial(n = 1, p = 0.6, size = n).reshape(-1, 1))
        elif noise_distribution == "hypergeometric":
            for var in range(total_vars):
                noise.append(np.random.hypergeometric(ngood = 1, 
                                                      nbad = 1, 
                                                      nsample = 1, 
                                                      size = n).reshape(-1, 1))
        elif noise_distribution == "mixed":  
            for i in range(total_vars):
                if i % 2 == 0:
                    # Hypergeometric.
                    noise.append(np.random.hypergeometric(ngood = 1, 
                                                          nbad = 1, 
                                                          nsample = 1, 
                                                          size = n).reshape(-1, 1))   
                else:
                    # Bernoulli.
                    noise.append(np.random.binomial(n = 1, p = 0.6, size = n).reshape(-1, 1))
        else:
            raise NameError("noise_distribution must take a value in ['bernoulli', 'hypergeometric', 'mixed']")

        # Define causal mechanisms.
        if function == "linear":
            fun = lambda x: np.around(1.9 * x) # Originally used 0.3.
        elif function == "quadratic":
            fun = lambda x: np.around(0.4 * x**2) # 0.4
        elif function == "square root":
            fun = lambda x: np.around(2.0 * np.sqrt(abs(x))) # low = 0.95, high = 1.1
        elif function == "cube root":
            fun = lambda x: np.around(1.2 * np.cbrt(abs(x))) # low = 0.95, high = 1.1
        else:
            raise NameError("Valid `function` values are in ['linear', 'square root', 'cube root', 'quadratic'].")
        
        # Define coefficient generator.
        coeff = lambda : np.random.randint(low = coefficient_range[0], high = coefficient_range[1], size = 1)
        
        # Define variables.
        if x_causes_y:
            xy_coeff = 1
        else:
            xy_coeff = 0  
        X_idx = 0
        Y_idx = 1
        Z1_idx = 2
        Z2_idx = 3
        Z3_idx = 4
        Z4_idx = 5
        Z5_idx = 6
        Z6_idx = 7
        Z7_idx = 8
        Z8_idx = 9
        C1_idx = 10
        C2_idx = 11
        C3_idx = 12
        C4_idx = 13
        C5_idx = 14
        C6_idx = 15  
        
        # Pretreatment variables on simple paths and isolated vars.
        Z1 = noise[Z1_idx]
        Z4 = noise[Z4_idx]
        Z5 = noise[Z5_idx]
        Z8 = noise[Z8_idx]
        
        # Complex backdoor path.
        C1 = noise[C1_idx]
        C2 = noise[C2_idx]
        C3 = coeff()*fun(C1 + C2) + noise[C3_idx]
        C4 = coeff()*fun(C1 + C3 + Z5) + noise[C4_idx]
        C5 = coeff()*fun(C2 + C3 + Z4) + noise[C5_idx]
        C6 = coeff()*fun(C1 + C2) + noise[C6_idx]  
        
        # X, Y, and post-treatment variables.
        X  = (coeff()*fun(Z1 + Z5 + C4) + noise[X_idx])
        Z3 = coeff()*fun(X) + noise[Z3_idx]
        Y  = (coeff()*fun(xy_coeff*X + Z1 + Z3 + Z4 + C5) + noise[Y_idx])
        Z2 = coeff()*fun(X + Y) + noise[Z2_idx]
        Z6 = coeff()*fun(Y) + noise[Z6_idx]
        Z7 = coeff()*fun(X) + noise[Z7_idx]
        
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z2": Z2.reshape(-1), 
                                "Z3": Z3.reshape(-1), 
                                "Z4": Z4.reshape(-1), 
                                "Z5": Z5.reshape(-1),
                                "Z6": Z6.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1),
                                "Z1 (C1)": C1.reshape(-1),
                                "Z1 (C2)": C2.reshape(-1),
                                "Z1 (C3)": C3.reshape(-1),
                                "Z1 (C4)": C4.reshape(-1),
                                "Z1 (C5)": C5.reshape(-1),
                                "Z2 (C6)": C6.reshape(-1)
                                }).astype(int)
 
        # Construct adjacency matrix.
        # [i,j] means i causes j.
        '''
        X causes: Y, Z2, Z3, Z7
        Y causes: Z2, Z6
        Z1 causes: X, Y
        Z3 causes: Y
        Z4 causes: Y, C5
        Z5 causes: X, C4
        C1 causes: C3, C4, C6
        C2 causes: C3, C5, C6
        C3 causes: C4, C5
        C4 causes: X
        C5 causes: Y
        '''
        c = xy_coeff
        adjacency = np.zeros((total_vars, total_vars), dtype = int)
        
        adjacency[X_idx, Y_idx] = 1
        adjacency[X_idx, Z2_idx] = 1
        adjacency[X_idx, Z3_idx] = 1
        adjacency[X_idx, Z7_idx] = 1
        
        adjacency[Y_idx, Z2_idx] = 1
        adjacency[Y_idx, Z6_idx] = 1
        
        adjacency[Z1_idx, X_idx] = 1
        adjacency[Z1_idx, Y_idx] = 1
        
        adjacency[Z3_idx, Y_idx] = 1
        
        adjacency[Z4_idx, Y_idx] = 1
        adjacency[Z4_idx, C5_idx] = 1
        
        adjacency[Z5_idx, X_idx] = 1
        adjacency[Z5_idx, C4_idx] = 1
        
        adjacency[C1_idx, C3_idx] = 1
        adjacency[C1_idx, C4_idx] = 1
        adjacency[C1_idx, C6_idx] = 1
        adjacency[C2_idx, C3_idx] = 1
        adjacency[C2_idx, C5_idx] = 1
        adjacency[C2_idx, C6_idx] = 1
        adjacency[C3_idx, C4_idx] = 1
        adjacency[C3_idx, C5_idx] = 1
        adjacency[C4_idx, X_idx] = 1
        adjacency[C5_idx, Y_idx] = 1

        df_adj = pd.DataFrame(adjacency, 
                              columns = df_vars.columns,
                              index = df_vars.columns)

        if verbose:
            print("VARIABLES:")
            display(df_vars.head())
            print("ADJACENCY:")
            display(df_adj)

        return df_vars, df_adj
    
    
    def generate_linear_gaussian(self,
                                 n: int = 1000, 
                                 x_causes_y: bool = True,
                                 xy_coeff: float = 1.0,
                                 m_structure: bool = False,
                                 butterfly_structure: bool = False,
                                 coefficient_range: tuple = (1.25, 2.5),
                                 verbose: bool = False):

        '''
        '''

        # Construct noise terms of structural equation.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        total_vars = 10
        if m_structure:
            total_vars += 3
        if butterfly_structure:
            total_vars += 3
        noise = []
        for var in range(total_vars):
            noise.append(np.random.normal(loc = 0.0, scale = 1.0, size = n).reshape(-1, 1))

        # Define coefficient generator.
        coeff = lambda : 1
        
        # Define variables.
        if not x_causes_y:
            xy_coeff = 0
        Z1 = noise[2]
        Z4 = noise[5]
        Z5 = noise[6]
        Z8 = noise[9]
        X = coeff()*Z1 + coeff()*Z5 + noise[0]
        if m_structure:
            M1 = noise[10]
            M2 = noise[11]
            M3 = coeff()*M1 + coeff()*M2 + noise[12]
            X = X + coeff()*M1
        if butterfly_structure:
            if m_structure:
                B1 = noise[13]
                B2 = noise[14]
                B3 = coeff()*B1 + coeff()*B2 + noise[15]
            else:
                B1 = noise[10]
                B2 = noise[11]
                B3 = coeff()*B1 + coeff()*B2 + noise[12]
            X = X + coeff()*B1 + coeff()*B3
        Z3 = coeff()*X + noise[4]
        Y = xy_coeff*X + coeff()*Z1 + coeff()*Z3 + coeff()*Z4 + noise[1]
        if m_structure:
            Y = Y + coeff()*M2
        if butterfly_structure:
            Y = Y + coeff()*B2 + coeff()*B3
        Z2 = coeff()*X + coeff()*Y + noise[3]
        Z6 = coeff()*Y + noise[7]
        Z7 = coeff()*X + noise[8]
            
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z2": Z2.reshape(-1), 
                                "Z3": Z3.reshape(-1), 
                                "Z4": Z4.reshape(-1), 
                                "Z5": Z5.reshape(-1),
                                "Z6": Z6.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1)})
        df_noise = pd.DataFrame({"X": noise[0].reshape(-1), 
                                 "Y": noise[1].reshape(-1), 
                                 "Z1": noise[2].reshape(-1),
                                 "Z2": noise[3].reshape(-1), 
                                 "Z3": noise[4].reshape(-1), 
                                 "Z4": noise[5].reshape(-1), 
                                 "Z5": noise[6].reshape(-1),
                                 "Z6": noise[7].reshape(-1), 
                                 "Z7": noise[8].reshape(-1), 
                                 "Z8": noise[9].reshape(-1)})
        var_names = ["Z" + str(i) for i in range(1, 9)]
        var_names = ["X", "Y"] + var_names
        
        if m_structure:
            df_vars["M1"]  = M1.reshape(-1)
            df_vars["M2"]  = M2.reshape(-1)
            df_vars["M3"]  = M3.reshape(-1)
            df_noise["M1"] = M1.reshape(-1)
            df_noise["M2"] = M2.reshape(-1)
            df_noise["M3"] = M3.reshape(-1)

        if butterfly_structure:
            df_vars["B1"]  = B1.reshape(-1)
            df_vars["B2"]  = B2.reshape(-1)
            df_vars["B3"]  = B3.reshape(-1)
            df_noise["B1"] = B1.reshape(-1)
            df_noise["B2"] = B2.reshape(-1)
            df_noise["B3"] = B3.reshape(-1)
        
        if verbose:
            print("VARIABLES:")
            display(df_vars.head())
            print("NOISE:")
            display(df_noise.head())

        return df_vars, xy_coeff
    
    
    def scale(self, var):
        scaler = StandardScaler()
        return scaler.fit_transform(var.reshape(-1, 1))
    