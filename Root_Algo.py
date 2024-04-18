#Packages
import time
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from PyRKHSstats import hsic
from conditional_independence import hsic_test

#This file implements a nonparametric method to identify root nodes of a linear non-gaussian noise 
#model using KRR to identify features that always have independent residuals from pairwise regression
#as roots.

def marg_dep(data):
    '''Inputs: 
    data - table that has d columns for each variable and n rows of samples (np matrix)
    Returns: a list of lists, where list at index i contains j if variable i and variable j from
    the data are marginally dependent.
    '''
    #Find Data Dimensionality
    d = data.shape[1]
    #Initalize 
    ind_collection = [[] for _ in range(d)]
    #Marginal Independence Testing
    for i in range(d):
        for j in range(i+1, d):
            #Append i and j if they are dependent
            if (hsic_test(data, i,j,[])['p_value'] < 0.01):
                ind_collection[i].append(j)
                ind_collection[j].append(i)
    return ind_collection


#Root Estimation
def root_detection(data, alpha = 0.005):
    '''Intakes data and a threshold alpha, 
    and outputs a list of indices that corresponds to root nodes (roots),
    as well as a list of lists containing the marginal dependencies.'''
    #Find number of features
    d = data.shape[1]
    #Check Marginal Dependencies
    marg_depend = marg_dep(data)
    #Initialize storage of p-values.
    ind_collection = [[] for _ in range(d)]
    #Perform pairwise nonparametric regression
    for i in range(d):
        for j in [x for x in range(d) if x!=i]:
            #Regress x_j on x_i, if they are dependent   
            if j in marg_depend[i]:
                #Split Data into Test/Train
                X_train, X_test, y_train, y_test = train_test_split(data[:,i].reshape(-1,1), data[:,j].reshape(-1,1), test_size=0.20)
                #Train and Test a Kernel Ridge Regression model
                #Gaussian Kernel
                #krr = KernelRidge(kernel='rbf',alpha = 0.01, gamma = 0.01)
                #Polynomial Kernel
                #Train
                krr = KernelRidge(kernel='polynomial',alpha = .01, degree = 1, coef0= 1)
                krr.fit(X_train,y_train)
                #Test
                y_pred = krr.predict(X_test)
                residuals = y_test - y_pred
                #Check for independence of residuals with x_i
                resid_data = np.hstack((X_test,residuals.reshape(-1,1)))
                #Using HSIC - Append p-value (0 IFF x_i is not independent of residual)
                ind_collection[i].append(hsic_test(resid_data, 0,1,[])['p_value'])
            #If x_j and x_i are independent, don't regress
            else:
                ind_collection[i].append(1)
    print(ind_collection)
    #Return indices of nodes that always have independent residuals. 
    #Node x_i has all independent residuals if ind_collection[i] > alpha for all the p-values.
    #Roots = [] if no roots found (assumptions violated or finite sample issue)
    roots = [index for index, sublist in enumerate(ind_collection) if all(value > alpha for value in sublist)]
    return roots, marg_depend

def standardize_vectors(*vectors):
    '''standardizes input vectors'''
    standardized_vectors = []
    for vector in vectors:
        mean = np.mean(vector)
        std = np.std(vector)
        standardized_vector = (vector - mean) / std
        standardized_vectors.append(standardized_vector)
    return standardized_vectors

def DAG(n=1000):
     '''produces standardized data from custom DAGs'''
     #DAG Structure
     x = np.random.gamma(1,2,n)
     y = np.random.gamma(1,2,n)
     z = x+y+np.random.gamma(1,2,n)
     data = np.column_stack(standardize_vectors(x,y,z))
     return data

#Synthetic Empirical Evaluation
start_time = time.time()
data = DAG(n=1000)
roots, marg_ind = root_detection(data)
print(roots)
print(marg_ind)
print(f"Runtime: {time.time() - start_time:.3f} s")