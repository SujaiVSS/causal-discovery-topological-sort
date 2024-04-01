#Packages
# Suppress all deprecation warnings
import warnings
import time
warnings.filterwarnings("ignore")
import numpy as np
from causallearn.utils.cit import CIT
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split

#This file implements a nonparametric method to identify root nodes of a ADM,
#using KRR to identify features that always have independent residuals.

#Root Estimation
def root_detection(data):
    '''Intakes a 2D numpy matrix and 
    outputs a list of indices that corresponds to root nodes.'''
    #Find number of features
    d = data.shape[1]
    #Initialize storage of p-values.
    ind_collection = [[] for _ in range(d)]
    #Perform pairwise nonparametric regression
    for i in range(d):
        for j in [x for x in range(d) if x!=i]:   
            #Regress x_j on x_i
            #Split Data into Test/Train
            X_train, X_test, y_train, y_test = train_test_split(data[:,i].reshape(-1,1), data[:,j].reshape(-1,1), test_size=0.1, random_state=0)
            # Train a Kernel Ridge Regression model
            krr = KernelRidge(kernel='rbf',alpha = 0.01, gamma = 0.01)
            krr.fit(X_train,y_train)
            y_pred = krr.predict(X_test)
            residuals = y_test - y_pred
            resid_data = np.hstack((X_test,residuals.reshape(-1,1)))
            #Check for independence of residuals with x_i
            fisherz_obj = CIT(resid_data, "kci")
            #Append p-value (0 IFF x_i is not independent of residual)
            ind_collection[i].append(round(fisherz_obj(0,1,[]),4))
    #Returns indices of nodes Returns [] if no roots found (additive noise assumption violated)
    roots = [index for index, sublist in enumerate(ind_collection) if all(value >= 0.01 for value in sublist)]
    return roots
#Testing
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
     z = 0.3*y**2+0.3*x+np.random.gamma(1,2,n)
     w = 0.3*y+0.3*x**2+np.random.gamma(1,2,n)
     data = np.column_stack(standardize_vectors(x,y,z,w))
     return data

#Synthetic Empirical Evaluation
count = 0
iter = 30
start_time = time.time()
for i in range(iter):
    data = DAG(n=1000)
    roots = root_detection(data)
    if roots == [0,1]:
        count+=1
print("Accuracy:", round(count/iter,2))
print(f"Runtime: {time.time() - start_time:.3f} s")