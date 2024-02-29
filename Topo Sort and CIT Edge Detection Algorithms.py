#This file defines all functions used to by the Topological Sort and CIT Edge Detection Algorithms
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial import KDTree

#Defining functions for Topo-Sort
def score_estim(k, value, data, b):
    '''
    Inputs: 
    k - the column index of the variable whose score function we are estimating (int)
    value  - the data point the score function is being calculated at (np vector)
    data - table that has columns for each variable and rows of samples (np matrix)
    b - bandwidth parameter for the kernel function (float)
    varepsilon - parameter between (0,1) that is equivalent to taking l-nearest neighbors, where
    varepsilon = l/log n (float)
    Returns:
    conditional score of variable k evaluated at value
    '''
    #Initialization
    score_numerator = 0
    score_denominator = 0
    #Implementing Kernel Density Estimator for Score Function
    kernels = rbf_kernel(np.array([value]), data, 1/b)[0]
    score_denominator = np.sum(kernels)
    differences = (value[k] - data[:,k])
    score_numerator = (1/b)*(np.sum(differences*kernels))
    return score_numerator/score_denominator

def cond_score_estim(k, value, data, b, l):
    '''
    Inputs: 
    k - the column index of the variable whose score function we are estimating (int)
    value  - the data point the score function is being calculated at (np vector)
    data - table that has columns for each variable and rows of samples (np matrix)
    b - bandwidth parameter for the kernel function (float)
    l - the number of nearest neighbors used for conditioning
    Returns:
    conditional score of variable k evaluated at value
    '''
    #Initialization
    score_numerator = 0
    score_denominator = 0
    #Setting x_{-k} 
    value_k = np.delete(value,k)
    #Removing x_k from data
    data_k = np.delete(data,k,axis=1)
    #Finding the LNN of x_{-k}
    tree = KDTree(data_k)
    distances,indices =tree.query(value_k, k = l)
    data_lnn = data[indices, :]
    #Implementing Kernel Density Estimator for Score Function
    kernels = rbf_kernel(np.array([value]), data_lnn, 1/b)[0]
    score_denominator = np.sum(kernels)
    differences = value[k] - data_lnn[:,k]
    score_numerator = (1/b)*(np.sum(differences*kernels))
    return score_numerator/score_denominator

def leaf_hypothesis_test(k, data, b, l):
    '''
    Inputs: 
    k - the column index of the variable whose score function we are estimating (int)
    data - table that has columns for each variable and rows of samples (np matrix)
    b - bandwidth parameter for the kernel function (float)
    l - the number of nearest neighbors used for conditioning
    
    Returns:
    whether a test statistic that is largest when x_k is a leaf.
    '''
    #Initialization
    rows = data.shape[0]
    estimates = []
    #Estimating Score^2, Cond Score^2
    for i in range(rows):
        score = score_estim(k, data[i], data, b)
        cond_score = cond_score_estim(k, data[i], data, b, l)
        estimates.append(abs(score-cond_score))

    #Cond Fisher Info to Fisher Info Ratio
    return np.mean(estimates)

def leaf_detection(data, b, l):
    '''
    Inputs: 
    data - table that has columns for each variable and rows of samples (np matrix)
    b - bandwidth parameter for the kernel function (float)
    l - the number of nearest neighbors used for conditioning
    
    Returns:
    the index of the node with the highest test statistic (most likely leaf)
    '''
    s = []
    for k in range(data.shape[1]):
        s.append(leaf_hypothesis_test(k, data, b, l = l))
    return s.index(max(s))

def topo_sort(data, b, l):
    '''
    Inputs: 
    data - table that has columns for each variable and rows of samples (np matrix)
    b - bandwidth parameter for the kernel function (float)
    l - the number of nearest neighbors used for conditioning
    
    Returns:
    a reversed topological sort, where index(x)<index(y) implies y cannot causes x (root is at beginning)
    '''
    filter = list(range(data.shape[1]))
    sort = []
    for _ in range(data.shape[1]-1):
        leaf = leaf_detection(data[:,filter], b, l = l)
        sort.append(filter[leaf])
        filter.pop(leaf)
    sort.append(filter[0])
    sort.reverse()
    return sort

#Defining Functions for CIT Edge Detection

def edge_detection(a,b,parents,children,data):
    '''
    Inputs:
    a - the possible parent
    b - the possible child
    parents - a list of lists, where the list at index i contains all parents of node i collected so far
    child - a list of lists, where the list at index i contains all children of node i collected so far
    
    Returns:
    the parent set and children set at this step of the edge detection
    '''
    #Potential Confounder Set
    confounders = parents[a]
    #Potential Mediator Set
    mediators = parents[b].union(children[a])
    fisherz_obj = CIT(data, "kci")
    cond_set = confounders.union(mediators)
    pvalue = fisherz_obj(a,b,list(cond_set))
    if pvalue < 0.05:
        parents[b].add(a)
        children[a].add(b)
    return parents, children

def path_tracing(topo_sort, data):
    '''
    Inputs:
    topo_sort - a reversed topological sort, where index(x)<index(y) implies y cannot causes x
    data - table that has columns for each variable and rows of samples (np matrix)
    '''
    #Initialization
    d = len(topo_sort)
    parents = [set() for _ in range(d)]
    children = [set() for _ in range(d)]
    #CIT Algorithm
    for i in range(1,d,+1):
        b = topo_sort[i]
        for j in range(i-1,-1,-1):
            a = topo_sort[j]
            parents, children = edge_detection(a,b,parents,children,data)
    return parents, children

def adj_matrix(parent_sets):
    '''
    takes in a parent set (list of lists) and returns an adjacency matrix (i causes j if 1 at (i,j))    
    '''
    # Determine the number of nodes
    num_nodes = len(parent_sets)
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    # Populate the adjacency matrix based on the parent sets
    for child_index, parents in enumerate(parent_sets):
        for parent_index in parents:
            # Set the entry to 1 to indicate an edge from the parent to the child
            adjacency_matrix[parent_index][child_index] = 1
    return adjacency_matrix

def Topo_CIT_Causal_Discovery(data):
    '''
    Inputs:
    data - table that has columns for each variable and rows of samples (np matrix)
    Returns:
    adjacency matrix - a DAG that represents the data in the forma of a dxd np matrix.
    '''
    #Generic Bandwidth 
    b = (data.shape[0])**(-1/(data.shape[1]*5))
    #Generic Neighbors
    neighbors = data.shape[1]-1
    #Obtain Reversed Topological Sort
    sort = topo_sort(data,b, neighbors)
    print(sort)
    #Obtain parent set
    parents, children = path_tracing(sort, data)
    #Return DAG matrix
    return np.array(adj_matrix(parents))

#Testing
def test(n=1000,a=-4,b=5,c=-3):
     x = np.random.normal(1,0.5,n)
     y = a*x**2 + np.random.normal(1,0.5,n)
     z = b*x**2 + 3+y**2 +  np.random.normal(1,0.5,n)
     w = np.random.normal(1,0.5,n)
     m = 3*w**2 + np.random.normal(1,0.5,n)
     return x,y,z,w,m

x,y,z,w,m = test(n=1500)
table = np.column_stack([x,y,z,w,m])
print(Topo_CIT_Causal_Discovery(table))