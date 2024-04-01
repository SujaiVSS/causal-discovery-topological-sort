'''This file implements a path tracing algo: given data generated from a Additive Noise Model and a correct topological sort, 
the path_tracing function returns the correct directed edge set (list of node parent set).'''

#Packages
# Suppress all deprecation warnings
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from causallearn.utils.cit import CIT

def edge_detection(a,b,parents,children,data, ind_collection):
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
    mediators = parents[b] 
    cond_set = confounders.union(mediators)
    #Include only marginally dependent (on b) potential mediators and confounders
    cond_set = cond_set.intersection(ind_collection[b])
    fisherz_obj = CIT(data, "kci")
    pvalue = fisherz_obj(a,b,list(cond_set))
    if pvalue < 0.05:
        parents[b].add(a)
        children[a].add(b)
    return parents, children

def path_tracing(topo_sort, data):
    '''
    Inputs:
    topo_sort - a reversed topological sort, where index(x)<index(y) implies y cannot cause x
    data - table that has columns for each variable and rows of samples (np matrix)
    '''
    #Initialization
    d = len(topo_sort)
    parents = [set() for _ in range(d)]
    children = [set() for _ in range(d)]
    ind_collection = marg_ind(data)
    #CIT Algorithm
    for i in range(1,d,+1):
        b = topo_sort[i]
        for j in range(i-1,-1,-1):
            a = topo_sort[j]
            #only check for edge if a and b are marginally dependent
            if b in ind_collection[a]:
                parents, children = edge_detection(a,b,parents,children,data,ind_collection)
    return parents, children

def marg_ind(data):
    '''Inputs: 
    data - table that has d columns for each variable and n rows of samples (np matrix)
    Returns:
    dxd symmetric matrix where matrix[i,j] = matrix[j,i] = 1 if x_i ind x_j, 0 otherwise.
    '''
    #Find Data Dimensionality
    d = data.shape[1]
    #Create matrix: there is a 0 in i,j (and j,i) if x_i \ind x_j, 1 otherwise.
    ind_collection = [set() for _ in range(d)]
    #Initialize Kernel Independence Test
    fisherz_obj = CIT(data, "kci")
    for i in range(d):
        for j in range(i+1, d):
            if fisherz_obj(i,j,[]) < 0.01:
                ind_collection[i].add(j)
                ind_collection[j].add(i)
    return ind_collection

#Testing
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
     y = 0.5*x+ np.random.gamma(1,2,n)
     z = 0.3*y+0.3*x+np.random.gamma(1,2,n)
     w = 0.3*y+0.3*x+np.random.gamma(1,2,n)
     data = np.column_stack(standardize_vectors(x,y,z,w))
     return data

#Testing
start_time = time.time()
#print(marg_ind(data))
sort = [0,1,2,3]
count = 0
iter = 30
for i in range(iter):
    data = DAG()
    parents, descendants = path_tracing(sort, data)
    if parents == [set([]),set([0]),set([0,1]),set([0,1])]:
        count +=1

print("Accuracy:", count/iter)
print(f"Runtime: {time.time() - start_time:.3f} s")