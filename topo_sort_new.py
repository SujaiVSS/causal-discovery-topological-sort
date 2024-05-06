#Packages
import time
import numpy as np
from sklearn.model_selection import train_test_split
from conditional_independence import hsic_test
from sklearn.linear_model import LinearRegression
import itertools
from collections import deque, defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

# This file implements a new topo sort algo for lingam, that 
# returns the topological sort. Works by finding roots, 
# then ancestor set of all other nodes.

def marg_dep(data, alpha = 0.01):
    '''
    Determine marginal dependencies between variables.

    Parameters:
    data (np.array): Dataset with d variables as columns and n samples as rows.
    alpha (float): Threshold for determining dependence, default is 0.01.

    Returns:
    list of lists: Each sublist contains indices of variables that are marginally dependent with the variable at the index of the sublist.
    '''
    # Number of variables
    d = data.shape[1]
    # Initialize list of lists for dependencies
    ind_collection = [[] for _ in range(d)]
    # prune original data to only use k samples (no need to use all data in pairwise)
    k = 1000
    random_indices = np.random.choice(data.shape[0], size = k, replace = False)
    sampled_data = data[random_indices,:]
    # Test for dependencies between all pairs (i, j) where neither are sorted
    for i in range(d):
        for j in range(i+1, d):
            #Append if x_i and x_j are dependent
            if (hsic_test(sampled_data, i,j,[])['p_value'] < alpha):
                ind_collection[i].append(j)
                ind_collection[j].append(i)
    return ind_collection

def marg_pop(marg_dep, ancestors):
    # Number of variables
    d = data.shape[1]
    #Populate ancestor lists using marg_dep information
    for i in range(d):
        for j in [x for x in range(d) if x!=i]:
            if j not in marg_dep[i]:
                ancestors[i][j] = 1
    return ancestors


def get_mutual_ancestors(i, j, ancestors):
    """ Returns indices of mutual ancestors of x_i and x_j """
    d = len(ancestors)
    return [k for k in range(d) if ancestors[i][k] == 3 and ancestors[j][k] == 3 and k != i and k != j]


def find_tightest_ancestor_cluster(data, mutual_ancestor_indices, o):
    """
    Find indices of the o closest data points forming the tightest cluster based on mutual ancestors,
    using a KDTree for efficient nearest neighbor searches.

    Parameters:
        data (numpy.ndarray): The dataset, where each row is a data point.
        mutual_ancestor_indices (list): Indices of features to consider for clustering.
        o (int): The number of points in the desired cluster.

    Returns:
        list: Indices of the o data points forming the tightest cluster.
    """
    if data.shape[0] < o:
        raise ValueError("Not enough data points to form a cluster of size o")

    # Extract only the relevant features for each data point
    filtered_data = data[:, mutual_ancestor_indices]

    # Initialize KDTree to find the o nearest points
    tree = KDTree(filtered_data)

    # Find the o nearest neighbors for each point in the filtered dataset
    distances, indices = tree.query(filtered_data, k=o)

    # Calculate the sum of distances in each neighborhood to find the tightest cluster
    distance_sums = np.sum(distances, axis=1)
    tightest_center_index = np.argmin(distance_sums)

    # Indices of the tightest cluster in the dataset
    tightest_cluster_indices = indices[tightest_center_index]
    return tightest_cluster_indices


def pair_reg(data, ancestors, alpha = 0.01, tolerance = 0.5,condition = 300):
    '''Subroutine that updates the ancestor table after each round of conditional regressions.'''
    # -1 for i's relation to i
    # 0 means i,j unknown relations, 
    # 1 means i,j have no ancestral relation, 
    # 2 means i < j,  
    # 3 means j < i,

    # Number of variables
    d = data.shape[1]
    # Initialize list of lists for p-values
    plist = [[0 for j in range(d)] for i in range(d)]
    # Initialize indices for knn of mutual ancestors
    mutual_ancestors = {}

    #Populate p-list using pairwise regression + adding mutual ancestors as covariates
    for i in range(d):
        for j in [x for x in range(d) if x!=i and ancestors[i][x] == 0]:
    
            #Grab mutual ancestors
            mutual_ancestor_indices = get_mutual_ancestors(i,j,ancestors)

            if mutual_ancestor_indices:
                print(mutual_ancestor_indices, i,j)
                #mutual_ancestor_indices = get_mutual_ancestors(i, j, ancestors)
                tightest_ancestor_indices = find_tightest_ancestor_cluster(data, mutual_ancestor_indices, condition)
                

                # Select data for x_i and x_j using the indices of the tightest cluster
                filtered_data = data[tightest_ancestor_indices, :]
                x_i = filtered_data[:, i].reshape(-1, 1)
                x_j = filtered_data[:, j].reshape(-1, 1)

                # Perform regression on this filtered subset
                reg = LinearRegression().fit(x_i, x_j)
                x_j_pred = reg.predict(x_i)
                residuals_ij = x_j - x_j_pred
                resid_data = np.hstack((x_i, residuals_ij))
                plist[i][j] = hsic_test(resid_data, 0,1,[])['p_value']
            else:
                # If no mutual ancestors, use original data
                # prune original data to only use k samples (no need to use all data for pairwise tests)
                k = 1000
                random_indices = np.random.choice(data.shape[0], size = k, replace = False)
                sampled_data = data[random_indices,:]
                #Grab x_i, x_j
                x_i = sampled_data[:,i].reshape(-1,1)
                x_j = sampled_data[:,j].reshape(-1,1)
                reg = LinearRegression().fit(x_i, x_j)
                x_j_pred = reg.predict(x_i)
                residuals_ij = x_j - x_j_pred
                resid_data = np.hstack((x_i, residuals_ij.reshape(-1, 1)))
                plist[i][j] = hsic_test(resid_data, 0,1,[])['p_value']

    #Populate ancestor list using p-list
    for i in range(d):
        for j in [x for x in range(d) if x!=i and ancestors[i][x] == 0]:
            #x_i and x_j share no ancestral relation
            if plist[i][j] >= alpha and plist[j][i] >= alpha:
                ancestors[i][j] = 1
                ancestors[j][i] = 1
            #x_i is ancestor of x_j
            if plist[i][j] >= alpha and plist[j][i] < alpha:
                ancestors[i][j] = 2
                ancestors[j][i] = 3
            #x_j is ancestor of x_i
            if plist[i][j] < alpha and plist[j][i] >= alpha:
                ancestors[i][j] = 3
                ancestors[j][i] = 2
    return ancestors



def ancestor_table(data):
    '''
    Returns a completed ancestor table, given a dataset.
    '''
    d = data.shape[1]
    ancestors = [[-1 if i == j else 0 for j in range(d)] for i in range(d)]

    # Obtain marginal independence table
    marg = marg_dep(data)

    # Obtain initial ancestor set
    ancestors = marg_pop(marg,ancestors)

    i = 0
    while any(0 in sublist for sublist in ancestors):
        ancestors = pair_reg(data,ancestors,alpha = 0.01, condition = 300)
        i+= 1
        if i == 100:
            print("Loop")
            return []
    
    return ancestors


def ancestor_sort(ancestors):
    '''
    Returns a topological ordering, given a completed ancestor table.
    '''
    # Initialize
    d = len(ancestors)
    # If the ancestor table failed to complete
    if d == 0:
        return "Error"
    order = []
    remaining = set(range(d))  # Set of indices of nodes still in the graph

    def has_no_descendants(node):
        """ Check if the node has no descendants in the remaining graph """
        return all(ancestors[node][j] != 2 for j in remaining)

    while remaining:
        # Find a node that has no descendants
        sink = next(node for node in remaining if has_no_descendants(node))
        # Add it to the order
        order.append(sink)
        # Remove it from the set of remaining nodes
        remaining.remove(sink)
    order.reverse()

    return order


def linear_topo_sort(data):
    '''Returns a linear topological ordering, given data.'''
    # Find ancestor table
    ancestors = ancestor_table(data)
    # Find linear topological ordering
    sort =  ancestor_sort(ancestors)
    return sort



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
     x = np.random.uniform(-2,2,n)
     y = np.random.uniform(-2,2,n)
     z = x + y + np.random.uniform(-2,2,n)
     w = z+np.random.uniform(-2,2,n)
     m = w+ np.random.uniform(-2,2,n)
     data = np.column_stack(standardize_vectors(x,y,z,w,m))
     #data = np.column_stack([x,y,z,w,m])
     return data

#Synthetic Empirical Evaluation
start_time = time.time()
data = DAG(n=3000)

#Testing
'''
# Initialize list of lists for ancestors
d = data.shape[1]
ancestors = [[-1 if i == j else 0 for j in range(d)] for i in range(d)]

# Obtain marginal independence table
marg = marg_dep(data)

#Obtain 
ancestors = marg_pop(marg,ancestors)

i = 0
while any(0 in sublist for sublist in ancestors) and i < 20:
    ancestors = pair_reg(data,ancestors,alpha = 0.01, condition = 300)
    i+= 1
    if i == 20:
        print("loop")
''' 


sort = linear_topo_sort(data)
print(sort)

#Time count
print(f"Runtime: {time.time() - start_time:.3f} s")