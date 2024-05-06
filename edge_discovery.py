'''This file implements a path tracing algo: given data generated from a Additive Noise Model and a correct topological sort, 
the path_tracing function returns the correct directed edge set (list of node parent set).'''

# Suppress all deprecation warnings
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from causallearn.utils.cit import CIT
import networkx as nx

class EdgeDiscovery():

    def __init__(self,
                 adjacency_matrix = None):

        self.adj = adjacency_matrix
    

    def oracle(self,
               var_0,
               var_1,
               conditioning_set = None) -> float:
    
        '''
        Oracle independence test given ground truth DAG.
        This method only works for variants of the 10-node DAG, where each covariate
        is directly adjacent to {X,Y} and has no paths to other covariates.
    
        This test is used for the experiments reported in Figure 2 and Table B.1.

        var_0 and var_1 are integers corresponding to the numpy column index, and conditioning_set
        is a list of integers.
        '''
    
        if self.adj is None:
            raise ValueError("self.adj is None; must supply ground truth adjacnecy matrix as numpy array to use oracle.")

        if conditioning_set is None:
            conditioning_set = set()
        
        # Get p-value using ground truth DAG.
        graph = nx.from_numpy_array(self.adj, create_using = nx.DiGraph)
        p_val = 1 if nx.d_separated(graph, {var_0}, {var_1}, conditioning_set) else 0
        
        return p_val

    
    def edge_detection(self,
                       a, 
                       b, 
                       parents,
                       children,
                       data, 
                       ind_collection,
                       cit_test = "fisherz", 
                       alpha = 0.001):
        
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
        
        # object views vs copy constructing new objects
        cond_set = confounders.union(mediators)
        
        #Include only marginally dependent (on b) potential mediators and confounders
        cond_set = cond_set.intersection(ind_collection[b])
        if cit_test != "oracle":
            cit_test = CIT(data, cit_test)
            pvalue = cit_test(a,b,list(cond_set))
        else:
            pvalue = self.oracle(var_0 = a, var_1 = b, conditioning_set = cond_set)
        if pvalue < alpha:
            parents[b].add(a)
            children[a].add(b)
        return parents, children


    def path_tracing(self, 
                     topo_sort,
                     data, 
                     cit_test = "fisherz",
                     alpha = 0.001):
        
        '''
        Inputs:
        topo_sort - a reversed topological sort, where index(x)<index(y) implies y cannot cause x
        data - table that has columns for each variable and rows of samples (np matrix)
        '''
        
        #Initialization
        d = len(topo_sort)
        
        #find out how to build xrange
        parents = [set() for _ in range(d)]
        children = [set() for _ in range(d)]
        ind_collection = self.marg_ind(data, cit_test, alpha)
        
        #CIT Algorithm
        for i in range(1,d):
            b = topo_sort[i]
            for j in range(i-1,-1,-1):
                a = topo_sort[j]
                #only check for edge if a and b are marginally dependent
                if b in ind_collection[a]:
                    parents, children = self.edge_detection(a,
                                                            b,
                                                            parents,
                                                            children,
                                                            data,
                                                            ind_collection,
                                                            cit_test,
                                                            alpha)
        return parents, children


    def marg_ind(self, 
                 data, 
                 cit_test = "fisherz",
                 alpha = 0.001):
        
        '''Inputs: 
        data - table that has d columns for each variable and n rows of samples (np matrix)
        Returns:
        dxd symmetric matrix where matrix[i,j] = matrix[j,i] = 1 if x_i ind x_j, 0 otherwise.
        '''
        
        #Find Data Dimensionality
        d = data.shape[1]
        
        #Create matrix: there is a 0 in i,j (and j,i) if x_i \ind x_j, 1 otherwise.
        ind_collection = [set() for _ in range(d)]
        
        #Initialize independence test object.
        for i in range(d):
            for j in range(i+1, d):
                if cit_test != "oracle":
                    cit_test = CIT(data, cit_test)
                    pvalue = cit_test(i,j,[])
                else:
                    pvalue = self.oracle(var_0 = i, var_1 = j, conditioning_set = None)
                if pvalue < alpha:
                    ind_collection[i].add(j)
                    ind_collection[j].add(i)
        return ind_collection


    #Testing
    def adj_matrix(self, parent_sets):
        
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


    def standardize_vectors(self, *vectors):
        
        '''standardizes input vectors'''
        
        standardized_vectors = []
        for vector in vectors:
            mean = np.mean(vector)
            std = np.std(vector)
            standardized_vector = (vector - mean) / std
            standardized_vectors.append(standardized_vector)
        return standardized_vectors


    def DAG(self, n = 1000):
         
        '''produces standardized data from custom DAGs'''
         
        # DAG Structure.
        x = np.random.gamma(1,2,n)
        y = x+ np.random.gamma(1,2,n)
        z = y+ np.random.gamma(1,2,n)
        w = np.random.gamma(1,2,n)
        p = x + z + w + np.random.gamma(1,2,n)
        data = np.column_stack(standardize_vectors(x,y,z,w,p))
        return data

'''
#Testing
start_time = time.time()
#print(marg_ind(data))
sort = [0,1,2,3,4]
count = 0

print("Accuracy:", count/iter)
print(f"Runtime: {time.time() - start_time:.3f} s")
'''
