import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn_extra.cluster import KMedoids
from pyclustering.utils.metric import distance_metric
from pyclustering.utils.metric import type_metric
from collections import defaultdict

from helper_print_function import *


def get_user_metric(distance_matrix):
    usr_metric = lambda p1, p2: distance_matrix[int(p1),int(p2)]
    dmat_metric = distance_metric(type_metric.USER_DEFINED, func=usr_metric)
    return dmat_metric


def prep_distance_matrix_tour(distance_matrix, home_vertex):
    """ Prepares the distance matrix for a tour. The home_vertex is removed from the distance matrix.
    The rest of the vertices are clustered
    """
    #home_distances = distance_matrix[home_vertex,:]
    distance_matrix = np.delete(distance_matrix, home_vertex, 0)
    distance_matrix = np.delete(distance_matrix, home_vertex, 1)
    # -- return the distance_matrix
    return distance_matrix


def prep_distance_matrix_path(distance_matrix, home_vertex, current_vertex):
    """ Prepares the distance matrix for a tour. The home_vertex and current_vertex 
    is removed from the distance matrix. The rest of the vertices are clustered.
    """
    cur_distances = distance_matrix[cur_vertex,:]
    #for cd in cur_distances:
        #print (f"{cd}", end="\t")
    #print()    
    # -- as removing the home_vertex will change the indices in the matrix
    if current_vertex > home_vertex:
        current_vertex = current_vertex - 1
    
    # -- delete the vertices
    distance_matrix = np.delete(distance_matrix, home_vertex, 0)
    distance_matrix = np.delete(distance_matrix, home_vertex, 1)
    distance_matrix = np.delete(distance_matrix, current_vertex, 0)
    distance_matrix = np.delete(distance_matrix, current_vertex, 1)
    #print ("Func: prep_distance_matrix_path: Clipped Distance_matrix")
    #print_distance_matrix(distance_matrix)
    return distance_matrix


def cluster_vertices(distance_matrix, n_clusters):
    """ Cluster vertex nodes based on the distance matrix between them.
    Input:
        distance_matrix
        n_clusters
    """
    # -- get the metric
    usr_def_metric = get_user_metric(distance_matrix)
    n_vertices = len(distance_matrix)
    X = np.linspace(0,n_vertices-1,n_vertices).astype('int')
    X = X.reshape(-1,1) # make it a column array
    cobj = KMedoids(n_clusters, metric=usr_def_metric, method='pam', init='build').fit(X)
    labels = cobj.labels_
    cluster_centers = cobj.cluster_centers_
    return labels, cluster_centers


def get_vertex_map(distance_matrix, labels, cluster_centers):
    """ Creates a vertex map using labels
    vertex_map takes the regular vertex and converts it to new vertex ID.
    rev_vertex_map takes a converted vertex ID and converts it back to the old vertex ID.
    """
    vertex_map = {}
    rev_vertex_map = {}
    # -- the number of unique labels are number of vertices
    # the labels would convert regular nodes to the converted node 
    # (this isn't really necessary as we'll be deleting the vertices anyway)
    # home vertex should be vertex zero
    cluster_centers = cluster_centers.tolist()
    print (cluster_centers)
    for new_v_id, cc in enumerate(cluster_centers):
        print ("cc: ", cc[0])
        print ("type(cc): ", type(cc))
        vertex_map[cc[0]] = new_v_id
        rev_vertex_map[new_v_id] = cc[0]

    for v_id, l in enumerate(labels):
        print (f"Vertex: {v_id}\tLabel: {l}\tCluster_center: {cluster_centers[l]}")
        
    print ("Vertex map: ")
    for key, val in vertex_map.items():
        print (f"Key: {key} Val: {val}")
    
    print ("Reverse Vertex map: ")
    for key, val in rev_vertex_map.items():
        print (f"Key: {key} Val: {val}")
   
   
def get_distance_matrix(distance_matrix, labels, cluster_centers):
    """ Get a new distance matrix
    """
    cluster_centers = np.squeeze(cluster_centers.transpose())
    print (cluster_centers)
    cc = cluster_centers.tolist()
    cc = tuple(cc)
    #print(f"Cluster centers: {cc}")
    new_dist_mat = distance_matrix[cc,:].copy()
    new_dist_mat = np.squeeze(new_dist_mat)
    new_dist_mat = np.squeeze(new_dist_mat[:,cc])
    return new_dist_mat


def test_new_dist_mat(distance_matrix, new_dist_mat, cluster_centers):
    cc = np.squeeze(cluster_centers.transpose())
    l = cc.shape[0]
    check = True
    for i in range(l):
        for j in range(l):
            print (f"{i} {j}\t{cc[i]},{cc[j]}\tOriginal: {distance_matrix[cc[i],cc[j]]}\t{new_dist_mat[i,j]}\t<--Calculated")
            if distance_matrix[cc[i],cc[j]] != new_dist_mat[i,j]:
                print ("ERROR! Check failed, new distance matrix is inaccurate")
                raise(ValueError)
    print ("All Okay! :)")


def get_distance_matrix_with_home(distance_matrix, labels, cluster_centers, home_vertex: int=0, cur_vertex: int=-1):
    """ Get new distance matrix with cluster labels.
    1. Updates the cluster labels with home vertex and current vertex.
    2. 
    """
    cluster_centers = np.squeeze(cluster_centers.transpose())
    l = cluster_centers.shape[0]
    # -- add the home vertex to the distance_matrix
    # NOTE: not updating labels
    # -- increment cluster_centers by 1 for elements >= home_index
    #int insert_home_pos = None;
    for idx in range(cluster_centers.shape[0]):
        if cluster_centers[idx] >= home_vertex:
            cluster_centers[idx]+=1
    cluster_centers = np.insert(cluster_centers, 0, home_vertex)
    # -- increment cluster_centers by 1 for elements >= cur_vertex
    if cur_vertex >= 0:
        idx = 0
        for idx in range(cluster_centers.shape[0]):
            if cluster_centers[idx]>=cur_vertex:
                cluster_centers[idx]+=1
        cluster_centers = np.insert(cluster_centers, 1, cur_vertex) # insertion node does not matter
    cc = tuple(cluster_centers)
    new_dist_mat = np.squeeze(distance_matrix[cc,:].copy())
    new_dist_mat = np.squeeze(new_dist_mat)
    new_dist_mat = np.squeeze(new_dist_mat[:,cc])
    return new_dist_mat, cluster_centers


def get_prizes(prize_list, rev_vertex_map, labels):
    """ Calculate labels based on clusters.
    Input:
        labels: A list of n vertices, where a label is assigned to each entry in vertex map
    Output:
        prize_list: A list of unique_labels with a prize assigned to each label.
                    Each prize is calculated by doing an average of the vertices in a cluster (label)
    """
    # -- initialize prize list
    unique_labels = set(labels)
    prize_list = np.zeros(len(unique_labels))
    n_elems = np.zeros(len(unique_labels), dtype='int')
    # -- loop through the label list
    for vertex_id in range(labels.shape[0]):
        g_v_id = rev_vertex_map[vertex_id] # graph vertex id
        # -- get prize from occ_graph_dict
        node_priority = prize_list[g_v_id]
        # -- update n_elems
        n_elems[labels[vertex_id]]+=1 # update 
        prize_list[labels[vertex_id]]+=node_priority # sum of prizes
    
    # -- divide prize list by n_elem to get average
    #for a label to exist, there would be at least one element at each entry of n_elems
    prize_list = prize_list/n_elems 
    return prize_list


def read_graph_file(filename):
    file_data = []
    with open(filename, "r") as f:
        line = f.readline()
        while(line!=""):
            line = line.strip()
            line = line.split()
            line_data = [float(i) for i in line]
            #print(line_data)
            file_data.append(line_data)
            line = f.readline()
    return file_data
        

if __name__ == '__main__':
    print ("K medoids clustering in python")
    print ("Input graph file")
    graph_file = "graph_file.txt"
    
    # ----------------------------------------------------------------------------------------------
    # -- test the basic working on the distance matrix reduction
    # -- No home vertex and current vertex
    # ----------------------------------------------------------------------------------------------
    if len(sys.argv) == 1:
        distance_matrix = np.array(read_graph_file(graph_file))
        print(distance_matrix.shape)
        labels, cluster_centers = cluster_vertices(distance_matrix, 6)
        unique_labels = set(labels)
        new_dist_mat = get_distance_matrix(distance_matrix, labels, cluster_centers)
        test_new_dist_mat(distance_matrix, new_dist_mat, cluster_centers)
    
    # ----------------------------------------------------------------------------------------------
    # -- test with home vertex
    # ----------------------------------------------------------------------------------------------
    if len(sys.argv) == 2:
        home_vertex = int(sys.argv[1])
        distance_matrix = np.array(read_graph_file(graph_file))
        print("Shape of input matrix: ", distance_matrix.shape)
        # -- cluster vertices after removing home vertex
        clipped_distance_matrix  = prep_distance_matrix_tour(distance_matrix, home_vertex)
        # -- cluster the vertices
        labels, cluster_centers = cluster_vertices(clipped_distance_matrix, 6)
        # -- get new distance matrix from original distance_matrix
        new_dist_mat, cluster_centers = get_distance_matrix_with_home(distance_matrix, labels, cluster_centers, home_vertex)
        # -- check
        test_new_dist_mat(distance_matrix, new_dist_mat, cluster_centers)
    
    # ----------------------------------------------------------------------------------------------
    # -- test with home and current vertex
    # ----------------------------------------------------------------------------------------------
    if len(sys.argv) == 3:
        home_vertex = int(sys.argv[1])
        cur_vertex = int(sys.argv[2])
        distance_matrix = np.array(read_graph_file(graph_file))
        print("Shape of input matrix: ", distance_matrix.shape)
        # -- cluster vertices after removing home vertex
        clipped_distance_matrix = prep_distance_matrix_path(distance_matrix, home_vertex, cur_vertex)
        # -- cluster the vertices
        labels, cluster_centers = cluster_vertices(clipped_distance_matrix, 6)
        # -- get new distance matrix
        new_dist_mat, cluster_centers = get_distance_matrix_with_home(distance_matrix, labels, cluster_centers, home_vertex, cur_vertex)
        # -- check
        test_new_dist_mat(distance_matrix, new_dist_mat, cluster_centers)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
