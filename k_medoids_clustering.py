from email.policy import default
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
    distance_matrix = np.delete(distance_matrix, home_vertex, 0)
    distance_matrix = np.delete(distance_matrix, home_vertex, 1)
    # -- return the distance_matrix
    return distance_matrix


def prep_distance_matrix_path(distance_matrix, home_vertex, current_vertex):
    """ Prepares the distance matrix for a tour. The home_vertex and current_vertex 
    is removed from the distance matrix. The rest of the vertices are clustered.
    """
    # -- as removing the home_vertex will change the indices in the matrix
    if current_vertex > home_vertex:
        current_vertex = current_vertex - 1
    if home_vertex > current_vertex:
        home_vertex = home_vertex - 1
    
    # -- delete the vertices
    distance_matrix = np.delete(distance_matrix, home_vertex, 0)
    distance_matrix = np.delete(distance_matrix, home_vertex, 1)
    distance_matrix = np.delete(distance_matrix, current_vertex, 0)
    distance_matrix = np.delete(distance_matrix, current_vertex, 1)
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
    # make the variables of a graph in a linear space X
    # this is possible as we have a custom distance metric defined
    X = np.linspace(0,n_vertices-1,n_vertices).astype('int')
    X = X.reshape(-1,1) # make it a column array
    cobj = KMedoids(n_clusters, metric=usr_def_metric, method='pam', init='build').fit(X)
    labels = cobj.labels_
    cluster_centers = cobj.cluster_centers_
    return labels, cluster_centers
   
   
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

def print_cluster_prize(labels, init_prize_list, clustered_prize_list, cluster_centers, home_vertex, cur_vertex):
    clusters_vertices = defaultdict(lambda:[])
    prize_clusters = defaultdict(lambda:[])

    for idx, l in enumerate(labels):
        cluster_id = cluster_centers[l]
        if (idx >= home_vertex):
            idx += 1
        if (idx >= cur_vertex) :
            idx += 1
        clusters_vertices[cluster_centers[l]].append(idx)
        prize_clusters[cluster_centers[l]].append(init_prize_list[idx])
    
    # -- print the prize list
    clusters_vertices = dict(clusters_vertices)
    prize_clusters = dict(prize_clusters)
    for id,c in enumerate(cluster_centers):
        print (f"Cluster center: {c}")
        print (f"Prize: {clustered_prize_list[id]}")
        if c in clusters_vertices:
            print (f"Element prizes:")
            for elem_idx, v_id in enumerate(clusters_vertices[c]):
                print(f"\tVertex:{clusters_vertices[c][elem_idx]}\tPrize: {prize_clusters[c][elem_idx]}")
        else:
            print(f"Element not clustered")
        print("-"*50)
    return
            
        

def get_average_prize_clusters(labels, cluster_centers, init_prize, home_vertex=None, cur_vertex=None):
    new_prize_dict = defaultdict(lambda:0)
    n_elements_cluster = defaultdict(lambda:0)
    if home_vertex != None:
        labels += 1
        new_prize_dict[home_vertex] = init_prize[home_vertex]
    if cur_vertex != None:
        labels += 1
        new_prize_dict[cur_vertex] = init_prize[cur_vertex]
        
    # -- get off the prizes from the labels
    for elem_idx, l in enumerate(labels):
        cluster_id = cluster_centers[l]
        if (elem_idx >= home_vertex):
            elem_idx += 1
        if (elem_idx >= cur_vertex) :
            elem_idx += 1
        new_prize_dict[cluster_id] += init_prize[elem_idx]
        n_elements_cluster[cluster_id] += 1
    
    n_elements_cluster = dict(n_elements_cluster)
    new_prize_dict = dict(new_prize_dict)
    # -- divide by the number of elements
    for key, _ in n_elements_cluster.items():
        new_prize_dict[key] = new_prize_dict[key] / n_elements_cluster[key] 
    
    # -- convert to list
    prize_list = []
    new_prize_dict = dict(new_prize_dict)
    for c in cluster_centers:
        prize_list.append(new_prize_dict[c])
    return prize_list, labels


def read_prize_file(filename):
    with open(filename, "r") as f:
        line = f.readline()
        line = line.strip()
        line = line.split()
        line_data = [float(i) for i in line]
        return line_data
        

if __name__ == '__main__':
    print ("K medoids clustering in python")
    print ("Input graph file")
    graph_file = "graph_file.txt"
    n_clusters = 6
    prize_file = "prize_list.txt"
    prizes = np.array(read_prize_file(prize_file))
    print (prizes)
    print(prizes.shape)
    # sys.exit(0)

    
    # ----------------------------------------------------------------------------------------------
    # -- test the basic working on the distance matrix reduction
    # -- No home vertex and current vertex
    # ----------------------------------------------------------------------------------------------
    if len(sys.argv) == 1:
        distance_matrix = np.array(read_graph_file(graph_file))
        print(distance_matrix.shape)
        labels, cluster_centers = cluster_vertices(distance_matrix, n_clusters)
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
        labels, cluster_centers = cluster_vertices(clipped_distance_matrix, n_clusters)
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
        labels, init_cluster_centers = cluster_vertices(clipped_distance_matrix, n_clusters)
        # -- get new distance matrix
        new_dist_mat, cluster_centers = get_distance_matrix_with_home(distance_matrix, labels, init_cluster_centers, home_vertex, cur_vertex)
        # -- check
        test_new_dist_mat(distance_matrix, new_dist_mat, cluster_centers)
        # -- prizes
        new_prize_list, new_labels  = get_average_prize_clusters(labels,cluster_centers, prizes, home_vertex, cur_vertex)
        # -- test the prizes
        print_cluster_prize(labels, prizes, new_prize_list, cluster_centers, home_vertex, cur_vertex)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
