import numpy

def print_distance_matrix_cluster_centers(dist_matrix, cluster_centers):
    cluster_centers = np.squeeze(cluster_centers.transpose())
    l = cluster_centers.shape[0]
    print(dist_matrix.shape)
    print("Clustered distance matrix without home or current vertex: ")
    for i in cluster_centers:
        print(f"{i}\t",end="")
        for j in range(dist_matrix.shape[1]):
            print(f"{dist_matrix[i,j]:.1f}", end="\t")
        print()
    print("="*100)

def print_distance_matrix(dist_matrix):
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            print(f"{dist_matrix[i,j]:.1f} ", end="\t")
        print()
    print("="*100)

def print_distance_matrix_with_indices(dist_matrix):
    print("\t",end="")
    for i in range(dist_matrix.shape[1]):
        print(f"{i}", end="\t")
    print()
    for i in range(dist_matrix.shape[0]):
        print(f"{i}",end="\t")
        for j in range(dist_matrix.shape[1]):
            print(f"{dist_matrix[i,j]:.1f} ", end="\t")
        print()
    print("="*100)
        
