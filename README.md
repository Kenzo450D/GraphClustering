# GraphClustering

This code makes a graph cluster using the kmedoids clustering algorithm.

## Run the code
To run the code without a home or current vertex:<br>
`python3 k_medoids_clustering.py`<br>

To run the code without a current vertex and only the home (depot) vertex: <br>
`python3 k_medoids_clustering.py <home_vertex:int>`<br>
**Example:** `python3 k_medoids_clustering.py 0`

To run the code with home(depot) and current vertex: <br>
`python3 k_medoids_clustering.py <home_verteix:int> <current_vertex:int>`<br>
**Example:** `python3 k_medoids_clustering.py 0 10`
