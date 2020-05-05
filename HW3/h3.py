import numpy as np 

data = []

#push edge Eij to the data [vi, vj, |Eij|]
data.append([0, 1, 2])
data.append([0, 2, 2])
data.append([1, 2, 3])
data.append([1, 3, 1])
data.append([3, 4, 2])
data.append([3, 6, 4])
data.append([4, 5, 2])
data.append([4, 6, 3])
data.append([4, 7, 1])
data.append([5, 6, 4])
data.append([5, 11, 1])
data.append([7, 8, 3])
data.append([7, 9, 2])
data.append([7, 11, 2])
data.append([8, 16, 1])
data.append([8, 9, 3])
data.append([8, 11, 4])
data.append([9, 10, 4])
data.append([9, 12, 1])
data.append([10, 11, 3])
data.append([11, 14, 1])
data.append([12, 14, 2])
data.append([12, 13, 3])
data.append([13, 15, 4])
data.append([13, 14, 3])

#construct adjacency matrix
adjmat = np.zeros((17, 17))
for i in range(17):
    adjmat[i][i] = 1    #diagonal

for edge in data:
    adjmat[edge[0]][edge[1]] = edge[2]
    adjmat[edge[1]][edge[0]] = edge[2]    #symmetric matrix

#normalize adjacency matrix
adj_sum = np.sum(adjmat, axis = 0)

for i in range(17):
    adjmat[:, i] = np.divide(adjmat[:, i], adj_sum[i])

#mcl
infla = [1.1, 1.3, 1.5, 1.7, 2.1]

import markov_clustering as mcl

final_mat = []
final_cl = []
for i in infla:
    result = mcl.run_mcl(adjmat, inflation = i, iterations = 10)
    final_mat.append(result)
    cluster = mcl.get_clusters(result)
    final_cl.append(cluster)

print(result)
k = 4
mcl.draw_graph(final_mat[k], final_cl[k], with_labels = True, edge_color = "black")