
import numpy as np 
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt 

P = np.random.uniform(size=(25,2))
n = P.shape[0]

## Nearest-neighbor
D = cdist(P,P)
map_zero = lambda A: np.fromiter(map(lambda x: np.inf if x == 0.0 else x, A), dtype=np.float64)
nn_ind = np.apply_along_axis(lambda A: np.argmin(map_zero(A)), arr=D, axis=0)
nn_edges = np.c_[np.fromiter(range(n), int), nn_ind]

## MST
import networkx as nx
from itertools import combinations, chain
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_weighted_edges_from([(i,j,w) for ((i,j), w) in zip(combinations(range(n),2), pdist(P))])
MST = np.array([(i,j,d['weight']) for (i,j,d) in nx.minimum_spanning_tree(G).edges(data=True)])
MST_edges = MST[:,:2].astype(int)

## Relative neighborhood graph
RNG = []
for (i,j) in combinations(range(n),2):
  w = D[i,j]
  is_rel_edge = True
  for r in np.setdiff1d(np.fromiter(range(n), int), np.array([i,j])):
    if D[i,r] < w and D[j,r] < w:
      is_rel_edge = False
  if is_rel_edge:
    RNG.append((i,j))
RNG_edges = np.array(RNG)

## Gabriel Graph 
gabriel_edges = []
for (i,j) in combinations(range(n), 2):
  r = D[i,j]/2.0 +np.finfo(float).resolution
  if np.sum(cdist(P, np.array([0.5*P[i,:] + 0.5*P[j,:]])) <= r) == 2:
    gabriel_edges.append((i,j))
gabriel_edges = np.array(gabriel_edges)

## Delaunay 
dt = Delaunay(P)
e = np.array([[[i,j], [i,k], [j,k]] for (i,j,k) in dt.simplices]).flatten()
del_edges = np.reshape(e, (int(len(e)/2), 2))

plot_graph(n, nn_edges, P)
plt.savefig('nn.png', transparent=True)
plot_graph(n, MST_edges, P)
plt.savefig('mst.png', transparent=True)
plot_graph(n, RNG_edges, P)
plt.savefig('rng.png', transparent=True)
plot_graph(n, gabriel_edges, P)
plt.savefig('gab.png', transparent=True)
plot_graph(n, del_edges, P)
plt.savefig('del.png', transparent=True)


del_edges
nn_edges





def plot_graph(n: int, edges, P):
  import networkx as nx 
  G = nx.Graph()
  G.add_nodes_from(range(n))
  Edges = []
  for (i,j,k) in dt.simplices:
    Edges.append([i,j])
    Edges.append([i,k])
    Edges.append([j,k])
  G.add_edges_from(Edges)
  fig = plt.figure(figsize=(5,5), dpi=300)
  ax = fig.gca()
  bc = np.fromiter(nx.betweenness_centrality(G).values(), float)
  ax.scatter(*P.T, zorder=2,s=(0.50+bc)*150, c='red')
  for (i,j) in edges:
    ax.plot(P[[i,j],0], P[[i,j],1], c='#80808068',linewidth=2.50, zorder=1)
  ax.set_aspect('equal')
  ax.axis('off')
# nx.draw_networkx(
#   G, 
#   pos = P, 
#   with_labels=False,
#   node_color='red',
#   node_size=bc*1050,
#   edge_color='gray'
# )
