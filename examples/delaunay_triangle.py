

import numpy as np 
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt 

from scipy.spatial import Delaunay

T = np.array([[0.1,-1], [1,0.05], [0.1,1], [-1,0.35]])
dt = Delaunay(T)

plt.scatter(*T.T)
S1, S2 = dt.simplices[0], dt.simplices[1]
plt.plot(*T[np.append(S1, S1[0]),:].T)  # blue
plt.plot(*T[np.append(S2, S2[0]),:].T)  # orange
plt.gca().set_aspect('equal')

p,q = np.intersect1d(S1,S2)
rm = np.setdiff1d(S1, [p,q]).item()
rp = np.setdiff1d(S2, [p,q]).item()

plt.text(*T[p,:].T, s='p')
plt.text(*T[q,:].T, s='q')
plt.text(*T[rm,:].T, s='rm')
plt.text(*T[rp,:].T, s='rp')

norm = lambda x: np.linalg.norm(x)
def triangle_angle(p: ArrayLike, r: ArrayLike, q: ArrayLike):
  """ Returns \angle prq in radians """
  RP, RQ = p-r, q-r
  return(np.arccos(np.dot(RP, RQ)/(norm(RP)*norm(RQ))))

prq1 = triangle_angle(*T[[p,rm,q],:])
prq2 = triangle_angle(*T[[p,rp,q],:])
assert prq1 + prq2 < np.pi



np.rad2deg(triangle_angle(*np.array([[0,0], [1,0], [0,1]])))
np.rad2deg(triangle_angle(*np.array([[1,0], [0,0], [0,1]])))
np.rad2deg(triangle_angle(*np.array([[-1,1], [0,0], [1,0]])))