## Load all exported functions in the package 
import numpy as np 
from geom_dtn import *
from geom_dtn import sat_contact_plan
from datetime import datetime, timedelta, time, date, timezone

## Load default set of satellites
satellites = load_satellites()

## Make dynamic point cloud (dpc) of select satellites (S)
I = np.array([0, 1673, 1529, 1545, 1856,   68, 1513, 2016, 2060,  750, 1120, 480, 1816,  516, 1543, 1547,  584,  299, 2070, 1984, 1056, 2257, 64, 1809, 1522, 1409, 1125,  632,   41,  551])
I = np.append(I, np.random.choice(np.setdiff1d(range(len(satellites)), I), size=60))
S = np.array(satellites)[I]

## Start and end time
st = min([s.epoch.utc_datetime() for s in S])
et = st + timedelta(minutes=182)

## Dynamic point cloud
f, (b,e) = satellite_dpc(S, s_time = st, e_time = et)

## Generate a contact plan over a time interval (s,e). 
## This is O(S^2), and may be slow if either the resolution or |S| is large.
sat_CP, (st, et) = sat_contact_plan(S, s_time = st, e_time = et, resolution=50, progress=True)

## 15 seconds === 15k ms / 250 == 60 ms between each frame
n_time_pts = 250
time_points_dt = st + np.array([timedelta(seconds=i*((et-st).seconds/n_time_pts)) for i in range(n_time_pts)])
# time_points_time = np.array([datetime_to_time(tp) for tp in time_points_dt])
time_points = np.linspace(0, 1, n_time_pts)

## Compute delaunay using stereographic projection
from scipy.spatial import Delaunay
P = f(0.0, 'geocentric')  ## 3d cartesian coordinates
Pn = normalize(P, axis=1) ## project to 2-sphere 
X, Y = Pn[:,0]/(1-Pn[:,2]), Pn[:,1]/(1-Pn[:,2])
dt = Delaunay(np.c_[X, Y])

## Plot 2D earth plot at time 0 
import cartopy.crs as ccrs
import networkx as nx
t = 0.0
lonlat = f(0.0, 'lonlat')

## Plot 2D projection of earth
fig, ax = plot_earth_2D(0.20)

## Plot the nodes colored by centrality
G = nx.Graph()
G.add_nodes_from(range(len(S)))
keys = G.add_edges_from([(int(i),int(j),{ "start": s, "end": e }) for (i,j,s,e) in sat_CP if (s <= t and t <= e)])
v_centrality = np.array(list(nx.betweenness_centrality(G).values()))
ax.scatter(*lonlat.T, s=v_centrality*1000, transform=ccrs.PlateCarree(), cmap='viridis', c=v_centrality, zorder=30)

## Plot visibility lines
visible_edge_pts = ((lonlat[i,:], lonlat[j,:]) for (i,j) in edges_at(t, sat_CP))
los_lines = wrap_lines(visible_edge_pts, x_interval=[-180,180])
for line in los_lines:
  ax.plot(*line.T, c='gray', linewidth=0.50, zorder=10)

## Plot delaunay subgraph 
from itertools import chain
g1 = ((lonlat[i,:], lonlat[j,:]) for i,j,k in dt.simplices if (i,j) in G.edges)
g2 = ((lonlat[i,:], lonlat[k,:]) for i,j,k in dt.simplices if (i,k) in G.edges)
g3 = ((lonlat[j,:], lonlat[k,:]) for i,j,k in dt.simplices if (j,k) in G.edges)
del_lines = wrap_lines(chain(g1,g2,g3), x_interval=[-180,180])
for line in del_lines:
  ax.plot(*line.T, c='orange', linewidth=0.75, zorder=20)


## 3D Earth animation 
from matplotlib.animation import FuncAnimation
LL = f(0.0, 'geocentric')
fig, ax = plot_earth_3D(wireframe=False, alpha=1.0, zorder=1, scale=1, figsize=(16,16), dpi=320)
points = ax.scatter([], [], color="green", zorder=4)
lines, = ax.plot([], [], color="crimson", zorder=4)

def get_visible(Z, azimuth, elev):
  a = azimuth*np.pi/180. -np.pi
  e = elev*np.pi/180. - np.pi/2.
  X = [ np.sin(e) * np.cos(a),np.sin(e) * np.sin(a),np.cos(e)]  
  cond = (np.dot(Z,X) >= 0)
  return(cond)

## Projection of points
visible = get_visible(LL, ax.azim, ax.elev)
ax.scatter(*LL[visible,:].T * 1.05, c='blue', s=15, alpha=1.0, zorder=10)
for p in LL[visible,:]:
  u = p/np.linalg.norm(p)
  q = u * 6378
  p = u * 1.05*np.linalg.norm(p)
  ax.plot(*np.vstack((p, q)).T, c='red', linewidth=3, zorder=9)
ax.axis('off')

def animate(t):
  LL = f(t, 'geocentric')
  ## Update satellite positions
  visible_pts = get_visible(LL, ax.azim, ax.elev)
  
  visible_edge_pts = ((LL[i,:], LL[j,:]) for (i,j) in edges_at(t, sat_CP))
  for p0, p1 in visible_edge_pts:
    ax.plot(*np.vstack((p0,p1)).T, c='gray', linewidth=0.50, zorder=10)


  points._offsets3d = (LL[visible_pts,0], LL[visible_pts,1], LL[visible_pts,2])
  # plot_visible()
  print(t) # status

# http://matplotlib.sourceforge.net/api/animation_api.html
anim = FuncAnimation(fig, animate, frames=np.linspace(0.0, 0.05, 10), interval=400, blit=False)
anim.save('delaunay_sats3d.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()


## 2D cylindrical projection animation 
from matplotlib.animation import FuncAnimation
LL = f(0.0, 'latlon')
fig, ax = plot_earth_2D()
points = ax.scatter([], [], color="green", zorder=4, s=4.5)
# lines, = ax.plot([], [], color="crimson", zorder=4)
# lines = []
def animate(t):
  # for l in lines: 
  #   l.remove()
  # lines.clear()
  nl = len(ax.lines)
  for i in range(nl):
    ax.lines.pop()
  LL = f(t, 'lonlat')
  visible_edge_pts = ((LL[i,:], LL[j,:]) for (i,j) in edges_at(t, sat_CP))
  los_lines = wrap_lines(visible_edge_pts, x_interval=[-180,180])
  for line in los_lines:
    lines.append(ax.plot(*line.T, c='gray', linewidth=0.50, zorder=1))
  ## Update satellite positions
  points.set_offsets(LL)

  ## Plot delaunay subgraph 
  P = f(t, 'geocentric') 
  Pn = normalize(P, axis=1) ## project to 2-sphere 
  X, Y = Pn[:,0]/(1-Pn[:,2]), Pn[:,1]/(1-Pn[:,2])
  dt = Delaunay(np.c_[X, Y])
  ## Plot the nodes colored by centrality
  G = nx.Graph()
  G.add_nodes_from(range(len(S)))
  keys = G.add_edges_from([(int(i),int(j),{ "start": s, "end": e }) for (i,j,s,e) in sat_CP if (s <= t and t <= e)])
  # v_centrality = np.array(list(nx.betweenness_centrality(G).values()))
  # ax.scatter(*LL.T, s=v_centrality*1000, transform=ccrs.PlateCarree(), cmap='viridis', c=v_centrality, zorder=30)

  from itertools import chain
  g1 = ((LL[i,:], LL[j,:]) for i,j,k in dt.simplices if (i,j) in G.edges)
  g2 = ((LL[i,:], LL[k,:]) for i,j,k in dt.simplices if (i,k) in G.edges)
  g3 = ((LL[j,:], LL[k,:]) for i,j,k in dt.simplices if (j,k) in G.edges)
  del_lines = wrap_lines(chain(g1,g2,g3), x_interval=[-180,180])
  for line in del_lines:
    ax.plot(*line.T, c='orange', linewidth=0.90, zorder=2)

  print(t)
anim = FuncAnimation(fig, animate, frames=np.linspace(0.0, 0.50, 300), interval=400, blit=False)
anim.save('delaunay_sats2d.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()


## Results
from scipy.spatial import Delaunay
from itertools import combinations

resolution = 400
Xt, Yt = [], []
for t in np.linspace(0, 1.0, resolution):
  P = f(t, 'geocentric')  ## 3d cartesian coordinates
  Pn = normalize(P, axis=1) ## project to 2-sphere 

  X, Y = Pn[:,0]/(1-Pn[:,2]), Pn[:,1]/(1-Pn[:,2])
  Xt.append(X)
  Yt.append(Y)

Xt = np.vstack(Xt)
Yt = np.vstack(Yt)

## Rough approximation:  get *lower bound* on number of changes via discrete steps
from itertools import chain, pairwise
RD = []
for i in progressbar(range(Xt.shape[0]), count=Xt.shape[0]):
  t = np.linspace(0, 1.0, resolution)[i]
  P = f(t, 'geocentric')
  G = nx.Graph()
  G.add_nodes_from(range(len(S)))
  keys = G.add_edges_from(edges_at(t, sat_CP))
  dt = Delaunay(np.c_[Xt[i,:], Yt[i,:]]).simplices
  constrained_edges = np.array([all([(i,j) in G.edges, (i,k) in G.edges, (j,k) in G.edges]) for (i,j,k) in dt])

  ## alpha-stable edges only
  E_alpha = []
  alpha = np.pi*0.50
  for t1, t2 in combinations(dt[constrained_edges,:], 2):
    if len(np.intersect1d(t1,t2)) == 2:
      p,q = np.intersect1d(t1,t2)
      rm = np.setdiff1d(t1, [p,q]).item()
      rp = np.setdiff1d(t2, [p,q]).item()
      prq1 = triangle_angle(*P[[p,rm,q],:])
      prq2 = triangle_angle(*P[[p,rp,q],:])
      #assert prq1 + prq2 < np.pi
      if prq1 + prq2 <= np.pi - alpha:
        E_alpha.append(rank_comb([p,q], k=2, n=len(S)))
  RD.append(np.unique(E_alpha))
  #RD.append(rank_combs(edges_from_triangles(dt[constrained_edges,:], nv=len(S)), k = 2, n = len(S)))
n_crit = sum([len(np.setdiff1d(B, A)) for A,B in pairwise(RD)])
n_crit += len(RD[0])

from scipy.sparse.csgraph import floyd_warshall


alpha_10 = [len(np.setdiff1d(B, A)) for A,B in pairwise(RD)]
alpha_20 = [len(np.setdiff1d(B, A)) for A,B in pairwise(RD)]
alpha_30 = [len(np.setdiff1d(B, A)) for A,B in pairwise(RD)]
alpha_40 = [len(np.setdiff1d(B, A)) for A,B in pairwise(RD)]
alpha_50 = [len(np.setdiff1d(B, A)) for A,B in pairwise(RD)]




base = [len(np.setdiff1d(B, A)) for A,B in pairwise(RE)]
n_time_pts = np.linspace(0, 1, resolution-1, endpoint=False)
plt.plot(n_time_pts, np.cumsum(base)+np.cumsum([len(d) for d in RE])[1:], c='black', label='Vis(G)')
excess = np.cumsum([len(d) for d in RD])[1:]
# plt.plot(n_time_pts, np.cumsum(alpha_10)+excess, label='0.90-stable')
# plt.plot(n_time_pts, np.cumsum(alpha_20)+excess, label='0.80-stable')
# plt.plot(n_time_pts, np.cumsum(alpha_30)+excess, label='0.70-stable')
# plt.plot(n_time_pts, np.cumsum(alpha_40)+excess, label='0.60-stable')
plt.plot(n_time_pts, np.cumsum(alpha_50)+excess, label='2-stable')
plt.gca().set_title("90 random satellites from STAR-LINK")
plt.suptitle("Cum. edges in (a-stable)-DT(G) vs. Vis(G)", y=1.025, fontsize=16)
plt.ylabel("Cum. topological events + # edges")
plt.xlabel("Fraction of orbital period")
plt.gca().legend()
plt.gca().set_yscale('log')

## Rough approximation:  get *lower bound* on number of regular edge changes -- should be lower than contact plan
from itertools import chain, pairwise
RE = []
for i in range(Xt.shape[0]):
  t = np.linspace(0, 1.0, resolution)[i]
  RE.append(rank_combs(edges_at(t, sat_CP), k=2, n=len(S)))
n_crit = sum([len(np.setdiff1d(B, A)) for A,B in pairwise(RE)])
n_crit += len(RE[0])

e_changes = np.array([len(np.setdiff1d(B, A)) for A,B in pairwise(RE)])
del_changes = np.array([len(np.setdiff1d(B, A)) for A,B in pairwise(RD)])

sym_diff = lambda A,B: np.setdiff1d(np.union1d(A, B), np.intersect1d(A, B))


## Try to get actual number of critical events via (1+epsilon)-spline approximation
time_dom = np.linspace(0, 1.0, resolution)
stereo_pos = {}
for i in range(Xt.shape[1]):
  fx = CubicSpline(time_dom, Xt[:,i])
  fy = CubicSpline(time_dom, Yt[:,i])
  f_xy = CubicSpline(time_dom, Xt[:,i]**2 + Yt[:,i]**2)
  stereo_pos[i] = { 'x' : fx, 'y' : fy, 'xy' : f_xy }

from math import comb
n_crit = 0
sp = stereo_pos
for clique in nx.find_cliques(G):
  for a,b,c,p in combinations(sorted(clique), 4):
    cc_det = []
    for t in time_dom:
      circum_circle = np.array([
        [ sp[a]['x'](t), sp[a]['y'](t), sp[a]['xy'](t), 1 ],
        [ sp[b]['x'](t), sp[b]['y'](t), sp[b]['xy'](t), 1 ],
        [ sp[c]['x'](t), sp[c]['y'](t), sp[c]['xy'](t), 1 ],
        [ sp[p]['x'](t), sp[p]['y'](t), sp[p]['xy'](t), 1 ] 
      ])
      cc_det.append(np.linalg.det(circum_circle))
  r = CubicSpline(time_dom, np.array(cc_det)).roots()
  print(clique)
  n_crit += len(list(filter(lambda x: x >= min(time_dom) and x <= max(time_dom), r)))

plt.plot(cc_det)
from scipy.spatial import ConvexHull

# x0 = Xt[0, a], Xt[0, b], Xt[0, c], Xt[0, p]
# y0 = Yt[0, a], Yt[0, b], Yt[0, c], Yt[0, p]

# plt.scatter(*np.c_[x0,y0].T)
# plt.gca().set_aspect('equal')
# for x,y,s in zip(x0,y0,['a','b','c','p']):
#   plt.text(x,y,s=s)
# a,b,c,p = 36,45,59,29
# circum_circle = np.array([
#   [ sp[a]['x'](t), sp[a]['y'](t), sp[a]['xy'](t), 1 ],
#   [ sp[b]['x'](t), sp[b]['y'](t), sp[b]['xy'](t), 1 ],
#   [ sp[c]['x'](t), sp[c]['y'](t), sp[c]['xy'](t), 1 ],
#   [ sp[p]['x'](t), sp[p]['y'](t), sp[p]['xy'](t), 1 ] 
# ])
# np.linalg.det(circum_circle)

dt = Delaunay(np.c_[X, Y])

fig = plt.figure(figsize=(5,5))
ax = plt.gca()
plt.plot(Xt[:,0], Yt[:,0])

from geom_dtn import interpolated_intercepts
S[0].at(time_points[0])



from geom_dtn import data as package_data_mod
load = Loader(package_data_mod.__path__._path[0])
ts = load.timescale()
time_points = ts.linspace(ts.from_datetime(st), ts.from_datetime(et), resolution)
tp_tt = np.array([tp-ts.from_datetime(st) for tp in time_points]) # time points numerical values
tp_tt = (tp_tt - np.min(tp_tt))/(np.max(tp_tt) - np.min(tp_tt))

sat_dist = {}
for i,j in combinations(range(len(S)), 2):
  sat1, sat2 = S[i], S[j]
  sat1_orbit = sat1.at(time_points).position.km.T
  sat2_orbit = sat2.at(time_points).position.km.T
  f = CubicSpline(np.linspace(0, 1, resolution), np.linalg.norm(sat1_orbit - sat2_orbit, axis=1))
  sat_dist[(i,j)] = f

dom = np.linspace(0,1,resolution)
plt.plot(dom, sat_dist[(0,2)](dom))

## Build the global 'multi-graph' representation from the contact plan
import networkx as nx
G = nx.MultiGraph()
G.add_nodes_from(range(len(S)))
keys = G.add_edges_from([(int(i),int(j),{ "start": s, "end": e }) for (i,j,s,e) in sat_CP])



from math import comb
n_crit = 0
for clique in nx.find_cliques(G):
  for a,b,c,d in combinations(sorted(clique), 4):
    lhs = sat_dist[(a,c)](dom)*sat_dist[(b,d)](dom)
    rhs = sat_dist[(a,b)](dom)*sat_dist[(c,d)](dom) + sat_dist[(b,c)](dom)*sat_dist[(a,d)](dom)
    print(sum(abs(lhs - rhs)))
    crit_x, crit_y = interpolated_intercepts(dom, lhs, rhs)
    n_crit += len(crit_x)
  print(clique)

# for (a,b,c,d) in progressbar(combinations(range(len(S)), 4), count=comb(len(S), 4)):
#   lhs = sat_dist[(a,c)](dom)*sat_dist[(b,d)](dom)
#   rhs = sat_dist[(a,b)](dom)*sat_dist[(c,d)](dom) + sat_dist[(b,c)](dom)*sat_dist[(a,d)](dom)
#   crit_x, crit_y = interpolated_intercepts(dom, lhs, rhs)
#   n_crit += len(crit_x)



plt.plot(dom, lhs)
plt.plot(dom, rhs)


## Animation involving stereographic projection

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=[10, 5])
ax = fig.add_subplot(111, projection=ccrs.Stereographic())
fig.subplots_adjust(bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02)
ax.gridlines()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

LL = f(0.0, 'lonlat')
ax.scatter(*LL.T, s=10, transform=ccrs.PlateCarree(), zorder=20)

P = f(0.0, 'geocentric')  ## 3d cartesian coordinates
Pn = normalize(P, axis=1) ## project to 2-sphere 
X, Y = Pn[:,0]/(1-Pn[:,2]), Pn[:,1]/(1-Pn[:,2])
dt = Delaunay(np.c_[X, Y])
## Plot delaunay subgraph 
from itertools import chain
g1 = ((LL[i,:], LL[j,:]) for i,j,k in dt.simplices if (i,j) in G.edges)
g2 = ((LL[i,:], LL[k,:]) for i,j,k in dt.simplices if (i,k) in G.edges)
g3 = ((LL[j,:], LL[k,:]) for i,j,k in dt.simplices if (j,k) in G.edges)
del_lines = wrap_lines(chain(g1,g2,g3), x_interval=[-180,180])
for line in del_lines:
  ax.plot(*line.T, c='orange', linewidth=0.75, zorder=10, transform=ccrs.PlateCarree())
  # ax.plot(*line.T, c='orange', linewidth=0.75, zorder=20, transform=None)


# P = np.array([stereo.transform_point(x,y, src_crs=wgs84) for x,y in LL])


# theta = np.linspace(0, 2*np.pi, 100)
# center, radius = [0.5, 0.5], 0.5
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * radius + center)
# ax.set_boundary(circle, transform=ax.transAxes)

#ax.scatter(0.50,0.50,s=50, transform=ax.transAxes)

#ax.scatter(*LL.T, transform=ax.transAxes)
# ax.scatter(0.0, 0.50, s=50, transform=ax.transAxes)

# ax.scatter((1)*(LL[:,0]/360 + 0.50), (1)*(LL[:,1]/180 + 0.50), s=50, transform=ax.transAxes)

# data_crs = ccrs.Stereographic()
# new_pt = data_crs.transform_point(0,0,src_crs=ccrs.Geodetic())
# ax.scatter(new_pt[0], new_pt[1], transform=False)
# ax.scatter((1)*(LL[:,1]/180 + 0.50), (1)*(LL[:,0]/360 + 0.50), s=50, transform=ax.transAxes)
# ax.get_xlim()

#ax.scatter(*np.c_[X, Y].T, s=50, transform=ax.transAxes)



LL = f(0.0, 'lonlat')
#ax.scatter(LL[:,0]/360, LL[:,1]/180, s=50, transform=ccrs.Stereographic())
ax.scatter(0.5, 0.5)
ax.get_xlim()


# from cartopy.crs import Projection
# Projection()
