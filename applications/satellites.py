import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from typing import Optional
from numpy.typing import ArrayLike
from skyfield.api import Loader, EarthSatellite
from skyfield.timelib import Time

# TLE1 = """1 44720C 19074H   22166.68212505  .00005391  00000-0  36124-3 0  1667
# 2 44720  53.0524 185.3810 0001659  56.0044 332.8715 15.06391974    19"""
with open("starlink.txt", "r") as f: 
  starlink = f.read().splitlines()

n_sat = int(len(starlink)/3)
satellites = [EarthSatellite(starlink[3*i+1], starlink[3*i+2], name=starlink[3*i]) for i in range(n_sat)]


# satellites[0].epoch.utc
ref_time = satellites[0].epoch
dist_orbit = np.array([np.linalg.norm(sat.at(ref_time).position.km) for sat in satellites])
sat_pos = np.array([sat.at(ref_time).position.km for sat in satellites])

load = Loader('~/Downloads/SkyData')
planets = load('de421.bsp')
ts   = load.timescale()
earth   = planets['earth']
re = 6378.0

hours = np.arange(0, 3, 0.01)
time = ts.utc(year=2018, month=2, day=7, hour=hours)

## Orbital plot 
fig = plt.figure(figsize=[10, 8])  # [12, 10]
ax  = fig.add_subplot(1, 1, 1, projection='3d')
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
x, y, Z = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
ax.plot_surface(x*re, y*re, Z*re, color="#00000000", edgecolor="r", linewidth=0.20)

ax.scatter(*sat_pos.T, s=0.15)

## Determine line-of-sight edge
s1 = np.array([sat.at(time[0]).position.km]).flatten()
s2 = np.array([sat.at(time[1]).position.km]).flatten()
a = np.sum((s2-s1)**2)
b = 2*np.sum([s2[i]*(s2[i]-s1[i]) for i in range(3)])
c = np.sum(s1**2) - re**2
#if (b**2 - 4 * a * c) < 0.0:


#p = (earth + satellite).at(two_hours).observe(venus).apparent()
#sunlit = p.is_behind_earth()

geocentric_pos = sat.at(time[0])
from skyfield.api import load, wgs84
lat, lon = wgs84.latlon_of(geocentric_pos)

for sat in satellites:
  x, y, z = sat.at(time).position.km
  ax.plot(x, y, z, c='r')


for TLE in orbits:
  L1, L2 = TLE.splitlines()
  sat = EarthSatellite(L1, L2)
  x, y, z = sat.at(time).position.km
  ax.plot(x, y, z, c='r')

## Plot earth wireframe 
# for x, y, z in lons: ax.plot(x, y, z, '-k')  
# for x, y, z in lats: ax.plot(x, y, z, '-k')

centers, hw = makecubelimits(ax)
plt.show()

# r_Roadster = np.sqrt((Rpos**2).sum(axis=0))
# alt_roadster = r_Roadster - re

# plt.figure()
# plt.plot(hours, r_Roadster)
# plt.plot(hours, alt_roadster)
# plt.xlabel('hours', fontsize=14)
# plt.ylabel('Geocenter radius or altitude (km)', fontsize=14)
# plt.show()





r = 0.5 
S = np.random.uniform(size=(40,3), low=-1.5*r, high=1.5*r)
def intersects_sphere(s1, s2, r=1.0, distance=False):
  # http://paulbourke.net/geometry/circlesphere/index.html#linesphere
  # "no_intersect", "tangent", "intersects"
  a = np.sum((s2-s1)**2)
  b = 2*np.sum([s1[i]*(s2[i]-s1[i]) for i in range(len(s1))])
  c = np.sum(s1**2) - r**2
  #d = np.max([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
  return(int(np.sign(b**2 - 4 * a * c)+1) if not(distance) else d)

from itertools import combinations
edges = np.array(list(combinations(range(40), 2)))
i_status = list(np.array(['blue', 'y', 'r'])[np.array([intersects_sphere(S[i,:], S[j,:], r) for i,j in edges])])
  
theta = np.linspace(0, 2*np.pi, 1000, endpoint=False)
C = r*np.c_[np.cos(theta), np.sin(theta)]
for cc, (i,j) in enumerate(edges):
  plt.plot(*S[np.ix_((i,j), (1,2))].T, c=i_status[cc], linewidth=0.15)
  #plt.plot(*S[(i,j),[0,1]].T, c=i_status[cc], linewidth=0.15)
plt.plot(*C.T, c='red', linewidth=1.5)
plt.gca().set_aspect('equal')



re = 0.5
fig = plt.figure(figsize=[10, 8])  # [12, 10]
ax  = fig.add_subplot(1, 1, 1, projection='3d')
u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j]
x, y, Z = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
ax.plot_surface(x*re, y*re, Z*re, color="#00000000", edgecolor="r", linewidth=0.20)

S = np.random.uniform(size=(40,3), low=-1.5*r, high=1.5*r)
edges = np.array(list(combinations(range(40), 2)))
i_status = list(np.array(['blue', 'y', 'r'])[np.array([intersects_sphere(S[i,:], S[j,:], re) for i,j in edges])])

for cc, (i,j) in enumerate(edges):
 if i_status[cc] == 'blue':
   plt.plot(*S[(i,j),:].T, c=i_status[cc], linewidth=0.15)
  #plt.plot(*S[(i,j),[0,1]].T, c=i_status[cc], linewidth=0.15)


## TODO: plot 2d projection of lat/lon positions of satellites + LOS graph over time
## make all calleable as functons in packageCha



sat1 = satellites[0]
sat2 = satellites[10]
c_time = time[0]

time_points = ts.linspace(sat1.epoch, sat1.epoch + timedelta(hours=5), 200)

sat1_orbit = sat1.at(time_points).position.km.T
sat2_orbit = sat2.at(time_points).position.km.T

fig = plt.figure(figsize=[10, 8])  # [12, 10]
ax  = fig.add_subplot(1, 1, 1, projection='3d')
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
x, y, Z = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
ax.plot_surface(x*re, y*re, Z*re, color="#00000000", edgecolor="r", linewidth=0.20)
ax.scatter(sat1_orbit[:,0], sat1_orbit[:,1], sat1_orbit[:,2])
ax.scatter(sat2_orbit[:,0], sat2_orbit[:,1], sat2_orbit[:,2])


from geom_dtn import load_satellites
satellites = load_satellites()

sat = satellites[0]

from skyfield.api import wgs84
lat, lon = wgs84.latlon_of(sat.at(sat.epoch))


from datetime import datetime, timedelta, time, date, timezone
# epoch_time = datetime(year=2022, month=6, day=15, hour=0, minute=0, second=0, tzinfo=timezone.utc) + timedelta(hours=12)
# threshold = timedelta(hours=1) 
# accurate_ind = np.array([abs(sat.epoch.utc_datetime()-epoch_time) <= threshold for sat in satellites])
# sats = np.array(satellites)[accurate_ind]
start_time = np.min([sat.epoch.utc_datetime() for sat in satellites])
sats = np.array(satellites)[np.random.choice(range(len(satellites)), 100).astype(int)]


time_points = ts.linspace(ts.from_datetime(start_time), ts.from_datetime(start_time + timedelta(hours=6)), 200)


# half_time = start_time + (end_time-start_time)/
# accurate_after = np.array([(end_time-sat.epoch.utc_datetime()) <= half_window for sat in satellites])
# np.sum(np.logical_and(accurate_before, accurate_after))

from scipy.interpolate import CubicSpline
from itertools import combinations
Contacts = []
tp_tt = np.array([tp-ts.from_datetime(start_time) for tp in time_points]) # time points numerical values
for i, (sat1, sat2) in enumerate(combinations(sats, 2)):
  sat1_orbit = sat1.at(time_points).position.km.T
  sat2_orbit = sat2.at(time_points).position.km.T
  D_line = np.array([dist_line2sphere(p0, p1, R=re) for (p0, p1) in zip(sat1_orbit, sat2_orbit)])
  f = CubicSpline(tp_tt, D_line)
  roots = f.roots(discontinuity=False, extrapolate='periodic')
  contacts = []
  if len(roots) > 1:
    LOS_sgn = np.sign(f(roots + 0.0001)) ## -1 := no in LOS, 1 := in LOS 
    for j in range(len(LOS_sgn)-1):
      if LOS_sgn[j] == 1.0 and LOS_sgn[j+1] == -1.0:
        contacts.append(np.append(unrank_comb2(i, len(sats)), [roots[j], roots[j+1]]))
  Contacts.append(contacts)
Contacts = np.vstack(list(filter(lambda x: len(x) > 0, Contacts)))

c_time_dt = (ts.from_datetime(start_time) + timedelta(hours=3)).utc_datetime()
st_time = ts.from_datetime(start_time)
c_edges = []
for (u,v,s,e) in Contacts:
  active_s = (st_time + s).utc_datetime() <= c_time_dt
  active_e = c_time_dt <= (st_time + e).utc_datetime()
  if (active_s and active_e):
    c_edges.append([u,v])
c_edges = np.array(c_edges).astype(int)

## Plot visibility (LOS) graph 
sat_pos = np.array([sat.at(ts.from_datetime(c_time_dt)).position.km for sat in sats])
fig, ax = plot_earth(figsize=(15,15), dpi=250)
ax.scatter(*sat_pos.T, c='red')
for (i,j) in c_edges: ax.plot(*sat_pos[(i,j),:].T, c='gray', linewidth=0.25)


## Make a multigraph w/ networkx 
import networkx as nx
G = nx.MultiGraph()
G.add_nodes_from(range(len(sats)))
keys = G.add_edges_from([(int(i),int(j),{ "start": s, "end": e }) for (i,j,s,e) in Contacts])

CP = Contacts # contact plan

# Contact Conditional Dijkstra’s Algorithm
def CCD():
  return(0)

# Conditional Contact Review Procedure
def CCRP(CP):
  return(0)

# Vertex selection procedure
def VSP():
  return(0)

# global w 
# w = 0.0
# nx.shortest_path(G, source=0, target=6, weight=lambda i,j,ea: ea['start'])

# (ts.from_datetime(start_time) + Contacts[0,2]).utc_strftime()
# (ts.from_datetime(start_time) + Contacts[0,3]).utc_strftime()
# time_points[0].utc_strftime()
# time_points[-1].utc_strftime()

# ts.tt(time_points[0].tt).utc_strftime()

from scipy.interpolate import interp1d 

# f = interp1d(list(range(len(D_line))), D_line, kind='cubic')


# t_edges = Contacts[np.logical_and(Contacts[:,2] >= t_begin, Contacts[:,3] > t_begin),:2].astype(int)

# import dynetx as dn
# G = dn.DynGraph(edge_removal=True)
# G.add_nodes_from(range(len(sats)))
# for u,v,t,e in Contacts:
#   g.add_interaction(u=int(u), v=int(v), t=t, e=e)




# x0, y0 = s1
# sx, sy = (s2 - s1)
# xc, yc = 0.50, 0.50 # sphere center
# t = (sx*(xc-x0)+sy*(yc-y0))/(sx**2 + sy**2)
# q = t * np.array([sx, sy]) + s1
# np.linalg.norm(q - np.array([xc, yc])) - R

# x0, y0 = s1 
# sx, sy = (s1 - s2)
# xc, yc = 0.50, 0.50 # sphere center
# t = (sx*(xc-x0)+sy*(yc-y0))/(sx**2 + sy**2)
# q = t * np.array([sx, sy]) + s1
# np.linalg.norm(q - np.array([xc, yc])) - R

# d  = np.sqrt((x0+sx*t-xc)**2 + (y0+sy*t-yc)**2) - 0.50
# q = np.array([xc + ((x0+sx*t-xc)*0.50)/(0.50 + d), ((y0+sy*t-yc)*0.50)/(0.50 + d)])

# dist_line2sphere(s1, s2, R=0.5, sc=np.array([0.5, 0.5]))

# plt.plot(*(C + np.array([xc, yc])).T, c='red', linewidth=1.5)
# plt.gca().set_aspect('equal')
# plt.plot(*np.vstack((s1,s2)).T, c='red')
# # p = s1 + t*(s1-s1)
# # plt.scatter(p[0], p[1], c='green')
# plt.scatter(q[0], q[1], c='purple')

from geom_dtn import testing_satellite




import PIL
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import importlib.resources as pkg_resources
from geom_dtn import data as package_data_mod
# starlink = pkg_resources.read_text(package_data_mod, 'starlink.txt')

# load bluemarble with PIL
# blue marble from: https://visibleearth.nasa.gov/images/73751/july-blue-marble-next-generation-w-topography-and-bathymetry/73753l



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from itertools import chain

def draw_map(m, scale=0.2):
  # draw a shaded-relief image
  m.shadedrelief(scale=scale)
  
  # lats and longs are returned as a dictionary
  lats = m.drawparallels(np.linspace(-90, 90, 13))
  lons = m.drawmeridians(np.linspace(-180, 180, 13))

  # keys contain the plt.Line2D instances
  lat_lines = chain(*(tup[1][0] for tup in lats.items()))
  lon_lines = chain(*(tup[1][0] for tup in lons.items()))
  all_lines = chain(lat_lines, lon_lines)
  
  # cycle through these lines and set the desired style
  for line in all_lines: line.set(linestyle='-', alpha=0.3, color='w')

fig = plt.figure(figsize=(8, 8), edgecolor='w')
m = Basemap(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
draw_map(m)
m.scatter(lon.degrees, lat.degrees, latlon=True, c='r', s=5.15)


km_to_latlon


sat.at(sat.epoch).position.km


pos = wgs84.geographic_position_of(sat.at(sat.epoch))
pos.latitude.degrees, pos.longitude.degrees




import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
from geom_dtn import rank_comb2


X = np.random.uniform(size=(10,2))
plt.scatter(*X.T)


D = squareform(pdist(X))

from itertools import combinations
n = X.shape[0]
ptolemy = np.array([D[a,c]*D[b,d] - (D[a,b]*D[c,d] + D[b,c]*D[a,d]) for a,b,c,d in combinations(range(n), 4)])


#theta = np.array([0, np.pi/2, np.pi+0.55, 3/2*np.pi])


theta = np.linspace(0, 2*np.pi, 350)
circle = np.c_[np.cos(theta), np.sin(theta)]

theta = np.sort(np.random.uniform(0, 2*np.pi, 4))
c_xy = np.c_[np.cos(theta), np.sin(theta)]

plt.plot(*circle.T, c='blue')
plt.scatter(*c_xy.T, c='red')
plt.gca().set_aspect('equal')

D = squareform(pdist(c_xy))
a,b,c,d = 0,1,2,3
D[a,c]*D[b,d] - (D[a,b]*D[c,d] + D[b,c]*D[a,d])

from scipy.spatial import ConvexHull
chull = ConvexHull(c_xy)
a,b,c,d = chull.vertices

A = np.c_[
  c_xy[(a,b,c),0]-c_xy[d,0],
  c_xy[(a,b,c),1]-c_xy[d,1],
  np.sum(c_xy[(a,b,c),:]**2 - c_xy[d,:]**2, axis=1)
]
np.linalg.det(A)


## TODO: track either det or Ptolemy's relationship for all choose(n, 4) subsets 
## throughout all time. The roots of the polynomial interpolating these values 
## should be when an edge flip occurs. One should be able to reduce the size of this 
## subset list be detecting all the 4-cliques in the edge-flattened multi-graph repr. 
from geom_dtn import *
import numpy as np 
satellites = load_satellites()
I = np.array([0, 1673, 1529, 1545, 1856,   68, 1513, 2016, 2060,  750, 1120, 480, 1816,  516, 1543, 1547,  584,  299, 2070, 1984, 1056, 2257, 64, 1809, 1522, 1409, 1125,  632,   41,  551])
#satellites = np.array(satellites)
f = satellite_dpc(satellites)
#I, E = maxmin(f(0.0, 'latlon'), 30)
f = satellite_dpc(np.array(satellites)[I])

import cartopy.crs as ccrs
from scipy.spatial import Delaunay

lonlat = f(0.0, 'latlon')[:,[1,0]]
#dt = Delaunay(f(0.0, 'geocentric'))
dt = Delaunay(lonlat)
fig, ax = plot_earth_2D(0.20)
ax.scatter(*lonlat.T, transform=ccrs.PlateCarree(), s=0.15, c='red')
for (i,j,k) in dt.simplices:
  ax.plot(*lonlat[(i,j,k,i),:].T, c='red', linewidth=0.15)
# ax.scatter(*lonlat[I,:].T, transform=ccrs.PlateCarree(), s=3.35, c='green')

plot_earth()

from datetime import datetime, timedelta, time, date, timezone
from geom_dtn import data as package_data_mod
load = Loader(package_data_mod.__path__._path[0])
ts = load.timescale()
s_time = np.min([sat.epoch.utc_datetime() for sat in satellites])
e_time = np.max([sat.epoch.utc_datetime() for sat in satellites])
tps = ts.linspace(ts.from_datetime(s_time), ts.from_datetime(e_time), 2000)

from scipy.spatial import ConvexHull
from itertools import combinations
import time
S = dt.simplices
for (i,j,k,l) in combinations(range(len(I)), 4):
  i,j,k,l = I[[i,j,k,l]]
  sat_i = satellites[i].at(tps).position.km
  sat_j = satellites[j].at(tps).position.km
  sat_k = satellites[k].at(tps).position.km
  sat_l = satellites[l].at(tps).position.km
  P = np.vstack([sat_i[:,0],sat_j[:,0],sat_k[:,0],sat_l[:,0]])
  i,j,k,l = np.array([i,j,k,l])[ConvexHull(P).vertices]
  sat_i = satellites[i].at(tps).position.km
  sat_j = satellites[j].at(tps).position.km
  sat_k = satellites[k].at(tps).position.km
  sat_l = satellites[l].at(tps).position.km
  dist_a = np.sqrt(np.sum((sat_i - sat_k)**2, axis=0))*np.sqrt(np.sum((sat_j - sat_l)**2, axis=0))
  dist_b = np.sqrt(np.sum((sat_i - sat_j)**2, axis=0))*np.sqrt(np.sum((sat_k - sat_l)**2, axis=0)) + np.sqrt(np.sum((sat_j - sat_k)**2, axis=0))*np.sqrt(np.sum((sat_i - sat_l)**2, axis=0))
  roots = CubicSpline(tps - tps[0], dist_a-dist_b).roots(extrapolate='periodic')
  roots = np.array(list(filter(lambda r: r <= (tps[-1]-tps[0]), roots)))
  if len(roots) > 0:
    plt.plot(tps - tps[0], dist_a-dist_b)
    break
  else:
    time.sleep(0.50)
    print((i,j,k,l))

from scipy.interpolate import CubicSpline


plt.scatter(*lonlat.T, s=10.45, c='red')
plt.gca().set_aspect('equal')

f = satellite_dpc(np.array(satellites)[I])