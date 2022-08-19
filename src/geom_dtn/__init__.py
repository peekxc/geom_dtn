from __future__ import division 
from .routing import load_satellites 

from typing import * 
from numpy.typing import ArrayLike
from skyfield.api import Loader, EarthSatellite, load, wgs84 
from skyfield.timelib import Time
from scipy.interpolate import CubicSpline
from itertools import combinations
from datetime import datetime
from array import array

import sys
import os 
import importlib.resources as pkg_resources
import numpy as np
import matplotlib.pyplot as plt



def partition_envelope(f: Callable, threshold: float, interval: Tuple = (0, 1), lower: bool = False):
  """
  Partitions the domain of a real-valued function 'f' into intervals by evaluating the pointwise maximum of 'f' and the constant function g(x) = threshold. 
  The name 'envelope' is because this can be seen as intersecting the upper-envelope of 'f' with 'g'

  Parameters: 
    f := Callable that supports f.operator(), f.derivative(), and f.roots() (such as the Splines from scipy)
    threshold := cutoff-threshold
    interval := interval to evalulate f over 
    lower := whether to evaluate the lower envelope instead of the upper 

  Return: 
    intervals := (m x 3) nd.array giving the intervals that partition the corresponding envelope
  
  Each row (b,e,s) of intervals has the form: 
    b := beginning of interval
    e := ending of interval
    s := 1 if in the envelope, 0 otherwise 

  """
  assert isinstance(interval, Tuple) and len(interval) == 2

  ## Partition a curve into intervals at some threshold
  in_interval = lambda x: x >= interval[0] and x <= interval[1]
  crossings = np.fromiter(filter(in_interval, f.solve(threshold)), float)

  ## Determine the partitioning of the upper-envelope (1 indicates above threshold)
  intervals = []
  if len(crossings) == 0:
    is_above = f(0.50).item() >= threshold
    intervals.append((0.0, 1.0, 1 if is_above else 0))
  else:
    if crossings[-1] != 1.0: 
      crossings = np.append(crossings, 1.0)
    b = 0.0
    df = f.derivative(1)
    df2 = f.derivative(2)
    for c in crossings:
      grad_sign = np.sign(df(c))
      if grad_sign == -1:
        intervals.append((b, c, 1))
      elif grad_sign == 1:
        intervals.append((b, c, 0))
      else: 
        accel_sign = np.sign(df2(c).item())
        if accel_sign > 0: # concave 
          intervals.append((b, c, 1))
        elif accel_sign < 0: 
          intervals.append((b, c, 0))
        else: 
          raise ValueError("Unable to detect")
      b = c
  
  ### Finish up and return 
  intervals = np.array(intervals)
  if lower: 
    intervals[:,2] = 1 - intervals[:,2]
  return(intervals)

## From: https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it, count=None, prefix="", size=60, out=sys.stdout): # Python3.6+
  count = len(it) if count == None else count 
  def show(j):
    x = int(size*j/count)
    print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
  show(0)
  for i, item in enumerate(it):
    yield item
    show(i+1)
  print("\n", flush=True, file=out)

def plot_earth_3D(wireframe=True, scale=1/5, surface_kwargs = None, **kwargs):
  # from: https://stackoverflow.com/questions/30269099/creating-a-rotatable-3d-earth
  assert scale <= 1.0, "scale must be between: 0 < scale <= 1.0"
  re = 6378.0
  fig = plt.figure(**kwargs)
  ax = fig.add_subplot(111, projection='3d')
  if wireframe:
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    x, y, Z = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax.plot_surface(x*re, y*re, Z*re, color="#00000000", edgecolor="blue", linewidth=0.20, zorder=10)
  else: 
    import PIL
    r = 1/scale
    from geom_dtn import data as package_data_mod
    bm = PIL.Image.open(os.path.join(package_data_mod.__path__._path[0], 'blue_marble.jpg'))
    bm = np.array(bm.resize([int(d/r) for d in bm.size]))/256
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 
    x = np.outer(np.cos(lons), np.cos(lats)).T
    y = np.outer(np.sin(lons), np.cos(lats)).T
    z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    ax.plot_surface(x*re, y*re, z*re, rstride=4, cstride=4, facecolors=bm, zorder=10)
  return(fig, ax)

def edges_from_triangles(triangles: ArrayLike, nv: int):
  ER = np.array([[rank_C2(*t[[0,1]], n=nv), rank_C2(*t[[0,2]], n=nv), rank_C2(*t[[1,2]], n=nv)] for t in triangles])
  ER = np.unique(ER.flatten())
  E = np.array([unrank_C2(r, n=nv) for r in ER])
  return(E)
def plot_earth_2D(scale=0.20, fig_kwargs: Optional[Dict] = {}):
  #from mpl_toolkits.basemap import Basemap
  import cartopy.crs as ccrs
  from itertools import chain
  assert scale > 0.0 and scale <= 1.0, "Scale must be in interval (0, 1]"

  ## Supplant default figure + cartopy options w/ supplied ones
  def_fig = dict(figsize=(10, 5))
  fig = plt.figure(**(def_fig | fig_kwargs))
  ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
  ax.set_global()
  ax.stock_img()

  # def_bm = dict(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
  # m = Basemap(**(def_bm | bm_kwargs))

  # ## Draw the basemap + latitude and longitude lines 
  # m.shadedrelief(scale=scale)  # donwgrade image resolution; use 0.0  
  # lats = m.drawparallels(np.linspace(-90, 90, 13))
  # lons = m.drawmeridians(np.linspace(-180, 180, 13))
  # lat_lines = chain(*(tup[1][0] for tup in lats.items()))
  # lon_lines = chain(*(tup[1][0] for tup in lons.items()))
  # all_lines = chain(lat_lines, lon_lines)
  
  # cycle through these lines and set the desired style
  # for line in all_lines: line.set(linestyle='-', alpha=0.3, color='w')
  return(fig, ax)

def rank_comb2(i, j, n):
	i, j = (j, i) if j < i else (i, j)
	return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_comb2(x, n):
	i = (n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
	j = x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
	return(int(i), int(j))

def maxmin(X: ArrayLike, n: int, seed: int = 0):
  import numpy as np
  from array import array
  from scipy.spatial.distance import pdist, cdist, squareform
  E, I = array('d'), array('I')
  E.append(float('inf'))
  I.append(seed)
  while len(I) < n:
    lm_dist = np.min(cdist(X[I,:], X), axis=0)
    I.append(np.setdiff1d(np.argsort(-lm_dist), I, assume_unique=True)[0])
    E.append(lm_dist[I[-1]])
  return((np.asarray(I), np.asarray(E)))

def dist_line2sphere(s1: ArrayLike, s2: ArrayLike, R: float = 1.0, sc = Optional[ArrayLike]):
  """
  Returns the minimum distance between any point on a line going through points s1, s2 
  and the sphere with radius 'R' centered at 'sc'. 
  """
  s1, s2 = np.asarray(s1), np.asarray(s2)
  sc = np.zeros(shape=s1.shape)
  s = s2 - s1
  if np.all(s == 0):
    return(np.linalg.norm(s1 - sc) - R)
  t = np.sum(s*(sc - s1))/np.linalg.norm(s)**2
  q = t * s + s1
  return(np.linalg.norm(q - sc) - R)

def datetime_to_time(time: datetime):
  from geom_dtn import data as package_data_mod
  load = Loader(package_data_mod.__path__._path[0])
  ts = load.timescale()
  return(ts.from_datetime(time, ))

def sat_contact_plan(satellites: List, s_time: Optional[datetime] = None, e_time: Optional[datetime] = None, resolution: int = 20, ground_stations: Optional[List] = None, progress: bool = False):
  '''
  Generate a contact plan matrix (m x 4) of m contacts whose rows have the format: 

  (i, j, s, e)

  whose contacts [s, e) correspond to line-of-sight between satellites i and j.

  i: int = integer index in [0, n) indicating the first satellite
  j: int = integer index in [0, n) indicating the second satellite
  s: datetime = 


  '''
  if s_time is None: 
    s_time = np.min([sat.epoch.utc_datetime() for sat in satellites])
  if e_time is None: 
    e_time = np.max([sat.epoch.utc_datetime() for sat in satellites])
  assert type(s_time) == datetime and type(e_time) == datetime
  
  ## Load timescale and create discretely-interpolated time events
  from geom_dtn import data as package_data_mod
  load = Loader(package_data_mod.__path__._path[0])
  ts = load.timescale()
  time_points = ts.linspace(ts.from_datetime(s_time), ts.from_datetime(e_time), resolution)
  tp_tt = np.array([tp-ts.from_datetime(s_time) for tp in time_points]) # time points numerical values
  tp_tt = (tp_tt - np.min(tp_tt))/(np.max(tp_tt) - np.min(tp_tt))

  ## Generate satellite contacts 
  sat_contacts = array('f')
  if not(satellites is None):
    n = len(satellites)
    N = n*(n-1)/2
    sat_pair_gen = progressbar(enumerate(combinations(satellites, 2)), count=N) if progress else enumerate(combinations(satellites, 2))
    for i, (sat1, sat2) in sat_pair_gen:
      sat1_orbit = sat1.at(time_points).position.km.T
      sat2_orbit = sat2.at(time_points).position.km.T
      D_line = np.array([dist_line2sphere(p0, p1, R=6378.0) for (p0, p1) in zip(sat1_orbit, sat2_orbit)])
      f = CubicSpline(tp_tt, D_line)
      contact_intervals = partition_envelope(f, threshold=0.0) # upper envelope of f 
      for (b,e,s) in contact_intervals:
        if s == 1:
          sat_contacts.extend(np.append(unrank_comb2(i, len(satellites)), [b, e]))
      # roots = f.roots(discontinuity=False, extrapolate='periodic')
      # if len(roots) > 0:
      #   LOS_sgn = np.sign(f(roots + 0.0001)) ## -1 := no in LOS, 1 := in LOS 
      #   for j in range(len(LOS_sgn)-1):
      #     if LOS_sgn[j] == 1.0 and LOS_sgn[j+1] == -1.0:
      #       sat_contacts.extend(np.append(unrank_comb2(i, len(satellites)), [roots[j], roots[j+1]]))
      # print(i)

  ground_contacts = array('f')
  if not(ground_stations is None):
    for i, sat in enumerate(satellites):
      sat_pos = sat.at(time_points).position.km.T
      for j, gs in enumerate(ground_stations):
        gs_pos = np.reshape(np.tile(gs, sat_pos.shape[0]), (sat_pos.shape[0],3))
        D_line = np.array([dist_line2sphere(p0, p1, R=6378.0) for (p0, p1) in zip(sat_pos, gs_pos)])
        f = CubicSpline(tp_tt, D_line)
        contact_intervals = partition_envelope(f, threshold=0.0) # upper envelope of f 
        for (b,e,s) in contact_intervals:
          if s == 1:
            ground_contacts.extend([i, j, b, e])
        # roots = f.roots(discontinuity=False, extrapolate='periodic')
        # if len(roots) > 1:
        #   LOS_sgn = np.sign(f(roots + 0.0001)) ## -1 := no in LOS, 1 := in LOS 
        #   for j in range(len(LOS_sgn)-1):
        #     if LOS_sgn[j] == 1.0 and LOS_sgn[j+1] <= 0.0:
        #       ground_contacts.extend([i,j,roots[j], roots[j+1]])
  
  n_sc, n_gc = int(len(sat_contacts)/4), int(len(ground_contacts)/4)
  s_cp = np.reshape(np.asarray(sat_contacts).flatten(), (n_sc, 4)) if n_sc > 0 else np.empty(shape=(0,4))
  g_cp = np.reshape(np.asarray(ground_contacts).flatten(), (n_gc, 4)) if n_gc > 0 else np.empty(shape=(0,4))

  if n_sc > 0 and n_gc == 0:
    return(s_cp, (s_time, e_time))
  elif n_gc > 0 and n_sc == 0:
    return(g_cp, (s_time, e_time))
  return(s_cp, g_cp, (s_time, e_time))

def satellite_dpc(satellites, s_time=None, e_time=None):
  '''
  Returns a closure parameterized by time whose evaluation yields the coordinates of a given dynamic point cloud.

  Returns (f, (b, e)), where: 
    - f := function that takes time 't' and returns cartesian coordinates of the satellites at time 't' 
    - b := time range where 't' is begins, as a datetime object
    - e := time range where 't' is ends, as a datetime object

  The resulting 'f' is overloaded to handle multiple inputs. It's signature is: 
  
  f(t: Union[float, Time, datetime], cformat: str)

  where: 
    t: float => time is given in interval between [0, 1]
    t: Time => 
    t: datetime => time is given as a standard UTC datetime object
    cformat == 'geocentric' => 
    cformat == 'latlon' => WGS84 latitude/longitude
  '''
  from datetime import datetime, timedelta, time, date, timezone
  s_time = np.min([sat.epoch.utc_datetime() for sat in satellites]) if s_time is None else s_time
  e_time = np.max([sat.epoch.utc_datetime() for sat in satellites]) if e_time is None else e_time
  
  from geom_dtn import data as package_data_mod
  load = Loader(package_data_mod.__path__._path[0])
  ts = load.timescale()

  def _gen_point_cloud(time: Union[float, Time, datetime], cformat=['geocentric', 'latlon', 'lonlat']):
    c_time = s_time + time*(e_time-s_time) if isinstance(time, float) else time
    c_time = ts.from_datetime(c_time) if isinstance(c_time, datetime) else c_time
    # print(c_time)
    assert isinstance(c_time, Time)
    if cformat == 'geocentric' or cformat == ['geocentric', 'latlon', 'lonlat']:
      xyz = np.array([sat.at(c_time).position.km for sat in satellites])
      return(xyz)
    elif cformat == 'latlon' or cformat == 'lonlat':
      def latlon(geo_icrs):
        pos = wgs84.geographic_position_of(geo_icrs)
        return(np.array([pos.latitude.degrees, pos.longitude.degrees]))
      LL = np.array([latlon(sat.at(c_time)) for sat in satellites])
      return(LL if cformat == 'latlon' else np.c_[LL[:,1], LL[:,0]])
    else: 
      raise ValueError("invalid input")
  return(_gen_point_cloud, (s_time, e_time))

def normalize(a, axis=-1, order=2):
  l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
  l2[l2==0] = 1
  return a / np.expand_dims(l2, axis)

def kinetic_events_del2D(f: Iterable):
  '''
  Given an iterable 'f' of pairwise distances (e.g. the result of pdist(*)), 
  '''
  return(0)

def wrap_lines(Lines: Iterable, x_interval = [-1, 1]):
  """
  Given an iterable of lines defined by point pairs (p0, p1), returns an array of lines 
  giving the shortest distance lines wrapped around the cylinder [x_interval] x [...]
  """
  width = np.diff(x_interval).item()
  r = []
  for (p0, p1) in Lines: 
    if abs(p0[0] - p1[0]) <= width/2:
      r.append(np.vstack((p0, p1)))
    else:
      p0, p1 = (p0, p1) if p0[0] < p1[0] else (p1, p0)
      p1_backward = np.array([p1[0] - width, p1[1]])
      p0_forward = np.array([p0[0] + width, p0[1]])
      r.append(np.vstack((p0, p1_backward)))
      r.append(np.vstack((p1, p0_forward)))
  return(r)
  
def edges_at(t: float, sat_CP: ArrayLike):
  for (i,j,s,e) in sat_CP:
    if s <= t and t <= e:
      yield (int(i), int(j))
  




# from https://stackoverflow.com/questions/42464334/find-the-intersection-of-two-curves-given-by-x-y-data-with-high-precision-in
def interpolated_intercepts(x, y1, y2):
  """Find the intercepts of two curves, given by the same x data"""
  def intercept(point1, point2, point3, point4):
    """find the intersection between two lines
    the first line is defined by the line between point1 and point2
    the first line is defined by the line between point3 and point4
    each point is an (x,y) tuple.

    So, for example, you can find the intersection between
    intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

    Returns: the intercept, in (x,y) format
    """    
    def line(p1, p2):
      A, B, C = (p1[1] - p2[1]), (p2[0] - p1[0]), (p1[0]*p2[1] - p2[0]*p1[1])
      return A, B, -C
    def intersection(L1, L2):
      D  = L1[0] * L2[1] - L1[1] * L2[0]
      Dx = L1[2] * L2[1] - L1[1] * L2[2]
      Dy = L1[0] * L2[2] - L1[2] * L2[0]
      x = Dx / D
      y = Dy / D
      return x,y
    L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
    L2 = line([point3[0],point3[1]], [point4[0],point4[1]])
    R = intersection(L1, L2)
    return R
  idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
  xcs, ycs = [], []
  for idx in idxs:
    xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
    xcs.append(xc)
    ycs.append(yc)
  return np.array(xcs), np.array(ycs)

from math import comb

def rank_C2(i: int, j: int, n: int):
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(x: int, n: int):
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

def unrank_comb(r: int, k: int, n: int):
  result = np.zeros(k, dtype=int)
  x = 1
  for i in range(1, k+1):
    while(r >= comb(n-x, k-i)):
      r -= comb(n-x, k-i)
      x += 1
    result[i-1] = (x - 1)
    x += 1
  return(result)

def unrank_combs(R: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([unrank_C2(r, n) for r in R], dtype=int))
  else: 
    return(np.array([unrank_comb(r, k, n) for r in R], dtype=int))

def rank_comb(c: Tuple, k: int, n: int):
  c = np.array(c, dtype=int)
  #index = np.sum(comb((n-1)-c, np.flip(range(1, k+1))))
  index = np.sum([comb(cc, kk) for cc,kk in zip((n-1)-c, np.flip(range(1, k+1)))])
  return(int((comb(n, k)-1) - int(index)))

def rank_combs(C: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([rank_C2(c[0], c[1], n) for c in C], dtype=int))
  else: 
    return(np.array([rank_comb(c, k, n) for c in C], dtype=int))