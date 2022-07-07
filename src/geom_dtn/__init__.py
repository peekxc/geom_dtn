from .routing import load_satellites 

from typing import * 
from numpy.typing import ArrayLike
from skyfield.api import Loader, EarthSatellite, load, wgs84 
from skyfield.timelib import Time
from scipy.interpolate import CubicSpline
from itertools import combinations
from datetime import datetime
from array import array

import os 
import importlib.resources as pkg_resources
import numpy as np
import matplotlib.pyplot as plt

def plot_earth(wireframe=True, scale=1/5, surface_kwargs = None, **kwargs):
  # from: https://stackoverflow.com/questions/30269099/creating-a-rotatable-3d-earth
  assert scale <= 1.0, "scale must be between: 0 < scale <= 1.0"
  re = 6378.0
  fig = plt.figure(**kwargs)
  ax = fig.add_subplot(111, projection='3d')
  if wireframe:
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    x, y, Z = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax.plot_surface(x*re, y*re, Z*re, color="#00000000", edgecolor="blue", linewidth=0.20)
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
    ax.plot_surface(x*re, y*re, z*re, rstride=4, cstride=4, facecolors=bm)
  return(fig, ax)

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

def sat_contact_plan(satellites: List, s_time: Optional[datetime] = None, e_time: Optional[datetime] = None, resolution: int = 200, ground_stations: Optional[List] = None):
  '''
  Generate a contact plan matrix (m x 4) of m contacts whose rows have the format: 

  (i, j, s, e)

  whose contacts [s, e) correspond to line-of-sight between satellites i and j.


  '''
  s_time = np.min([sat.epoch.utc_datetime() for sat in satellites])
  e_time = np.max([sat.epoch.utc_datetime() for sat in satellites])
  
  from geom_dtn import data as package_data_mod
  load = Loader(package_data_mod.__path__._path[0])
  ts = load.timescale()
  # planets = load('de421.bsp')

  Contacts = array('f')
  time_points = ts.linspace(ts.from_datetime(s_time), ts.from_datetime(e_time), resolution)
  tp_tt = np.array([tp-ts.from_datetime(s_time) for tp in time_points]) # time points numerical values
  for i, (sat1, sat2) in enumerate(combinations(satellites, 2)):
    sat1_orbit = sat1.at(time_points).position.km.T
    sat2_orbit = sat2.at(time_points).position.km.T
    D_line = np.array([dist_line2sphere(p0, p1, R=6378.0) for (p0, p1) in zip(sat1_orbit, sat2_orbit)])
    f = CubicSpline(tp_tt, D_line)
    roots = f.roots(discontinuity=False, extrapolate='periodic')
    #contacts = []
    if len(roots) > 1:
      LOS_sgn = np.sign(f(roots + 0.0001)) ## -1 := no in LOS, 1 := in LOS 
      for j in range(len(LOS_sgn)-1):
        if LOS_sgn[j] == 1.0 and LOS_sgn[j+1] == -1.0:
          Contacts.extend(np.append(unrank_comb2(i, len(satellites)), [roots[j], roots[j+1]]))
    #Contacts.append(contacts)

  ground_contacts = array('f')
  for i, sat in enumerate(satellites):
    sat_pos = sat.at(time_points).position.km.T
    for j, gs in enumerate(ground_stations):
      gs_pos = np.reshape(np.tile(gs, sat_pos.shape[0]), (sat_pos.shape[0],3))
      D_line = np.array([dist_line2sphere(p0, p1, R=6378.0) for (p0, p1) in zip(sat_pos, gs_pos)])
      f = CubicSpline(tp_tt, D_line)
      roots = f.roots(discontinuity=False, extrapolate='periodic')
      if len(roots) > 1:
        LOS_sgn = np.sign(f(roots + 0.0001)) ## -1 := no in LOS, 1 := in LOS 
        for j in range(len(LOS_sgn)-1):
          if LOS_sgn[j] == 1.0 and LOS_sgn[j+1] <= 0.0:
            ground_contacts.extend([i,j,roots[j], roots[j+1]])
  contact_plan = np.reshape(np.asarray(Contacts).flatten(), (int(len(Contacts)/4), 4))
  ground_contact_plan = np.reshape(np.asarray(ground_contacts).flatten(), (int(len(ground_contacts)/4), 4))
  return(contact_plan, ground_contact_plan)

def satellite_dpc(satellites):
  '''
  Returns a closure parameterized by time whose evaluation yields the coordinates of a given dynamic point cloud.

  Returns: 
    - f()
    - [b,e) := time range
  '''
  from datetime import datetime, timedelta, time, date, timezone
  s_time = np.min([sat.epoch.utc_datetime() for sat in satellites])
  e_time = np.max([sat.epoch.utc_datetime() for sat in satellites])
  
  from geom_dtn import data as package_data_mod
  load = Loader(package_data_mod.__path__._path[0])
  ts = load.timescale()

  def _gen_point_cloud(time: Union[float, Time, datetime], coords=['geocentric', 'latlon']):
    c_time = s_time + timedelta(microseconds=time*(e_time-s_time).microseconds) if isinstance(time, float) else time
    c_time = ts.from_datetime(c_time) if isinstance(c_time, datetime) else c_time
    assert isinstance(c_time, Time)
    coords = 'geocentric' if coords == ['geocentric', 'latlon'] else coords
    if coords == 'geocentric':
      xyz = np.array([sat.at(c_time).position.km for sat in satellites])
      return(xyz)
    else:
      def latlon(geo_icrs):
        pos = wgs84.geographic_position_of(geo_icrs)
        return(np.array([pos.latitude.degrees, pos.longitude.degrees]))
      LL = np.array([latlon(sat.at(c_time)) for sat in satellites])
      return(LL)
  return(_gen_point_cloud)


def kinetic_events_del2D(f: Iterable):
  '''
  Given an iterable 'f' of pairwise distances (e.g. the result of pdist(*)), 
  '''
  return(0)


