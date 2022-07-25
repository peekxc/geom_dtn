## Load all exported functions in the package 
import numpy as np 
from geom_dtn import *

## Load default set of satellites
satellites = load_satellites()

## Make dynamic point cloud (dpc) of select satellites
I = np.array([0, 1673, 1529, 1545, 1856,   68, 1513, 2016, 2060,  750, 1120, 480, 1816,  516, 1543, 1547,  584,  299, 2070, 1984, 1056, 2257, 64, 1809, 1522, 1409, 1125,  632,   41,  551])
f, (b,e) = satellite_dpc(np.array(satellites)[I])

## Get satellite (x,y) == (longitude, latitude) information at time = 0.0
X = f(0.0, 'lonlat')

## Plot 2D projection of earth + satellite positions as points using cartopy
## See: https://scitools.org.uk/cartopy/docs/latest/reference/crs.html
import cartopy.crs as ccrs
fig, ax = plot_earth_2D(0.20)
ax.scatter(*X.T, s=25.15, c='red', transform=ccrs.PlateCarree())

## Get satellite (x,y,z) position (geocentric) at time = 0.0
X = f(0.0, 'geocentric')

## Plots regular spherical pyplot wireframe
fig, ax = plot_earth_3D(wireframe=True)
ax.scatter(*X.T)

## Plot sphere with an earth image mapped onto it (required PIL, can be slow)
fig, ax = plot_earth_3D(wireframe=False)
ax.scatter(*X.T)