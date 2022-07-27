## Load all exported functions in the package 
import numpy as np 
from geom_dtn import *
from datetime import datetime, timedelta, time, date, timezone

## Load default set of satellites
satellites = load_satellites()




## Make dynamic point cloud (dpc) of select satellites (S)
I = np.array([0, 1673, 1529, 1545, 1856,   68, 1513, 2016, 2060,  750, 1120, 480, 1816,  516, 1543, 1547,  584,  299, 2070, 1984, 1056, 2257, 64, 1809, 1522, 1409, 1125,  632,   41,  551])
S = np.array(satellites)[I]

st = min([s.epoch.utc_datetime() for s in S])
et = st + timedelta(minutes=182)

f, (b,e) = satellite_dpc(S, s_time = st, e_time = et)

## Generate a contact plan. This can be very slow. 
## TODO: redo this function completely
sat_CP, (st, et) = sat_contact_plan(S, s_time = st, e_time = et, resolution=2000, progress=True)

## 15 seconds === 15k ms / 250 == 60 ms between each frame
time_points = np.linspace(np.min(sat_CP[:,2]), np.max(sat_CP[:,3]), 250)

from scipy.spatial import Delaunay
P = f(0.0, 'geocentric')
Pn = normalize(P, axis=1)
X, Y = Pn[:,0]/(1-Pn[:,2]), Pn[:,1]/(1-Pn[:,2])
dt = Delaunay(np.c_[X, Y])

## Draw Earth + satellites as points
fig, ax = plot_earth_2D(0.20)
ax.scatter(*LL.T, s=25.15, c='red', transform=ccrs.PlateCarree())

## Draw Delaunay graph
# for (i,j,k) in dt.simplices:
#   ax.plot(*LL[(i,j),:].T, c='red', linewidth=0.85)
#   ax.plot(*LL[(i,k),:].T, c='red', linewidth=0.85)
#   ax.plot(*LL[(j,k),:].T, c='red', linewidth=0.85)

## Draw visibility graph
# for i,j,s,e in sat_CP:
#   if t0 >= s and t0 < e:
#     ax.plot(*LL[np.array([i,j]).astype(int),:].T, c='gray', linewidth=1.85, transform=ccrs.PlateCarree())
# elif ind == 2:
#   x, y = L[i,:], L[j,:]
#   y2 = y + 0.5*(y-x)
#   x2 = x - 0.5*(y-x)
#   ax.plot(*np.hstack((x,x2)).T, c='gray', linewidth=1.85, transform=ccrs.PlateCarree())
#   ax.plot(*np.hstack((y,y2)).T, c='gray', linewidth=1.85, transform=ccrs.PlateCarree())

# LL = f(t, 'lonlat')
# ax.scatter(*LL.T, s=25.15, c='red', transform=ccrs.PlateCarree())

def LOS_edges(t, LL, sat_CP):
  Lines = []
  for i,j,s,e in sat_CP:
    if t >= s and t < e:
      x, y = LL[int(i),:], LL[int(j),:]
      d1 = np.linalg.norm(x - y) # base 
      d2 = np.linalg.norm(np.array([x[0] + 360, x[1]]) - y) # wrap left-right
      ind = np.argmin([d1,d2])
      if ind == 0: 
        Lines.append(np.vstack((x,y)))
      else:
        x2 = np.array([y[0] - 360, y[1]]) #y2 = np.array([x[0] + 360, x[1]])
        Lines.append(np.vstack((x,x2)))
  return(Lines)
#ax.plot(*LL[np.array([i,j]).astype(int),:].T, c='gray', linewidth=1.85, transform=ccrs.PlateCarree())        
#ax.plot(*np.vstack((x,x2)).T, c='blue', linewidth=1.85, transform=ccrs.PlateCarree())
#ax.plot(*np.vstack((y,y2)).T, c='blue', linewidth=1.85, transform=ccrs.PlateCarree())

fig, ax = plot_earth_2D(0.20)
points = ax.scatter([], [], color="green", zorder=4)
lines, = ax.plot([], [], color="crimson", zorder=4)

def animate(t):
  # ax.clear()
  LL = f(t, 'lonlat')
  Lines = LOS_edges(t, LL, sat_CP)
  for l in Lines:
    ax.plot(*l.T, c='blue')

  ## Update satellite positions
  points.set_offsets(LL)

from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, animate, frames=time_points[:10], interval=60, blit=False)
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('delaunay_sats.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()

t = time_points[10]
np.sum(sat_CP[:,2] <= t)
np.sum(t < sat_CP[:,3])
np.sum(np.logical_and(sat_CP[:,2] <= t, t < sat_CP[:,3]))


LL = f(t0, 'lonlat')
fig, ax = plot_earth_2D(0.20)
pts = ax.scatter(*LL.T, s=25.15, c='red', transform=ccrs.PlateCarree())
