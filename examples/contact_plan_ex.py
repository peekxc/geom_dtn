## Load all exported functions in the package 
import numpy as np 
from geom_dtn import *

## Load default set of satellites
satellites = load_satellites()

## Make dynamic point cloud (dpc) of select satellites (S)
I = np.array([0, 1673, 1529, 1545, 1856,   68, 1513, 2016, 2060,  750, 1120, 480, 1816,  516, 1543, 1547,  584,  299, 2070, 1984, 1056, 2257, 64, 1809, 1522, 1409, 1125,  632,   41,  551])
S = np.array(satellites)[I]
f, (b,e) = satellite_dpc(S)

## Generate a contact plan. This can be very slow. 
sat_CP, (st, et) = sat_contact_plan(S, resolution=20, progress=True)

## Build the global 'multi-graph' representation from the contact plan
import networkx as nx
G = nx.MultiGraph()
G.add_nodes_from(range(len(S)))
keys = G.add_edges_from([(int(i),int(j),{ "start": s, "end": e }) for (i,j,s,e) in sat_CP])
