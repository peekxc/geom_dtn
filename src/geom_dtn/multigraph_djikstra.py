import numpy as np



# ## Dynamic network that supports intervals of edge activity
# import dynetx as dn
# g = dn.DynGraph(edge_removal=False)
# g.add_interaction(u=1, v=2, t=0)
# g.add_interaction(u=1, v=2, t=0, e=3)


## Let's represent multigraph as (m x 5) matrix (multi-edge list)
#   (u, v, d, s, e)
# where: 
# u, v := edge, where labels are non-negative integers (should these be pos/neg to incorporate direction?)
# d := integer indicating u -> v (1) or u <- v (-1) or u <-> (0)
# s, e := non-negative integers starting/ending times, stored inclusively/exclusively [s, e), 



## Contact plan 

