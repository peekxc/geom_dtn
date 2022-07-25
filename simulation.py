from typing import *
from numpy.typing import ArrayLike

import numpy as np 
import networkx as nx


from array import array
from networkx import Graph


contact_plan = np.array([
  [1, 2, 0.0, 1.0],
  [1, 2, 2.0, 3.0],
  [1, 2, 3.5, 8.0],
  [1, 3, 0.0, 2.0],
  [1, 4, 1.0, 2.5],
  [2, 3, 2.0, 3.0],
  [2, 4, 0.0, 0.5],
  [3, 5, 1.0, 5.0],
  [4, 6, 1.0, 3.0],
  [2, 6, 3.0, 5.0]
])

import networkx as nx
G = nx.MultiGraph()
n = int(np.max(contact_plan[:,[0,1]].flatten()))
G.add_nodes_from(range(1, n+1))
keys = G.add_edges_from([(int(i),int(j),{ "start": s, "end": e }) for (i,j,s,e) in contact_plan])

# nx.draw_networkx(G)
pos = nx.planar_layout(G)
ax = plt.gca()
nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 200, alpha = 1)
for e in G.edges:
  opt = dict(arrowstyle="->", color="0.5", shrinkA=5, shrinkB=5, patchA=None, patchB=None, connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])))
  ax.annotate("",xy=pos[e[0]], xycoords='data',xytext=pos[e[1]], textcoords='data',arrowprops=opt)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 200, alpha = 1)
nx.draw_networkx_labels(G, pos)
plt.show()


# def RoutingNode():


class RoutingSimulation():
  def __init__(contact_plan: ArrayLike):
    self.CP: ArrayLike = contact_plan
    self.time: float = 0.00
    self.unit: float = 0.01
    self.n: int = int(np.max(contact_plan[:,0:2].flatten()))
    self.G: Graph = Graph()
    node_ids = np.array(list(range(7)), dtype=np.int32)
    
    G = nx.Graph()
    G.add_nodes_from(node_ids)

  def set_queue_sizes(queue_sizes: ArrayLike):
    assert len(queue_sizes) == n
    for nid, qs in zip(node_ids, queue_sizes):
      G.nodes[nid]['queue'] = np.zeros(qs, dtype=np.int32)

  def send_message(u: int, v: int):
    G[1][2]

  def tick(self):
    self.time += self.unit


# Process's are generator functions that create and yield events, and wait for event triggers
# - Processes get suspended upon yielding events; resumed upon event triggers 
# - Messages, vehicles, customers, etc. are modeled as processes 
# - Timeout is the prototypical *event* that is yielded: a process that calls timeout(x) gets suspended for 'x' amount of time, 
# - resuming after the callback registered by timeout(...) is trigger after 'x' units of time
# - Once the yielded event is triggered (“it occurs”), the simulation will resume the function/process
# - The two most common actions by a process: 1) wait for another process to finish and 2) interrupt another process while it is waiting for an event.



#   def 

from typing import * 
from numpy.typing import ArrayLike

import numpy as np 
import simpy 
from simpy import Environment, Resource, Store, Timeout, Event, Container

from dataclasses import dataclass


G = nx.Graph()
G.add_nodes_from(range(3))
G.add_edges_from([(0,1), (1,2)])

def router(node_id: int):
  if node_id == 0:
    return(1)
  elif node_id == 1:
    return(2)
  return(2)

class Node(object):
  def __init__(self, env, G, node_id: int, f: Callable, qs: int = 10):
    self.G = G
    self.env = env
    self.id = node_id
    self.buffer = Container(env, capacity=qs)
    self.action = env.process(self.run())
    self.oracle = f

  def run(self):
    while True:
      next_node = self.oracle(self.id)
      if next_node == self.id:
        yield self.env.timeout(1) # tick 
      else: 
        
        yield self.env.timeout(1) # tick 

class Edge(object):
  def __init__(self, env, edge_id: Tuple):
    self.env = env
    self.id = edge_id
    self.buffer = Container(env, capacity=qs)

def transfer_edge(self, G, node_id: int, target_id: int, amount: int): 
  assert target_id in list(G.neighbors(node_id))
  a, b = min(node_id,target_id), max(node_id,target_id)
  E = G.edges[(a,b)]['p']
  E.buffer.put(amount)
  yield self.env.timeout(duration)



env = Environment()



node_ids = np.fromiter(G.nodes, int)
node_attr = { nid : dict(p=Node(env, nid, cap)) for nid, cap in zip(node_ids, [10,2,10]) }
nx.set_node_attributes(G, node_attr)






def Node(env, id: int, ):
  while True: 
    yield env.timeout(1) # tick 



@dataclass(init=True)
class Message:
  source: int 
  target: int
  ttlive: int
  size: int
  time: int
  def __init__(self, s,t,ttl,sz,ct):
    self.source=s
    self.target=t
    self.ttlive=ttl
    self.size=sz
    self.time=ct

class Node(object):
  def __init__(self, env: Environment, node_id: int, oracle: Callable, qs: int = 10):
    self.env = env
    self.id = node_id
    self.buffer = Store(env, capacity=qs)
    self.rtable = oracle # routing scheme / table

  def run(self):
    while True:
      print(f'Node {self.id}: {self.env.now}')
      yield self.env.timeout(1)

      for msg in self.queue.items:
        oracle(self.id, msg)




## A payload is a planned amount of messages that need to be sent along the network


class GroundStation(object):
  def __init__(self, env: Environment, G: Graph, capacity: int, name: str):
    self.env = env 
    self.store = Store(env, capacity=100)
    self.name = name
    self.network = G

  def load_payload(self):
    n = len(self.network.nodes)
    target_ids = np.random.choice(range(n), self.store.capacity)
    for t in target_ids:
      self.store.put(Message(0, t, 20, 3, 0))

  # def transmit():
  #   while True:
  #     yield self.env.timeout(1)
  #     msg = yield self.store.get()
  #     print(f"Sending message ({msg.source}, {msg.target}) at time {env.now}")


class RoutingSimulation(object):
  def __init__(self, env: Environment, network: Graph, nodes: List[Node], sources: List[GroundStation]):
    self.env = env
    self.G = network
    self.Nodes = nodes
    self.sources = sources

  def route(self, node_id: int, msg: Message, scheme: str = "optimal"):
    ''' 
    Routing scheme - uses some knowledge oracles to determine how to route a given message 

    Given 'node_id' carrying 'msg', returns the next node id to send 'msg' to, if possible

    If it's not possible to send 'msg' at the current simulation time, returns the current node_id. 
    '''
    c_time = self.env.now

    ## First contact strategy 
    u = node_id
    for v,C in G[u].items():
      for contact in C.values():
        if c_time >= contact['start'] and c_time < contact['end']:
          return(v)
    return(u)

  def run(self):
    while True:
      for src in self.sources:
        msg = yield src.store.get()
        print(f"Transmitting message ({msg.source}, {msg.target}) at time {env.now}")

    # What should this function do? 

env = simpy.Environment()
gs = GroundStation(env, G, 10, "JPL")
gs.load_payload()

sim = RoutingSimulation(env, G, [], [gs])

sim.route(node_id=1, msg=Message(s=1, t=2, ttl=10, sz=10, ct=0.5))

prod = env.process(sim.run())
env.run(until=10)

# def RoutingScheduler(object):
#   def __init__(self,env,G,payloads):
#     self.env = env
#     self.network = G
#     self.payloads = payloads

#   def schedule():


prod = env.process(producer(env, store))

def routing_sim(, ):



node1 = Node(env, 1).run()
p = simpy.events.Process(env=env, generator=node1)

env.run(until=10)