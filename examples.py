"""Example runs with Karate Club."""

import networkx as nx

import random
import community
import numpy as np
from scipy.sparse import coo_matrix

from karateclub.community_detection.non_overlapping import EdMot, LabelPropagation, SCD, GEMSEC
from karateclub.node_embedding.attributed import BANE, TENE, TADW, FSCNMF, SINE, MUSAE, FeatherNode
from karateclub.dataset import GraphReader, GraphSetReader

#---------------
# Feather Node
#---------------

g = nx.newman_watts_strogatz_graph(150, 10, 0.2)

X = np.random.uniform(0, 1, (150, 127))

model = FeatherNode()

model.fit(g, X)
embedding = model.get_embedding()

#----------------
# GEMSEC example
#----------------

g = nx.newman_watts_strogatz_graph(50, 5, 0.05)

model = GEMSEC()

model.fit(g)
memberships = model.get_memberships()


#------------------------------------
# MUSAE example
#------------------------------------

g = nx.newman_watts_strogatz_graph(50, 10, 0.2)

X = {i: random.sample(range(150),50) for i in range(50)}

row = np.array([k for k, v in X.items() for val in v])
col = np.array([val for k, v in X.items() for val in v])
data = np.ones(50*50)
shape = (50, 150)

X = coo_matrix((data, (row, col)), shape=shape)

model = MUSAE()

model.fit(g, X)

#--------------
# SINE example
#--------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.2)

X = {i: random.sample(range(150),50) for i in range(100)}

row = np.array([k for k, v in X.items() for val in v])
col = np.array([val for k, v in X.items() for val in v])
data = np.ones(100*50)
shape = (100, 150)

X = coo_matrix((data, (row, col)), shape=shape)

model = SINE()

model.fit(g, X)

#-------------
# SCD example
#-------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.2)

model = SCD()

model.fit(g)

model.get_memberships()

#-----------------
# FSCNMF example
#-----------------

g = nx.newman_watts_strogatz_graph(200, 20, 0.05)

x = np.random.uniform(0, 1, (200, 200))

model = FSCNMF()

model.fit(g, x)

#---------------
# TADW example
#---------------

g = nx.newman_watts_strogatz_graph(200, 20, 0.05)

x = np.random.uniform(0, 1, (200, 200))

model = TADW()

model.fit(g, x)


#-------------------------
# GraphSet reader example
#-------------------------

reader = GraphSetReader("reddit10k")

graphs = reader.get_graphs()
y = reader.get_target()

#----------------------
# Graph reader example
#----------------------

reader = GraphReader("facebook")

graph = reader.get_graph()
target = reader.get_target()

#---------------
# TENE example
#---------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

t = np.random.uniform(0,1,(100,2000))

tp = nx.newman_watts_strogatz_graph(100, 20, 0.05)

tp = nx.adjacency_matrix(tp)

model = TENE()

model.fit(g, t)
model.get_embedding()

model.fit(g, tp)
model.get_embedding()


#---------------
# BANE example
#---------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

x = np.random.uniform(0,1,(100,2000))

p = nx.newman_watts_strogatz_graph(100, 20, 0.05)

x = nx.adjacency_matrix(p)
model = BANE()

model.fit(g, x)

#----------------
# Edmot example
#----------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.9)

model = EdMot(3, 0.5)

model.fit(g)

#----------------------------
# Label Propagation example
#----------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = LabelPropagation()

model.fit(g)
