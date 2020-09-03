"""Example runs with Karate Club."""

import networkx as nx

import random
import numpy as np
from scipy.sparse import coo_matrix

from karateclub.node_embedding.attributed import BANE, TENE, TADW, FSCNMF, SINE, MUSAE


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
