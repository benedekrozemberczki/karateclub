"""SINE Example."""

import random
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from karateclub.node_embedding.attributed import SINE

g = nx.newman_watts_strogatz_graph(100, 10, 0.2)

X = {i: random.sample(range(150),50) for i in range(100)}

row = np.array([k for k, v in X.items() for val in v])
col = np.array([val for k, v in X.items() for val in v])
data = np.ones(100*50)
shape = (100, 150)

X = coo_matrix((data, (row, col)), shape=shape)

model = SINE()

model.fit(g, X)
