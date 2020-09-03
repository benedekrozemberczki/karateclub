"""MUSAE Example."""

import random
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix

from karateclub.node_embedding.attributed import MUSAE

g = nx.newman_watts_strogatz_graph(50, 10, 0.2)

X = {i: random.sample(range(150),50) for i in range(50)}

row = np.array([k for k, v in X.items() for val in v])
col = np.array([val for k, v in X.items() for val in v])
data = np.ones(50*50)
shape = (50, 150)

X = coo_matrix((data, (row, col)), shape=shape)

model = MUSAE()

model.fit(g, X)
