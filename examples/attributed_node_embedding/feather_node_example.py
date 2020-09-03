"""Feather-N Example"""

import numpy as np
import networkx as nx
from karateclub.node_embedding.attributed import FeatherNode

g = nx.newman_watts_strogatz_graph(150, 10, 0.2)

X = np.random.uniform(0, 1, (150, 16))

model = FeatherNode()

model.fit(g, X)
embedding = model.get_embedding()
