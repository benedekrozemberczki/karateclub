"""GLEE illustrative example."""

import networkx as nx
from karateclub.node_embedding.neighbourhood import GLEE

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

model = GLEE()

model.fit(g)
