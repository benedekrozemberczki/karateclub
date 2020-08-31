"""Diff2Vec illustrative example."""

import networkx as nx
from karateclub.node_embedding.neighbourhood import Diff2Vec

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = Diff2Vec()

model.fit(g)
