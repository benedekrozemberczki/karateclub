"""Node2Vec illustrative example."""

import networkx as nx
from karateclub.node_embedding.neighbourhood import Node2Vec

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = Node2Vec()

model.fit(g)
