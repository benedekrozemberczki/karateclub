"""Role2Vec illustrative example."""

import networkx as nx
from karateclub.node_embedding.structural import Role2Vec

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = Role2Vec()

model.fit(g)
