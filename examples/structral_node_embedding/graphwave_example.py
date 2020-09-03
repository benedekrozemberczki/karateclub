"""GraphWave illustrative example."""

import networkx as nx
from karateclub.node_embedding.structural import GraphWave

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = GraphWave()

model.fit(g)
