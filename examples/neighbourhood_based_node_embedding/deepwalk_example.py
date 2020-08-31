"""DeepWalk illustrative example."""

import networkx as nx
from karateclub.node_embedding.neighbourhood import DeepWalk

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = DeepWalk()

model.fit(g)
