"""NNSED example."""

import networkx as nx
from karateclub.community_detection.overlapping import NNSED

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = NNSED()

model.fit(g)
