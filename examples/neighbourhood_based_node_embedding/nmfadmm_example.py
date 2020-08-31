"""NMFADMM illustrative example."""

import networkx as nx
from karateclub.node_embedding.neighbourhood import NMFADMM

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = NMFADMM()

model.fit(g)
