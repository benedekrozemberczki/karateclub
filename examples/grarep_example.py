"""GraRep illustrative example."""

import networkx as nx
from karateclub.node_embedding.neighbourhood import GraRep

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = GraRep()

model.fit(g)
