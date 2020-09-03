"""SCD example."""

import networkx as nx
from karateclub.community_detection.non_overlapping import SCD

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = SCD()

model.fit(g)
