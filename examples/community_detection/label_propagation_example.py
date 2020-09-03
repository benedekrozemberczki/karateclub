"""Label Propagation example."""

import networkx as nx
from karateclub.community_detection.non_overlapping import LabelPropagation

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = LabelPropagation()

model.fit(g)
