"""Ego Splitting example."""

import networkx as nx
from karateclub.community_detection.overlapping import EgoNetSplitter

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = EgoNetSplitter()

model.fit(g)
