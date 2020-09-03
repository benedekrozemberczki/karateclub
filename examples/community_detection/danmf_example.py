"""DANMF example."""

import networkx as nx
from karateclub.community_detection.overlapping import DANMF

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = DANMF()

model.fit(g)
