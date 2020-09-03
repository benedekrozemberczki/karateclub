"""FGSD illustrative example."""

import networkx as nx
from karateclub.graph_embedding import FGSD

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

model = FGSD()

model.fit(graphs)
model.get_embedding()
