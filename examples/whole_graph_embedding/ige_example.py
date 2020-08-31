"""IGE illustrative example."""

import networkx as nx
from karateclub.graph_embedding import IGE

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(1000)]

model = IGE()

model.fit(graphs)
model.get_embedding()
