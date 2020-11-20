"""LDP illustrative example."""

import networkx as nx
from karateclub.graph_embedding import LDP

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(1000)]

model = LDP()

model.fit(graphs)
embedding = model.get_embedding()

print(embedding.shape)
