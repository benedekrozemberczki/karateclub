"""Laplacian Eigenmaps illustrative example."""

import networkx as nx
from karateclub.node_embedding.neighbourhood import LaplacianEigenmaps

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

model = LaplacianEigenmaps()

model.fit(g)
