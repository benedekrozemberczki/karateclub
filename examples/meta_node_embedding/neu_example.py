"""NEU illustrative example."""

import networkx as nx
from karateclub.node_embedding.meta import NEU
from karateclub.node_embedding.neighbourhood import NetMF

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

model = NetMF()

meta_model = NEU()

meta_model.fit(g, model)
