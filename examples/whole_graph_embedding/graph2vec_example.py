"""Graph2Vec illustrative example."""

import networkx as nx
from karateclub.graph_embedding import Graph2Vec

# Graph2Vec attributed example

graphs = []

for i in range(50):
    graph = nx.newman_watts_strogatz_graph(50, 5, 0.3)
    nx.set_node_attributes(graph, {j: str(j) for j in range(50)}, "feature")
    graphs.append(graph)
model = Graph2Vec(attributed=True)

model.fit(graphs)
model.get_embedding()


# Graph2Vec generic example

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(1000)]

model = Graph2Vec()

model.fit(graphs)
model.get_embedding()
