import networkx as nx

import random
import community
import numpy as np
from scipy.sparse import coo_matrix
from karateclub.node_embedding.attributed import BANE, TENE, TADW, FSCNMF, SINE, MUSAE, FeatherNode


def test_feather_node():

    graph = nx.newman_watts_strogatz_graph(250, 10, 0.2)
    features = np.random.uniform(0, 1, (250, 2000))
    model = FeatherNode()
    model.fit(graph, features)
    embedding = model.get_embedding()
    print(embedding.shape)
    assert embedding.shape[0] == graph.number_of_nodes()
