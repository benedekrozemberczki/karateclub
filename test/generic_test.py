import networkx as nx

from karateclub import DeepWalk

def test_sampler():
    """
    Testing the sampler base class.
    """
    model = DeepWalk()

    graph = nx.watts_strogatz_graph(200, 10, 0)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[1] == model.dimensions
    

