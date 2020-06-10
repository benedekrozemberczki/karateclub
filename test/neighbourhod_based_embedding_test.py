import networkx as nx

from karateclub import DeepWalk, Walklets, HOPE, NetMF


def test_deepwalk():
    """
    Testing the DeepWalk class.
    """
    model = DeepWalk()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions


def test_walklets():
    """
    Testing the Walklets class.
    """
    model = Walklets()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions*model.window_size


def test_netmf():
    """
    Testing the NetMF class.
    """
    model = NetMF()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
   

