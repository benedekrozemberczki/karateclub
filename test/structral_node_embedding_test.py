import numpy as np
import networkx as nx
from karateclub import Role2Vec, GraphWave


def test_role2vec():
    """
    Testing the Role2Vec class.
    """
    model = Role2Vec()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = Role2Vec(dimensions=16)

    graph = nx.watts_strogatz_graph(200, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_graphwave():
    """
    Testing the GraphWave class.
    """
    model = GraphWave()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == 2*model.sample_number
    assert type(embedding) == np.ndarray

    model = GraphWave(mechanism="exact")

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == 2*model.sample_number
    assert type(embedding) == np.ndarray

    model = GraphWave(mechanism="whatelse")

    model = GraphWave()

    graph = nx.watts_strogatz_graph(1100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == 2*model.sample_number
    assert type(embedding) == np.ndarray
