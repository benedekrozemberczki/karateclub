import numpy as np
import networkx as nx
from karateclub import DeepWalk, Walklets, HOPE, NetMF, Diff2Vec, GraRep, Node2Vec
from karateclub import NodeSketch, LaplacianEigenmaps, NMFADMM


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
    assert type(embedding) == np.ndarray

    model = DeepWalk(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

def test_node2vec():
    """
    Testing the Node2Vec class.
    """
    model = Node2Vec()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = Node2Vec(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

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
    assert type(embedding) == np.ndarray

    model = Walklets(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions*model.window_size
    assert type(embedding) == np.ndarray


def test_hope():
    """
    Testing the HOPE class.
    """
    model = HOPE()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = HOPE(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


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
    assert type(embedding) == np.ndarray

    model = NetMF(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_diff2vec():
    """
    Testing the Diff2Vec class.
    """
    model = Diff2Vec()

    graph = nx.watts_strogatz_graph(100, 10, 1.0)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = Diff2Vec(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 1.0)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray



def test_grarep():
    """
    Testing the GraRep class.
    """
    model = GraRep()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions*model.order
    assert type(embedding) == np.ndarray

    model = GraRep(dimensions=16)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions*model.order
    assert type(embedding) == np.ndarray


def test_nodesketch():
    """
    Testing the NodeSketch class.
    """
    model = NodeSketch()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = NodeSketch(dimensions=16)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray
   
   

def test_laplacianeigenmaps():
    """
    Testing the Laplacian Eigenmaps class.
    """
    model = LaplacianEigenmaps()

    graph = nx.watts_strogatz_graph(500, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = LaplacianEigenmaps(dimensions=16)

    graph = nx.watts_strogatz_graph(200, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_nmf_admm():
    """
    Testing the NMF ADMM class.
    """
    model = NMFADMM()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions*2
    assert type(embedding) == np.ndarray

    model = NMFADMM(dimensions=32)

    graph = nx.watts_strogatz_graph(200, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions*2
    assert type(embedding) == np.ndarray
