import numpy as np
import networkx as nx
from karateclub.graph_embedding import Graph2Vec, FGSD, GL2Vec, SF, IGE
from karateclub.graph_embedding import NetLSD, GeoScattering, FeatherGraph


def test_feather_graph():
    """
    Test the graph level FEATHER embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = FeatherGraph()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == 4*model.order*model.eval_points
    assert type(embedding) == np.ndarray

    graphs = [nx.newman_watts_strogatz_graph(150, 5, 0.3) for _ in range(100)]

    model = FeatherGraph(order=3)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == 4*model.order*model.eval_points
    assert type(embedding) == np.ndarray

    model = FeatherGraph(pooling="mean")

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == 4*model.order*model.eval_points
    assert type(embedding) == np.ndarray

    model = FeatherGraph(pooling="max")

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == 4*model.order*model.eval_points
    assert type(embedding) == np.ndarray

    model = FeatherGraph(pooling="min")

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == 4*model.order*model.eval_points
    assert type(embedding) == np.ndarray


def test_fgsd():
    """
    Test the FGSD embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = FGSD()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.hist_bins
    assert type(embedding) == np.ndarray

    graphs = [nx.newman_watts_strogatz_graph(150, 5, 0.3) for _ in range(100)]

    model = FGSD(hist_bins=8)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.hist_bins
    assert type(embedding) == np.ndarray


def test_graph2vec():
    """
    Test the Graph2Vec embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = Graph2Vec()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions

    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = Graph2Vec(erase_base_features=True)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions

    graphs = []

    for _ in range(50):
        graph = nx.newman_watts_strogatz_graph(50, 5, 0.3)
        nx.set_node_attributes(graph, {j: str(j) for j in range(50)}, "feature")
        graphs.append(graph)

    model = Graph2Vec(attributed=True)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_gl2vec():
    """
    Test the GL2Vec embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = GL2Vec()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    graphs = [nx.newman_watts_strogatz_graph(150, 5, 0.3) for _ in range(100)]

    model = GL2Vec(dimensions=16)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_sf():
    """
    Test the SF embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = SF(dimensions=8)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = SF(dimensions=128)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_netlsd():
    """
    Test the NetLSD embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = NetLSD()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.scale_steps
    assert type(embedding) == np.ndarray

    graphs = [nx.newman_watts_strogatz_graph(500, 5, 0.3) for _ in range(100)]

    model = NetLSD()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.scale_steps
    assert type(embedding) == np.ndarray


def test_geoscattering():
    """
    Test the GeoScattering embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(10)]

    for order in range(4, 20):
        for moment in range(4, 7):

            model = GeoScattering(order=order, moments=moment)

            model.fit(graphs)
            embedding = model.get_embedding()

            first_block = 3*model.order
            second_block = 3*(model.order+1)*(model.moments-1)
            third_block = 3*(model.order-1)*model.order*(model.moments-1)/2

            feature_count = first_block+second_block+third_block
    
            assert embedding.shape[0] == len(graphs)
            assert embedding.shape[1] == feature_count
            assert type(embedding) == np.ndarray

def test_ige():
    """
    Test the IGE embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(10)]

    for order in range(4, 20):
        for moment in range(4, 7):

            model = IGE()

            model.fit(graphs)
            embedding = model.get_embedding()
    
            assert embedding.shape[0] == len(graphs)
            assert type(embedding) == np.ndarray
