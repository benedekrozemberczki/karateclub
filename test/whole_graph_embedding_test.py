import networkx as nx

from karateclub.graph_embedding import Graph2Vec, FGSD, GL2Vec, SF, NetLSD, GeoScattering, FeatherGraph


def test_feather_graph():
    """
    Test the grap level FEATHER embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = FeatherGraph()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == 4*model.order*model.eval_points


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

def test_fgsd():
    """
    Test the GL2Vec embedding.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = GL2Vec()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions
