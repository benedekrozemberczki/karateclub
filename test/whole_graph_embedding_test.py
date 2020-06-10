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

    graphs = []

    for i in range(50):
        graph = nx.newman_watts_strogatz_graph(50, 5, 0.3)
        nx.set_node_attributes(graph, {j: str(j) for j in range(50)}, "feature")
        graphs.append(graph)

    model = Graph2Vec(attributed=True)

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == model.dimensions

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
