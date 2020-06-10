import networkx as nx

from karateclub.graph_embedding import Graph2Vec, FGSD, GL2Vec, SF, NetLSD, GeoScattering, FeatherGraph


def test_feather_graph():
    """
    Test the grap level FEATHER.
    """
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

    model = FeatherGraph()

    model.fit(graphs)
    embedding = model.get_embedding()
    
    assert embedding.shape[0] == len(graphs)
    assert embedding.shape[1] == 2*model.order*model.eval_points
