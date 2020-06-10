import networkx as nx

from karateclub.dataset import GraphReader, GraphSetReader

def test_graphset_reader():
    """
    Testing the reading a set of graphs.
    """
    reader = GraphSetReader("reddit10k")

    graphs = reader.get_graphs()
    target = reader.get_target()

    assert len(graphs) == target.shape[0]


def test_graph_reader():
    """
    Testing the reading of a graph with features.
    """
    reader = GraphReader("facebook")

    graph = reader.get_graph()
    target = reader.get_target()
    features = reader.get_features()

    assert graph.number_of_nodes() == target.shape[0]
    assert target.shape[0] == features.shape[0]

