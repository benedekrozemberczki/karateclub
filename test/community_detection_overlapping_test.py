import numpy as np
import networkx as nx
from karateclub.community_detection.overlapping import EgoNetSplitter, NNSED, DANMF, MNMF, BigClam, SymmNMF

def test_egonet_splitter():
    """
    Test the Ego Net splitter procedure.
    """
    graph = nx.newman_watts_strogatz_graph(100, 5, 0.3)

    model = EgoNetSplitter()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    graph = nx.newman_watts_strogatz_graph(150, 5, 0.3)

    model = EgoNetSplitter()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    # Test weighted graph
    graph = nx.les_miserables_graph()
    graph = nx.convert_node_labels_to_integers(graph)
    model = EgoNetSplitter(weight='weight')
    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    # Force unweighted
    graph = nx.les_miserables_graph()
    graph = nx.convert_node_labels_to_integers(graph)
    model = EgoNetSplitter(weight=None)
    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict



def test_nnsed():
    """
    Test the NNSED procedure.
    """
    graph = nx.newman_watts_strogatz_graph(100, 5, 0.3)

    model = NNSED()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    graph = nx.newman_watts_strogatz_graph(150, 5, 0.3)

    model = NNSED(dimensions=16)

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_danmf():
    """
    Test the DANMF procedure.
    """
    graph = nx.newman_watts_strogatz_graph(100, 5, 0.3)

    model = DANMF()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == 2*model.layers[-1]
    assert type(embedding) == np.ndarray


    graph = nx.newman_watts_strogatz_graph(200, 5, 0.3)

    model = DANMF()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == 2*model.layers[-1]
    assert type(embedding) == np.ndarray


def test_bigclam():
    """
    Test the BigClam procedure.
    """
    graph = nx.newman_watts_strogatz_graph(100, 5, 0.3)

    model = BigClam()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    graph = nx.newman_watts_strogatz_graph(200, 5, 0.3)

    model = BigClam(dimensions=8)

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray


def test_mnmf():
    """
    Test the MNMF procedure.
    """
    graph = nx.newman_watts_strogatz_graph(100, 5, 0.3)

    model = MNMF()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    centers = model.get_cluster_centers()

    assert centers.shape[0] == model.clusters
    assert centers.shape[1] == model.dimensions
    assert type(centers) == np.ndarray

    graph = nx.newman_watts_strogatz_graph(200, 5, 0.3)

    model = MNMF(dimensions=8)

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    centers = model.get_cluster_centers()

    assert centers.shape[0] == model.clusters
    assert centers.shape[1] == model.dimensions
    assert type(centers) == np.ndarray


def test_symmnmf():
    """
    Test the symmnmf procedure.
    """
    graph = nx.newman_watts_strogatz_graph(100, 5, 0.3)

    model = SymmNMF()

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    graph = nx.newman_watts_strogatz_graph(200, 5, 0.3)

    model = SymmNMF(dimensions=16)

    model.fit(graph)
    memberships = model.get_memberships()
    
    indices = [k for k, v in memberships.items()].sort()
    nodes = [node for node in graph.nodes()].sort()

    assert graph.number_of_nodes() == len(memberships)
    assert indices == nodes
    assert type(memberships) == dict

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray
