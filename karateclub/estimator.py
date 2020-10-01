import random
import numpy as np
import networkx as nx
from typing import List

"""General Estimator base class."""

class Estimator(object):
    """Estimator base class with constructor and public methods."""

    def __init__(self):
        """Creatinng an estimator."""
        pass


    def fit(self):
        """Fitting a model."""
        pass


    def get_embedding(self):
        """Getting the embeddings (graph or node level)."""
        pass


    def get_memberships(self):
        """Getting the membership dictionary."""
        pass


    def get_cluster_centers(self):
        """Getting the cluster centers."""
        pass


    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _check_connectivity(self, graph: nx.classes.graph.Graph):
        """Checking the connected nature of a single graph."""
        connected = nx.is_connected(graph)
        assert connected, "Graph is not connected."


    def _check_directedness(self, graph: nx.classes.graph.Graph):
        """Checking the undirected nature of a single graph."""
        directed = nx.is_directed(graph)
        assert directed == False, "Graph is directed."


    def _check_indexing(self, graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])
        assert numeric_indices == node_indices, "The node indexing is wrong."


    def _check_graph(self, graph: nx.classes.graph.Graph):
        """Check the Karate Club assumptions about the graph."""
        self._check_connectivity(graph)
        self._check_directedness(graph)
        self._check_indexing(graph)


    def _check_graphs(self, graphs: List[nx.classes.graph.Graph]):
        """Check the Karate Club assumptions for a list of graphs."""
        for graph in graphs:
            self._check_graph(graph)

