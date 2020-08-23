import math
import numpy as np
import networkx as nx
from typing import List
import scipy.stats.mstats
import scipy.sparse as sparse
from karateclub.estimator import Estimator

class IGE(Estimator):
    r"""An implementation of `"Invariant Graph Embedding" <https://graphreason.github.io/papers/16.pdf>`_
    from the ICML 2019 Workshop on Learning and Reasoning with Graph-Structured Data paper 
    "Invariant Embedding for Graph Classification". The procedure
    uses ...

    Args:
        feature_embedding_dimensions (list): Feature embedding dimensions. Default is [3, 5]
        spectral_embedding_dimensions (list): Spectral embedding dimensions. Default is [10, 20].
        histogram_bins (list): Number of histogram bins. Default is [21, 31].
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, feature_embedding_dimensions: List[int]=[3, 5],
                 spectral_embedding_dimensions: List[int]=[10, 20],
                 histogram_bins: List[int]=[21, 31],
                 seed: int=42):
        self.feature_embedding_dimensions = feature_embedding_dimensions
        self.spectral_embedding_dimensions = spectral_embedding_dimensions
        self.histogram_bins = histogram_bins
        self.seed = seed


    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


    def _get_normalized_adjacency(self, graph):
        """
        Calculating the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **A_hat** *(SciPy array)* - The scattering matrix of the graph.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = sparse.identity(graph.number_of_nodes()) + D_inverse.dot(A)
        A_hat = 0.5*A_hat
        return A_hat


    def _calculate_invariant_embedding(self, graph):
        features = []
        features = self._get_embedding_features(graph, features)
        features = self._get_spectral_features(graph, features)
        features = self._get_histogram_features(graph, features
        return np.ones((1, 128))

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting an Invariant Graph Embedding model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_invariant_embedding(graph) for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
