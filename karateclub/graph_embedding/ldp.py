import numpy as np
import networkx as nx
from typing import List
from scipy.sparse.linalg import eigsh
from karateclub.estimator import Estimator

class LDP(Estimator):
    r"""An implementation of `"LDP" <A Simple Baseline Algorithm for Graph Classification>`_
    from the NeurIPS Relational Representation Learning Workshop '18 paper "A Simple Baseline Algorithm for Graph Classification".
    The procedure calculates the k lowest egeinvalues of the normalized Laplacian.
    If the graph has a lower number of eigenvalues than k the representation is padded with zeros.

    Args:
        bins (int): Number of histogram bins. Default is 32.
    """
    def __init__(self, bins: int=32):
        self.bins = bins

    def _calculate_ldp(self, graph):
        """
        Calculating the local degree profile features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of a single graph.
        """
        degrees = np.log(np.array([graph.degree[n] for n in range(graph.number_of_nodes())]))
        embedding = np.hist(degrees, bins=self.bins)
        return embedding

    def fit(self, graphs):
        """
        Fitting an LDP model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_sf(graph) for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
