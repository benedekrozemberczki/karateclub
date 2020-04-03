import numpy as np
import networkx as nx
import scipy.sparse as sps
from karateclub.estimator import Estimator

class GeoScattering(Estimator):
    r"""An implementation of `"GeoScattering" <https://arxiv.org/abs/1805.10712>`_
    from the ICML '18 paper "NetLSD: Hearing the Shape of a Graph".

    Args:
        order (int): Adjacency matrix powers. Default is 4.
    """
    def __init__(self, order=4):
        self.order = order

    def _calculate_geoscattering(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **features** *(Numpy array)* - The embedding of a single graph.
        """
        return features

    def fit(self, graphs):
        """
        Fitting a NetLSD model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._check_graphs(graphs)
        self._embedding = [self._calculate_geoscattering(graph) for graph in graphs]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
