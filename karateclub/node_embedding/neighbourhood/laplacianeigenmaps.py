import numpy as np
import networkx as nx
import scipy.sparse as sps
from karateclub.estimator import Estimator

class LaplacianEigenmaps(Estimator):
    r"""An implementation of `"Laplacian Eigenmaps" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
    """
    def __init__(self, dimensions=128):

        self.dimensions = dimensions

    def fit(self, graph):
        """
        Fitting a Laplacian EigenMaps model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        number_of_nodes = graph.number_of_nodes()
        L_tilde = nx.normalized_laplacian_matrix(graph, nodelist=range(number_of_nodes))
        eigenvalues, embedding = sps.linalg.eigsh(L_tilde, k=self.dimensions, return_eigenvectors=True)
        self._embedding = embedding


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
