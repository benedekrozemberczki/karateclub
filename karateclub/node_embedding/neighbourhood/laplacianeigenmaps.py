import numpy as np
import networkx as nx
import scipy.sparse as sps
from karateclub.estimator import Estimator

class LaplacianEigenmaps(Estimator):
    r"""An implementation of `"Laplacian Eigenmaps" <https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering>`_
    from the NIPS '01 paper "Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering".
    The procedure extracts the eigenvectors corresponding to the largest values 
    of the graph Laplacian. These vectors are used as the node embedding.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions=128, seed=42):

        self.dimensions = dimensions
        self.seed = seed

    def fit(self, graph):
        """
        Fitting a Laplacian EigenMaps model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
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
