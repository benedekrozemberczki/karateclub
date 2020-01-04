import math
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

class BoostNE(Estimator):
    r"""An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.

    Args:
        dimensions (int): Number of individual embedding dimensions. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 5

    """
    def __init__(self, dimensions=32, iteration=10, order=5, seed=42):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree[0] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.

        Return types:
            * **(A_hat, A_hat, A_hat)** *(Tuple of SciPy arrays)* - Normalized adjacencies.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return (A_hat, A_hat, A_hat)

    def _create_target_matrix(self, graph):
        """
        Creating a log transformed target matrix.

        Return types:
            * **target_matrix** *(SciPy array)* - The PMI matrix.
        """
        A_tilde, A_hat, A_accum = self._create_base_matrix(graph)
        for _ in range(self.order-1):
            A_tilde = sparse.coo_matrix(A_tilde.dot(A_hat))
            A_accum = A_accum + A_tilde
        A_accum = A_accum / self.order
        return A_accum


    def fit(self, graph):
        """
        Fitting a GraRep model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        target_matrix = self._create_target_matrix(graph)






    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate(self.embeddings, axis=1)
        return embedding
