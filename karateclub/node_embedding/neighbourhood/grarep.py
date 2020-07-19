import math
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

class GraRep(Estimator):
    r"""An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.

    Args:
        dimensions (int): Number of individual embedding dimensions. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 5.
        seed (int): SVD random seed. Default is 42.
    """
    def __init__(self, dimensions: int=32, iteration: int=10, order: int=5, seed: int=42):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order
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

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.

        Return types:
            * **(A_hat, A_hat)** *(Tuple of SciPy arrays)* - Normalized adjacency matrices.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return (A_hat, A_hat)

    def _create_target_matrix(self):
        """
        Creating a log transformed target matrix.

        Return types:
            * **target_matrix** *(SciPy array)* - The PMI matrix.
        """
        self._A_tilde = sparse.coo_matrix(self._A_tilde.dot(self._A_hat))
        scores = np.log(self._A_tilde.data)-math.log(self._A_tilde.shape[0])
        rows = self._A_tilde.row[scores < 0]
        cols = self._A_tilde.col[scores < 0]
        scores = scores[scores < 0]
        target_matrix = sparse.coo_matrix((scores, (rows, cols)),
                                          shape=self._A_tilde.shape,
                                          dtype=np.float32)

        return target_matrix

    def _create_single_embedding(self, target_matrix):
        """
        Fitting a single SVD embedding of a PMI matrix.
        """
        svd = TruncatedSVD(n_components=self.dimensions,
                           n_iter=self.iterations,
                           random_state=self.seed)
        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        self._embeddings.append(embedding)

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a GraRep model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        self._A_tilde, self._A_hat = self._create_base_matrix(graph)
        self._embeddings = []
        target_matrix = self._create_target_matrix()
        self._create_single_embedding(target_matrix)
        for step in range(self.order-1):
            target_matrix = self._create_target_matrix()
            self._create_single_embedding(target_matrix)

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate(self._embeddings, axis=1)
        return embedding
