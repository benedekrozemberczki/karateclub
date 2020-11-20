import numpy as np
import networkx as nx
from scipy import sparse
from karateclub.estimator import Estimator

class RandNE(Estimator):
    r"""An implementation of `"RandNE" <https://zw-zhang.github.io/files/2018_ICDM_RandNE.pdf>`_ from the ICDM '18 paper "Billion-scale Network Embedding with Iterative Random Projection". The procedure uses normalized adjacency matrix based
    smoothing on an orthogonalized random normally generate base node embedding matrix.

    Args:
        dimensions (int): Number of embedding dimension. Default is 128.
        alphas (list): Smoothing weights for adjacency matrix powers. Default is [0.5, 0.5].
        seed (int): Random seed. Default is 42.
    """
    def __init__(self, dimensions: int=128, alphas: list=[0.5, 0.5], seed: int=42):
        self.dimensions = dimensions
        self.alphas = alphas
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

    def _create_smoothing_matrix(self, graph):
        """
        Creating the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **(A_hat, A_hat, A_hat, D_inverse)** *(SciPy arrays)* - Normalized adjacency matrices.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def _create_embedding(self, A_hat):
        """
        Using the random orthogonal smoothing.
        """
        sd = 1/self.dimensions
        base_embedding = np.random.normal(0, sd, (A_hat.shape[0], self.dimensions))
        base_embedding, _ = np.linalg.qr(base_embedding)
        embedding = np.zeros(base_embedding.shape)
        alpha_sum = sum(self.alphas)
        for alpha in self.alphas:
            base_embedding = A_hat.dot(base_embedding)
            embedding = embedding + alpha * base_embedding
        embedding = embedding / alpha_sum
        embedding = (embedding-embedding.mean(0))/embedding.std(0)
        return embedding

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a NetMF model.
    
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        A_hat = self._create_smoothing_matrix(graph)
        self._embedding = self._create_embedding(A_hat)

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.
    
        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
