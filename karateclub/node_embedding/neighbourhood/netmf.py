import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

class NetMF(Estimator):
    r"""An implementation of `"NetMF" <https://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf>`_
    from the WSDM '18 paper "Network Embedding as Matrix Factorization: Unifying
    DeepWalk, LINE, PTE, and Node2Vec". The procedure uses sparse truncated SVD to
    learn embeddings for the pooled powers of the PMI matrix computed from powers
    of the normalized adjacency matrix.

    Args:
        dimensions (int): Number of embedding dimension. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 2.
        negative_samples (in): Number of negative samples. Default is 1.
        seed (int): SVD random seed. Default is 42.
    """
    def __init__(self, dimensions: int=32, iteration: int=10, order: int=2,
                 negative_samples: int=1, seed: int=42):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order
        self.negative_samples = negative_samples
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
        Creating the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **(A_hat, A_hat, A_hat, D_inverse)** *(SciPy arrays)* - Normalized adjacency matrices.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return (A_hat, A_hat, A_hat, D_inverse)

    def _create_target_matrix(self, graph):
        """
        Creating a log transformed target matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **target_matrix** *(SciPy array)* - The shifted PMI matrix.
        """
        A_pool, A_tilde, A_hat, D_inverse = self._create_base_matrix(graph)
        for _ in range(self.order-1):
            A_tilde = sparse.coo_matrix(A_tilde.dot(A_hat))
            A_pool = A_pool + A_tilde
        A_pool = (graph.number_of_edges()*A_pool)/(self.order*self.negative_samples)
        A_pool = sparse.coo_matrix(A_pool.dot(D_inverse))
        A_pool.data[A_pool.data < 1.0] = 1.0
        target_matrix = sparse.coo_matrix((np.log(A_pool.data), (A_pool.row, A_pool.col)),
                                          shape=A_pool.shape,
                                          dtype=np.float32)
        return target_matrix

    def _create_embedding(self, target_matrix):
        """
        Fitting a truncated SVD embedding of a PMI matrix.
        """
        svd = TruncatedSVD(n_components=self.dimensions,
                           n_iter=self.iterations,
                           random_state=self.seed)
        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        return embedding

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a NetMF model.
    
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        target_matrix = self._create_target_matrix(graph)
        self._embedding = self._create_embedding(target_matrix)

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.
    
        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
