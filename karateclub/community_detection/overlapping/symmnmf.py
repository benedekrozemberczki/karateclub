import numpy as np
import networkx as nx
from typing import Dict
from scipy import sparse
from karateclub.estimator import Estimator

class SymmNMF(Estimator):

    r"""An implementation of `"Symm-NMF" <https://www.cc.gatech.edu/~hpark/papers/DaDingParkSDM12.pdf>`_
    from the SDM'12 paper "Symmetric Nonnegative Matrix Factorization for Graph Clustering". The procedure
    decomposed the second power od the normalized adjacency matrix with an ADMM based non-negative matrix
    factorization based technique. This results in a node embedding and each node is associated with an
    embedding factor in the created latent space.

    Args:
        dimensions (int): Number of dimensions. Default is 32.
        iterations (int): Number of power iterations. Default is 200.
        rho (float): Regularization tuning parameter. Default is 100.0.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions: int=32, iterations: int=200, rho: float=100.0, seed: int=42):

        self.dimensions = dimensions
        self.iterations = iterations
        self.rho = rho
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
            * **A_hat** *Scipy array* - Normalized adjacency.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def _setup_embeddings(self, graph):
        """
        Setup the node embedding matrices.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        number_of_nodes = graph.shape[0]
        non_zero = graph.nonzero()[0].shape[0]
        self._H = np.random.uniform(0, non_zero/(number_of_nodes**2), size=(number_of_nodes, self.dimensions))
        self._H_gamma = np.zeros((number_of_nodes, self.dimensions))
        self._I = np.identity(self.dimensions)

    def get_memberships(self) -> Dict[int, int]:
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        index = np.argmax(self._W, axis=1)
        memberships = {int(i): int(index[i]) for i in range(len(index))}
        return memberships

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self._H
        return embedding

    def _do_admm_update(self, A_hat):
        """
        Doing a single ADMM update with the adjacency matrix.
        
        """
        H_covar = np.linalg.inv(self._H.T.dot(self._H) + self.rho*self._I)
        self._W = (A_hat.dot(A_hat.T.dot(self._H)) + self.rho*self._H - self._H_gamma).dot(H_covar)
        self._W = np.maximum(self._W, 0)
        W_covar = np.linalg.inv(self._W.T.dot(self._W) + self.rho*self._I)
        self._H = (A_hat.dot(A_hat.T.dot(self._W)) + self.rho*self._W + self._H_gamma).dot(W_covar)
        self._H = np.maximum(self._H, 0)
        self._H_gamma = self._H_gamma + self.rho*(self._W-self._H)

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Symm-NMF clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._set_seed()
        self._check_graph(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        A_hat = self._create_base_matrix(graph)
        self._setup_embeddings(A_hat)
        for step in range(self.iterations):
            self._do_admm_update(A_hat)
