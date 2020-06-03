import numpy as np
import networkx as nx
from scipy import sparse
from karateclub.estimator import Estimator

class TENE(Estimator):
    r"""An implementation of `"TENE" <https://ieeexplore.ieee.org/document/8545577>`_
    from the ICPR '18 paper "Enhanced Network Embedding with Text Information". The 
    procedure jointly factorizes the adjacency and node feature matrices using alternating
    least squares.
       
    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        lower_control (float): Embedding score minimal value. Default is 10**-15.
        alpha (float): Adjacency matrix regularization coefficient. Default is 0.1. 
        beta (float): Feature matrix regularization coefficient. Default is 0.1.
        iterations (int): ALS iterations. Default is 200.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions=32, lower_control=10**-15,
                 alpha=0.1, beta=0.1, iterations=200, seed=42):
        self.dimensions = dimensions
        self.lower_control = lower_control
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.seed = seed

    def _init_weights(self):
        """
        Setup basis and feature matrices.
        """
        self._M = np.random.uniform(0, 1, (self._X.shape[0], self.dimensions))
        self._U = np.random.uniform(0, 1, (self._X.shape[0], self.dimensions))
        self._Q = np.random.uniform(0, 1, (self._X.shape[0], self.dimensions))
        self._V = np.random.uniform(0, 1, (self._T.shape[1], self.dimensions))
        self._C = np.random.uniform(0, 1, (self.dimensions, self.dimensions))

    def _update_M(self):
        """
        Update node bases.
        """
        enum = self._X.dot(self._U)
        denom = self._M.dot(self._U.T.dot(self._U))
        self._M = np.multiply(self._M, enum/denom)
        self._M[self._M < self.lower_control] = self.lower_control

    def _update_V(self):
        """
        Update node features.
        """
        enum = self._T.T.dot(self._Q)
        denom = self._V.dot(self._Q.T.dot(self._Q))
        self._V = np.multiply(self._V, enum/denom)
        self._V[self._V < self.lower_control] = self.lower_control

    def _update_C(self):
        """
        Update transformation matrix.
        """
        enum = self._Q.T.dot(self._U)
        denom = self._C.dot(self._U.T.dot(self._U))
        self._C = np.multiply(self._C, enum/denom)
        self._C[self._C < self.lower_control] = self.lower_control

    def _update_U(self):
        """
        Update features.
        """
        enum = self._X.T.dot(self._M)+self.alpha*self._Q.dot(self._C)
        denom = self._U.dot((self._M.T.dot(self._M)+self.alpha*self._C.T.dot(self._C)))
        self._U = np.multiply(self._U, enum/denom)
        self._U[self._U < self.lower_control] = self.lower_control

    def _update_Q(self):
        """
        Update feature bases.
        """
        enum = self.alpha*self._U.dot(self._C.T)+self.beta*self._T.dot(self._V)
        denom = self.alpha*self._Q+self.beta*self._Q.dot(self._V.T.dot(self._V))
        self._Q = np.multiply(self._Q, enum/denom)
        self._Q[self._Q < self.lower_control] = self.lower_control

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
        Creating a normalized adjacency matrix.

        Return types:
            * **A_hat* - Normalized adjacency matrix.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def fit(self, graph, T):
        """
        Fitting a TENE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **T** *(Scipy COO or Numpy array)* - The matrix of node features.
        """
        self._set_seed()
        self._check_graph(graph)
        self._X = self._create_base_matrix(graph)
        self._T = T
        self._init_weights()
        for _ in range(self.iterations):
            self._update_M()
            self._update_V()
            self._update_C()
            self._update_U()
            self._update_Q()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate([self._M, self._Q], axis=1)
        return embedding
