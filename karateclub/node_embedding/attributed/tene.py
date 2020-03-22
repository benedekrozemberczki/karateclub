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
    """
    def __init__(self, dimensions=32, lower_control=10**-15,
                 alpha=0.1, beta=0.1, iterations=200):
        self.dimensions = dimensions
        self.lower_control = lower_control
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def _init_weights(self):
        """
        Setup basis and feature matrices.
        """
        self.M = np.random.uniform(0, 1, (self.X.shape[0], self.dimensions))
        self.U = np.random.uniform(0, 1, (self.X.shape[0], self.dimensions))
        self.Q = np.random.uniform(0, 1, (self.X.shape[0], self.dimensions))
        self.V = np.random.uniform(0, 1, (self.T.shape[1], self.dimensions))
        self.C = np.random.uniform(0, 1, (self.dimensions, self.dimensions))

    def _update_M(self):
        """
        Update node bases.
        """
        enum = self.X.dot(self.U)
        denom = self.M.dot(self.U.T.dot(self.U))
        self.M = np.multiply(self.M, enum/denom)
        self.M[self.M < self.lower_control] = self.lower_control

    def _update_V(self):
        """
        Update node features.
        """
        enum = self.T.T.dot(self.Q)
        denom = self.V.dot(self.Q.T.dot(self.Q))
        self.V = np.multiply(self.V, enum/denom)
        self.V[self.V < self.lower_control] = self.lower_control

    def _update_C(self):
        """
        Update transformation matrix.
        """
        enum = self.Q.T.dot(self.U)
        denom = self.C.dot(self.U.T.dot(self.U))
        self.C = np.multiply(self.C, enum/denom)
        self.C[self.C < self.lower_control] = self.lower_control

    def _update_U(self):
        """
        Update features.
        """
        enum = self.X.T.dot(self.M)+self.alpha*self.Q.dot(self.C)
        denom = self.U.dot((self.M.T.dot(self.M)+self.alpha*self.C.T.dot(self.C)))
        self.U = np.multiply(self.U, enum/denom)
        self.U[self.U < self.lower_control] = self.lower_control

    def _update_Q(self):
        """
        Update feature bases.
        """
        enum = self.alpha*self.U.dot(self.C.T)+self.beta*self.T.dot(self.V)
        denom = self.alpha*self.Q+self.beta*self.Q.dot(self.V.T.dot(self.V))
        self.Q = np.multiply(self.Q, enum/denom)
        self.Q[self.Q < self.lower_control] = self.lower_control

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
        self._check_graph(graph)
        self.X = self._create_base_matrix(graph)
        self.T = T
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
        embedding = np.concatenate([self.M, self.Q], axis=1)
        return embedding
