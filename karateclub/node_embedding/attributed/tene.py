import numpy as np
import networkx as nx
from numpy.linalg import inv
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

class TENE(object):
    r"""An implementation of `"TENE" <https://ieeexplore.ieee.org/document/8545577>`_
    from the ICASSP '18 paper "Enhanced Network Embedding with Text Information". The 
    procedure first calculates the truncated SVD of an adjcacency - feature matrix
    product. This matrix is further decomposed by a binary CCD based technique. 
       

    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        svd_iterations (int): SVD iteration count. Default is 20.
        seed (int): Random seed. Default is 42.
        alpha (float): Kernel matrix inversion parameter. Default is 0.3. 
        approximation_rounds (int): Matrix decomoposition iterations. Default is 100.
        binarization_rounds (int): Binarization iterations. Default is 20.
    """
    def __init__(self, dimensions=32, lower_countrol=10**-15,
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

    def fit(self, graph, T):
        """
        Fitting a TENE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **T** *(Scipy COO or Numpy matrix)* - The matrix of node features.
        """
        self.X = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
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
