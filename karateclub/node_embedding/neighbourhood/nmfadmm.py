import numpy as np
import scipy as sp
import networkx as nx
from karateclub.estimator import Estimator

class NMFADMM(Estimator):
    r"""An implementation of `"NMF-ADMM" <http://statweb.stanford.edu/~dlsun/papers/nmf_admm.pdf>`_
    from the ICASSP '14 paper "Alternating Direction Method of Multipliers for 
    Non-Negative Matrix Factorization with the Beta-Divergence.". The procedure
    learns an embedding of the normalized adjacency matrix with by using the alternating
    direction method of multipliers to solve a non negative matrix factorization problem.

    Args:
        dimensions (int): Number of individual embedding dimensions. Default is 32.
        iterations (int): Number of ADMM iterations. Default is 100.
        rho (float): ADMM Tuning parameter. Default is 1.0.
    """
    def __init__(self, dimensions=32, iterations=100, rho=1.0):
        self.dimensions = dimensions
        self.iterations = iterations
        self.rho = rho
        
    def _init_weights(self):
        """
        Initializing model weights.
        """
        self.W = np.random.uniform(-0.1, 0.1, (self.V.shape[0], self.dimensions))
        self.H = np.random.uniform(-0.1, 0.1, (self.dimensions, self.V.shape[1]))
        X_i, Y_i = sp.nonzero(self.V)
        scores = self.W[X_i]*self.H[:, Y_i].T+np.random.uniform(0, 1, (self.dimensions, ))
        values = np.sum(scores, axis=-1)
        self.X = sp.sparse.coo_matrix((values, (X_i, Y_i)), shape=self.V.shape)
        self.W_plus = np.random.uniform(0, 0.1, (self.V.shape[0], self.dimensions))
        self.H_plus = np.random.uniform(0, 0.1, (self.dimensions, self.V.shape[1]))
        self.alpha_X = sp.sparse.coo_matrix((np.zeros(values.shape[0]), (X_i, Y_i)), shape=self.V.shape)
        self.alpha_W = np.zeros(self.W.shape)
        self.alpha_H = np.zeros(self.H.shape)

    def _update_W(self):
        """
        Updating user_1 matrix.
        """
        left = np.linalg.pinv(self.H.dot(self.H.T)+np.eye(self.dimensions))
        right_1 = self.X.dot(self.H.T).T+self.W_plus.T
        right_2 = (1.0/self.rho)*(self.alpha_X.dot(self.H.T).T-self.alpha_W.T)
        self.W = left.dot(right_1+right_2).T

    def _update_H(self):
        """
        Updating user_2 matrix.
        """
        left = np.linalg.pinv(self.W.T.dot(self.W)+np.eye(self.dimensions))
        right_1 = self.X.T.dot(self.W).T+self.H_plus
        right_2 = (1.0/self.rho)*(self.alpha_X.T.dot(self.W).T-self.alpha_H)
        self.H = left.dot(right_1+right_2)

    def _update_X(self):
        """
        Updating user_1-user_2 matrix.
        """
        iX, iY = sp.nonzero(self.V)
        values = np.sum(self.W[iX]*self.H[:, iY].T, axis=-1)
        scores = sp.sparse.coo_matrix((values-1, (iX, iY)), shape=self.V.shape)
        left = self.rho*scores-self.alpha_X
        right = (left.power(2)+4.0*self.rho*self.V).power(0.5)
        self.X = (left+right)/(2*self.rho)

    def _update_W_plus(self):
        """
        Updating positive primal user_1 factors.
        """
        self.W_plus = np.maximum(self.W+(1/self.rho)*self.alpha_W, 0)

    def _update_H_plus(self):
        """
        Updating positive primal user_2 factors.
        """
        self.H_plus = np.maximum(self.H+(1/self.rho)*self.alpha_H, 0)

    def _update_alpha_X(self):
        """
        Updating target matrix dual.
        """
        iX, iY = sp.nonzero(self.V)
        values = np.sum(self.W[iX]*self.H[:, iY].T, axis=-1)
        scores = sp.sparse.coo_matrix((values, (iX, iY)), shape=self.V.shape)
        self.alpha_X = self.alpha_X+self.rho*(self.X-scores)

    def _update_alpha_W(self):
        """
        Updating user dual factors.
        """
        self.alpha_W = self.alpha_W+self.rho*(self.W-self.W_plus)

    def _update_alpha_H(self):
        """
        Updating item dual factors.
        """
        self.alpha_H = self.alpha_H+self.rho*(self.H-self.H_plus)

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
        D_inverse = sp.sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **A_hat** *SciPy array* - Normalized adjacency matrix.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def fit(self, graph):
        """
        Fitting an NMF model on the normalized adjacency matrix with ADMM.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self.V = self._create_base_matrix(graph)
        self._init_weights()
        for _ in range(self.iterations):
            self._update_W()
            self._update_H()
            self._update_X()
            self._update_W_plus()
            self._update_H_plus()
            self._update_alpha_X()
            self._update_alpha_W()
            self._update_alpha_H()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate([self.W_plus, self.H_plus.T], axis=1)
        return embedding
