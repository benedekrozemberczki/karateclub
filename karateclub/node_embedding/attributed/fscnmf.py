import numpy as np
import networkx as nx
from scipy import sparse
from karateclub.estimator import Estimator

class FSCNMF(Estimator):
    r"""An implementation of `"TENE" <https://ieeexplore.ieee.org/document/8545577>`_
    from the ICPR '18 paper "Enhanced Network Embedding with Text Information". The 
    procedure jointly factorizes the adjacency and node feature matrices using alternating
    least squares.
       
    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        lower_control (float): Embedding score minimal value. Default is 10**-15.
        alpha (float): Adjacency matrix regularizer coefficient. Default is 0.1. 
        beta (float): Feature matrix regularizer coefficient. Default is 0.1.
        iterations (int): ALS iterations. Default is 200.
    """
    def __init__(self, dimensions=32, alpha_1=1.0, alpha_2=1.0, lower_control=10**-15,
                 beta_1=1.0, beta_2=1.0):



    def init_weights(self):
        """
        Setup basis and feature matrices.
        """
        self.U = np.random.uniform(0, 1, (self.A.shape[0], self.args.dimensions))
        self.V = np.random.uniform(0, 1, (self.dimensions, self.X.shape[1]))
        self.B_1 = np.random.uniform(0, 1, (self.A.shape[0], self.dimensions))
        self.B_2 = np.random.uniform(0, 1, (self.dimensions, self.A.shape[0]))

    def update_B1(self):
        """
        Update node bases.
        """
        simi_term = self.A.dot(np.transpose(self.B_2)) + self.alpha_1*self.U
        regul = self.alpha_1*np.eye(self.dimensions)
        regul = regul + self.alpha_2*np.eye(self.dimensions)
        covar_term = inv(np.dot(self.B_2, np.transpose(self.B_2))+regul)
        self.B_1 = np.dot(simi_term, covar_term)
        self.B_1[self.B_1 < self.lower_control] = self.lower_control

    def update_B2(self):
        """
        Update node features.
        """
        to_inv = np.dot(np.transpose(self.B_1), self.B_1)
        to_inv = to_inv + self.args.alpha_3*np.eye(self.args.dimensions)
        covar_term = inv(to_inv)
        simi_term = self.A.dot(self.B_1).transpose()
        self.B_2 = covar_term.dot(simi_term)
        self.B_2[self.B_2 < self.args.lower_control] = self.args.lower_control

    def update_U(self):
        """
        Update feature basis.
        """
        simi_term = self.X.dot(np.transpose(self.V)) + self.args.beta_1*self.B_1
        regul = self.args.beta_1*np.eye(self.args.dimensions)
        regul = regul + self.args.beta_2*np.eye(self.args.dimensions)
        covar_term = inv(np.dot(self.V, np.transpose(self.V))+regul)
        self.U = np.dot(simi_term, covar_term)
        self.U[self.U < self.args.lower_control] = self.args.lower_control

    def update_V(self):
        """
        Update features.
        """
        to_inv = np.dot(np.transpose(self.U), self.U)
        to_inv = to_inv + self.args.beta_3*np.eye(self.args.dimensions)
        covar_term = inv(to_inv)
        simi_term = self.X.transpose().dot(self.U)
        self.V = np.dot(simi_term, covar_term).transpose()
        self.V[self.V < self.args.lower_control] = self.args.lower_control


    def fit(self, graph, X):
        """
        Fitting a TENE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **T** *(Scipy COO or Numpy array)* - The matrix of node features.
        """
        self.A = self._create_base_matrix(graph)
        self.X = X
        self._init_weights()
        for _ in range(self.iterations):
            self._update_B1()
            self._update_B2()
            self._update_U()
            self._update_V()



    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self.gamma*self.B_1+(1-self.gamma)*self.U
        return embedding
