import numpy as np
import networkx as nx
from numpy.linalg import inv
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

class TENE(object):
    """
    Enhanced Network Embedding with Text Information Abstract Class.
    For details see https://ieeexplore.ieee.org/document/8545577.
    """
    def __init__(self, X, T, args):
        """
        Set up model and weights.
        :param X: Adjacency target matrix. (Sparse Scipy matrix.)
        :param T: Feature matrix.
        :param args: Arguments object for model.
        """
        self.X = X
        self.T = T
        self.args = args
        self.init_weights()

    def init_weights(self):
        """
        Setup basis and feature matrices.
        """
        self.M = np.random.uniform(0, 1, (self.X.shape[0], self.args.dimensions))
        self.U = np.random.uniform(0, 1, (self.X.shape[0], self.args.dimensions))
        self.Q = np.random.uniform(0, 1, (self.X.shape[0], self.args.dimensions))
        self.V = np.random.uniform(0, 1, (self.T.shape[1], self.args.dimensions))
        self.C = np.random.uniform(0, 1, (self.args.dimensions, self.args.dimensions))

    def update_M(self):
        """
        Update node bases.
        """
        enum = self.X.dot(self.U)
        denom = self.M.dot(self.U.T.dot(self.U))
        self.M = np.multiply(self.M, enum/denom)
        self.M[self.M < self.args.lower_control] = self.args.lower_control

    def update_V(self):
        """
        Update node features.
        """
        enum = self.T.T.dot(self.Q)
        denom = self.V.dot(self.Q.T.dot(self.Q))
        self.V = np.multiply(self.V, enum/denom)
        self.V[self.V < self.args.lower_control] = self.args.lower_control

    def update_C(self):
        """
        Update transformation matrix.
        """
        enum = self.Q.T.dot(self.U)
        denom = self.C.dot(self.U.T.dot(self.U))
        self.C = np.multiply(self.C, enum/denom)
        self.C[self.C < self.args.lower_control] = self.args.lower_control

    def update_U(self):
        """
        Update features.
        """
        enum = self.X.T.dot(self.M)+self.args.alpha*self.Q.dot(self.C)
        denom = self.U.dot((self.M.T.dot(self.M)+self.args.alpha*self.C.T.dot(self.C)))
        self.U = np.multiply(self.U, enum/denom)
        self.U[self.U < self.args.lower_control] = self.args.lower_control

    def update_Q(self):
        """
        Update feature bases.
        """
        enum = self.args.alpha*self.U.dot(self.C.T)+self.args.beta*self.T.dot(self.V)
        denom = self.args.alpha*self.Q+self.args.beta*self.Q.dot(self.V.T.dot(self.V))
        self.Q = np.multiply(self.Q, enum/denom)
        self.Q[self.Q < self.args.lower_control] = self.args.lower_control

    def fit(self, graph, T):
        """
        Run updates.
        """
        
        for _ in tqdm(range(self.args.iterations)):
            self.update_M()
            self.update_V()
            self.update_C()
            self.update_U()
            self.update_Q()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate([self.M, self.Q], axis=1)
        return embedding
