import numpy as np
import networkx as nx
from scipy import sparse
from karateclub.estimator import Estimator

class NMFADMM(Estimator):
    r"""An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.

    Args:
        dimensions (int): Number of individual embedding dimensions. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 5
        seed (int): SVD random seed. Default is 42.
    """

    def _init_weights(self):
        """
        Initializing model weights.
        """
        self.W = np.random.uniform(-0.1, 0.1, (self.V.shape[0], self.args.dimensions))
        self.H = np.random.uniform(-0.1, 0.1, (self.args.dimensions, self.V.shape[1]))
        X_i, Y_i = sp.nonzero(self.V)
        scores = self.W[X_i]*self.H[:, Y_i].T+np.random.uniform(0, 1, (self.args.dimensions, ))
        values = np.sum(scores, axis=-1)
        self.X = sp.sparse.coo_matrix((values, (X_i, Y_i)), shape=self.V.shape)
        self.W_plus = np.random.uniform(0, 0.1, (self.V.shape[0], self.args.dimensions))
        self.H_plus = np.random.uniform(0, 0.1, (self.args.dimensions, self.V.shape[1]))
        self.alpha_X = sp.sparse.coo_matrix(([0]*len(values), (X_i, Y_i)), shape=self.V.shape)
        self.alpha_W = np.zeros(self.W.shape)
        self.alpha_H = np.zeros(self.H.shape)

    def _update_W(self):
        """
        Updating user matrix.
        """
        left = np.linalg.pinv(self.H.dot(self.H.T)+np.eye(self.args.dimensions))
        right_1 = self.X.dot(self.H.T).T+self.W_plus.T
        right_2 = (1.0/self.args.rho)*(self.alpha_X.dot(self.H.T).T-self.alpha_W.T)
        self.W = left.dot(right_1+right_2).T

    def _update_H(self):
        """
        Updating item matrix.
        """
        left = np.linalg.pinv(self.W.T.dot(self.W)+np.eye(self.args.dimensions))
        right_1 = self.X.T.dot(self.W).T+self.H_plus
        right_2 = (1.0/self.args.rho)*(self.alpha_X.T.dot(self.W).T-self.alpha_H)
        self.H = left.dot(right_1+right_2)

    def _update_X(self):
        """
        Updating user-item matrix.
        """
        iX, iY = sp.nonzero(self.V)
        values = np.sum(self.W[iX]*self.H[:, iY].T, axis=-1)
        scores = sp.sparse.coo_matrix((values-1, (iX, iY)), shape=self.V.shape)
        left = self.args.rho*scores-self.alpha_X
        right = (left.power(2)+4.0*self.args.rho*self.V).power(0.5)
        self.X = (left+right)/(2*self.args.rho)

    def _update_W_plus(self):
        """
        Updating positive primal user factors.
        """
        self.W_plus = np.maximum(self.W+(1/self.args.rho)*self.alpha_W, 0)

    def _update_H_plus(self):
        """
        Updating positive primal item factors.
        """
        self.H_plus = np.maximum(self.H+(1/self.args.rho)*self.alpha_H, 0)

    def _update_alpha_X(self):
        """
        Updating target matrix dual.
        """
        iX, iY = sp.nonzero(self.V)
        values = np.sum(self.W[iX]*self.H[:, iY].T, axis=-1)
        scores = sp.sparse.coo_matrix((values, (iX, iY)), shape=self.V.shape)
        self.alpha_X = self.alpha_X+self.args.rho*(self.X-scores)

    def _update_alpha_W(self):
        """
        Updating user dual factors.
        """
        self.alpha_W = self.alpha_W+self.args.rho*(self.W-self.W_plus)

    def _update_alpha_H(self):
        """
        Updating item dual factors.
        """
        self.alpha_H = self.alpha_H+self.args.rho*(self.H-self.H_plus)

    def _optimize(self):
        """
        Running ADMM steps.
        """
        for i in tqdm(range(self.args.epochs)):
            self.update_W()
            self.update_H()
            self.update_X()
            self.update_W_plus()
            self.update_H_plus()
            self.update_alpha_X()
            self.update_alpha_W()
            self.update_alpha_H()

