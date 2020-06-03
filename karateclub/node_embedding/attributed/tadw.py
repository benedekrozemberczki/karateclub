import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from karateclub.estimator import Estimator
from sklearn.decomposition import TruncatedSVD

class TADW(Estimator):
    r"""An implementation of `"TADW" <https://www.ijcai.org/Proceedings/15/Papers/299.pdf>`_
    from the IJCAI '15 paper "Network Representation Learning with Rich Text Information". The
    procedure uses the node attribute matrix with a factorization matrix to reproduce a power
    of the adjacency matrix to create representations.

    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        reduction_dimensions (int): SVD reduction dimensions. Default is 64.
        svd_iterations (int): SVD iteration count. Default is 20.
        seed (int): Random seed. Default is 42.
        alpha (float): Learning rate. Default is 0.01. 
        iterations (int): Matrix decomposition iterations. Default is 10.
        lambd (float): Regularization coefficient. Default is 10.0.
    """
    def __init__(self, dimensions=32, reduction_dimensions=64, svd_iterations=20,
                 seed=42, alpha=0.01, iterations=10, lambd=10.0):
        self.dimensions = dimensions
        self.reduction_dimensions = reduction_dimensions
        self.svd_iterations = svd_iterations
        self.seed = seed
        self.alpha = alpha
        self.iterations = iterations
        self.lambd = lambd
        self.seed = seed

    def _create_target_matrix(self, graph):
        """
        Creating a normalized sparse adjacency matrix power target.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **A_tilde** *(Scipy COO matrix) - The target matrix.
        """
        weighted_graph = nx.Graph()
        for (u, v) in graph.edges():
            weighted_graph.add_edge(u, v, weight=1.0/graph.degree(u))
            weighted_graph.add_edge(v, u, weight=1.0/graph.degree(v))
        A_hat = nx.adjacency_matrix(weighted_graph,
                                    nodelist=range(graph.number_of_nodes()))

        A_tilde = A_hat.dot(A_hat)
        return coo_matrix(A_tilde)

    def _init_weights(self):
        """
        Initialization of weights and loss container.
        """
        self._W = np.random.uniform(0, 1, (self.dimensions, self._A.shape[0]))
        self._H = np.random.uniform(0, 1, (self.dimensions, self._T.shape[0]))

    def _update_W(self):
        """
        A single update of the node embedding matrix.
        """
        penalty = (self.lambd/np.linalg.norm(self._W))*self._W
        transformed_features = self._H.dot(self._T)
        scores = 0
        for i in range(self.dimensions):
            scores = scores + transformed_features[i,self._A.row] * self._W[i,self._A.col]
        score_matrix = coo_matrix((scores, (self._A.row, self._A.col)), shape=self._A.shape)
        diff_matrix = self._A-score_matrix
        main_grad = diff_matrix.dot(transformed_features.T).T/np.sum(np.square(scores))
        grad = penalty-main_grad
        self._W = self._W-self.alpha*grad

    def _update_H(self):
        """
        A single update of the feature basis matrix.
        """
        penalty = (self.lambd/np.linalg.norm(self._H))*self._H
        transformed_features = self._H.dot(self._T)
        scores = 0
        for i in range(self.dimensions):
            scores = scores + transformed_features[i,self._A.col] * self._W[i,self._A.row]
        score_matrix = coo_matrix((scores, (self._A.row, self._A.col)), shape=self._A.shape)
        diff_matrix = self._A-score_matrix
        main_grad = self._W.dot(diff_matrix.dot(self._T.T))/np.sum(np.square(scores))
        grad = penalty-main_grad
        self._H = self._H-self.alpha*grad

    def _create_reduced_features(self, X):
        """
        Creating a dense reduced node feature matrix.

        Arg types:
            * **X** *(Scipy COO or Numpy array)* - The wide feature matrix.

        Return types:
            * **T** *(Numpy array)* - The reduced feature matrix of nodes.
        """
        svd = TruncatedSVD(n_components=self.reduction_dimensions,
                           n_iter=self.svd_iterations,
                           random_state=self.seed)
        svd.fit(X)
        T = svd.transform(X)
        return T.T

    def fit(self, graph, X):
        """
        Fitting a TADW model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO or Numpy array)* - The matrix of node features.
        """
        self._set_seed()
        self._check_graph(graph)
        self._A = self._create_target_matrix(graph)
        self._T = self._create_reduced_features(X)
        self._init_weights()
        for _ in range(self.iterations):
            self._update_W()
            self._update_H()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate([np.transpose(self._W), np.transpose(np.dot(self._H, self._T))], axis=1)
        return embedding

