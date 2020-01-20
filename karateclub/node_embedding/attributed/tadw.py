import numpy as np
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

class TADW(Estimator):
    r"""An implementation of `"TADW" <https://arxiv.org/abs/1403.6652>`_
    from the ICDM '18 paper "Binarized Attributed Network Embedding Class". The 
    procedure first calculates the truncated SVD of an adjcacency - feature matrix
    product. This matrix is further decomposed by a binary CCD based technique. 
       
    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        svd_iterations (int): SVD iteration count. Default is 20.
        seed (int): Random seed. Default is 42.
        alpha (float): Kernel matrix inversion parameter. Default is 0.3. 
        iterations (int): Matrix decomoposition iterations. Default is 100.
    """
    def __init__(self, dimensions=32, reduction_dimensions=128, svd_iterations=20,
                 seed=42, alpha=0.3, iterations=100, lower_control=10**-15, lambd=1.0):

        self.dimensions = dimensions
        self.reduction_dimensions = reduction_dimensions
        self.svd_iterations = svd_iterations
        self.seed = seed
        self.alpha = alpha
        self.iterations = iterations
        self.lower_control = lower_control
        self.lambd = lambd

    def _create_target_matrix(self, graph):
        """
        Creating a normalized sparse adjacency matrix target.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded. 

        Return types:
            * **P** *(Scipy COO matrix) - The target matrix.    
        """
        weighted_graph = nx.Graph()
        for (u, v) in graph.edges():
            weighted_graph.add_edge(u, v, weight=1.0/graph.degree(u))
            weighted_graph.add_edge(v, u, weight=1.0/graph.degree(v))
        P = nx.adjacency_matrix(weighted_graph,
                                nodelist=range(graph.number_of_nodes()))
        return P

    def _init_weights(self):
        """
        Initialization of weights and loss container.
        """
        self.W = np.random.uniform(0, 1, (self.dimensions, self.A.shape[0]))
        self.H = np.random.uniform(0, 1, (self.dimensions, self.T.shape[0]))

    def _update_W(self):
        """
        A single update of the node embedding matrix.
        """
        H_T = self.T.transpose().dot(self.H.transpose()).transpose()
        grad = self.lambd*self.W -np.dot(H_T, self.A-np.dot(np.transpose(H_T), self.W))
        self.W = self.W-self.alpha * grad
        self.W[self.W < self.lower_control] = self.lower_control

    def _update_H(self):
        """
        A single update of the feature basis matrix.
        """
        inside = self.A - self.T.transpose().dot(np.transpose(self.W).dot(self.H).transpose())
        right = self.T.dot(np.dot(self.W, inside).transpose()).transpose()
        grad = self.lambd*self.H-right
        self.H = self.H-self.alpha * grad
        self.H[self.H < self.lower_control] = self.lower_control

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
        T = T.transpose()
        return T

    def fit(self, graph, X):
        """
        Fitting a TADW model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO or Numpy array)* - The matrix of node features.
        """
        self.A = self._create_target_matrix(graph)
        self.T = self._create_reduced_features(X)
        self._init_weights()
        for _ in range(self.iterations):
            self._update_W()
            self._update_H()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate([np.transpose(self.W), np.transpose(np.dot(self.H, self.T))], axis=1)
        return embedding

