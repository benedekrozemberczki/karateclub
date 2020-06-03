import numpy as np
import networkx as nx
from numpy.linalg import inv
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

class BANE(Estimator):
    r"""An implementation of `"BANE" <https://shiruipan.github.io/publication/yang-binarized-2018/yang-binarized-2018.pdf>`_
    from the ICDM '18 paper "Binarized Attributed Network Embedding Class". The 
    procedure first calculates the truncated SVD of an adjacency - feature matrix
    product. This matrix is further decomposed by a binary CCD based technique. 
       
    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        svd_iterations (int): SVD iteration count. Default is 20.
        seed (int): Random seed. Default is 42.
        alpha (float): Kernel matrix inversion parameter. Default is 0.3. 
        iterations (int): Matrix decomposition iterations. Default is 100.
        binarization_iterations (int): Binarization iterations. Default is 20.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions=32, svd_iterations=20, seed=42, alpha=0.3,
                 iterations=100, binarization_iterations=20):
        self.dimensions = dimensions
        self.svd_iterations = svd_iterations
        self.seed = seed
        self.alpha = alpha
        self.iterations = iterations
        self.binarization_iterations = binarization_iterations
        self.seed = seed

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

    def fit(self, graph, X):
        """
        Fitting a BANE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO or Numpy array)* - The matrix of node features.
        """
        self._set_seed()
        self._check_graph(graph)
        self._P = self._create_target_matrix(graph)
        self._X = X
        self._fit_base_SVD_model()
        self._binary_optimize()

    def _fit_base_SVD_model(self):
        """
        Reducing the dimensionality with SVD in the 1st step.
        """
        self._P = self._P.dot(self._X)
        self.model = TruncatedSVD(n_components=self.dimensions,
                                  n_iter=self.svd_iterations,
                                  random_state=self.seed)

        self.model.fit(self._P)
        self._P = self.model.fit_transform(self._P)

    def _update_G(self):
        """
        Updating the kernel matrix.
        """
        self._G = np.dot(self._B.transpose(), self._B)
        self._G = self._G + self.alpha*np.eye(self.dimensions)
        self._G = inv(self._G)
        self._G = self._G.dot(self._B.transpose()).dot(self._P)

    def _update_Q(self):
        """
        Updating the rescaled target matrix.
        """
        self._Q = self._G.dot(self._P.transpose()).transpose()

    def _update_B(self):
        """
        Updating the embedding matrix.
        """
        for _ in range(self.iterations):
            for d in range(self.dimensions):
                sel = [x for x in range(self.dimensions) if x != d]
                self._B[:, d] = self._Q[:, d]-self._B[:, sel].dot(self._G[sel, :]).dot(self._G[:, d]).transpose()
                self._B[:, d] = np.sign(self._B[:, d])

    def _binary_optimize(self):
        """
        Starting 2nd optimization phase with power iterations and CCD.
        """
        self._B = np.sign(np.random.normal(size=(self._P.shape[0], self.dimensions)))
        for _ in range(self.binarization_iterations):
            self._update_G()
            self._update_Q()
            self._update_B()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self._B
        return embedding
