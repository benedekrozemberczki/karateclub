import scipy
import numpy as np
import networkx as nx
from typing import Union
from scipy.sparse import coo_matrix
from karateclub.estimator import Estimator
from sklearn.decomposition import TruncatedSVD

class FeatherNode(Estimator):
    r"""An implementation of `"FEATHER-N" <https://arxiv.org/abs/2005.07959>`_
    from the CIKM '20 paper "Characteristic Functions on Graphs: Birds of a Feather,
    from Statistical Descriptors to Parametric Models". The procedure
    uses characteristic functions of node features with random walk weights to describe
    node neighborhoods.

    Args:
        reduction_dimensions (int): SVD reduction dimensions. Default is 64.
        svd_iterations (int): SVD iteration count. Default is 20.
        theta_max (float): Maximal evaluation point. Default is 2.5.
        eval_points (int): Number of characteristic function evaluation points. Default is 25.
        order (int): Scale - number of adjacency matrix powers. Default is 5.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, reduction_dimensions: int=64, svd_iterations: int=20,
                 theta_max: float=2.5, eval_points: int=25, order: int=5, seed: int=42):

        self.reduction_dimensions = reduction_dimensions
        self.svd_iterations = svd_iterations
        self.seed = seed
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        self.seed = seed

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
        D_inverse = scipy.sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


    def _create_A_tilde(self, graph):
        """
        Creating a sparse normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        Return types:
            * **A_tilde** *(Scipy array)* - The normalized adjacency matrix.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_tilde = D_inverse.dot(A)
        return A_tilde


    def _reduce_dimensions(self, X):
        """
        Using Truncated SVD.

        Arg types:
            * **X** *(Scipy COO or Numpy array)* - The wide feature matrix.

        Return types:
            * **X** *(Numpy array)* - The reduced feature matrix of nodes.
        """
        svd = TruncatedSVD(n_components=self.reduction_dimensions,
                           n_iter=self.svd_iterations,
                           random_state=self.seed)
        svd.fit(X)
        X = svd.transform(X)
        return X


    def _create_reduced_features(self, X):
        """
        Creating a dense reduced node feature matrix.

        Arg types:
            * **X** *(Scipy COO or Numpy array)* - The wide feature matrix.

        Return types:
            * **X** *(Numpy array)* - The reduced feature matrix of nodes.
        """
        if scipy.sparse.issparse(X):
            X = self._reduce_dimensions(X)
        elif (type(X) is np.ndarray) and (X.shape[1] > self.reduction_dimensions):
            X = self._reduce_dimensions(X)
        else:
            X = X
        return X

    def fit(self, graph: nx.classes.graph.Graph, X: Union[np.array, coo_matrix]):
        """
        Fitting a FEATHER-N model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO or Numpy array)* - The matrix of node features.
        """
        self._set_seed()
        self._check_graph(graph)
        X = self._create_reduced_features(X)
        A_tilde = self._create_A_tilde(graph)
        theta = np.linspace(0.01, self.theta_max, self.eval_points)
        X = np.outer(X, theta)
        X = X.reshape(graph.number_of_nodes(), -1)
        X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
        self._feature_blocks = []
        for _ in range(self.order):
            X = A_tilde.dot(X)
            self._feature_blocks.append(X)
        self._feature_blocks = np.concatenate(self._feature_blocks, axis=1)


    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._feature_blocks
