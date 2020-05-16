import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from karateclub.estimator import Estimator
from sklearn.decomposition import TruncatedSVD

class FeatherNode(Estimator):
    r"""An implementation of `"TADW" <https://www.ijcai.org/Proceedings/15/Papers/299.pdf>`_
    from the IJCAI '15 paper "Network Representation Learning with Rich Text Information". The
    procedure uses the node attribute matrix with a factorization matrix to reproduce a power
    of the adjacency matrix to create representations.

    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        reduction_dimensions (int): SVD reduction dimensions. Default is 64.
        svd_iterations (int): SVD iteration count. Default is 20.
        seed (int): Random seed. Default is 42.
    """
    def __init__(self, dimensions=32, reduction_dimensions=64, svd_iterations=20,
                 seed=42):
        self.dimensions = dimensions
        self.reduction_dimensions = reduction_dimensions
        self.svd_iterations = svd_iterations
        self.seed = seed

    def _create_reduced_features(self, X):
        """
        Creating a dense reduced node feature matrix.

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

    def fit(self, graph, X):
        """
        Fitting a TADW model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO or Numpy array)* - The matrix of node features.
        """
        self._check_graph(graph)
        X = self._create_reduced_features(X)


