import math
import numpy as np
import networkx as nx
import scipy.sparse as sparse
from karateclub.estimator import Estimator

class FeatherGraph(Estimator):
    r"""An implementation of `"FEATHER-G" <https://arxiv.org/abs/2005.07959>`_
    from the paper "Characteristic Functions on Graphs: Birds of a Feather,
    from Statistical Descriptors to Parametric Models". The procedure
    uses characteristic functions of node features with random walk weights to describe
    node neighborhoods. These node level features are pooled by mean pooling to
    create graph level statistics.

    Args:
        order (int): Adjacency matrix powers. Default is 5.
        eval_points (int): Number of evaluation points. Default is 25.
        theta_max (int): Maximal evaluation point value. Default is 2.5.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, order=5, eval_points=25, theta_max=2.5, seed=42):
        self.order = order
        self.eval_points = eval_points
        self.theta_max = theta_max
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
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


    def _get_normalized_adjacency(self, graph):
        """
        Calculating the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **A_hat** *(SciPy array)* - The scattering matrix of the graph.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat


    def _create_node_feature_matrix(self, graph):
        """
        Calculating the node features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **X** *(NumPy array)* - The node features.
        """
        log_degree = np.array([math.log(graph.degree(node)+1) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        clustering_coefficient = np.array([nx.clustering(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        X = np.concatenate([log_degree, clustering_coefficient], axis=1)
        return X


    def _calculate_feather(self, graph):
        """
        Calculating the characteristic function features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **features** *(Numpy vector)* - The embedding of a single graph.
        """
        A_tilde = self._get_normalized_adjacency(graph)
        X = self._create_node_feature_matrix(graph)
        theta = np.linspace(0.01, self.theta_max, self.eval_points)
        X = np.outer(X, theta)
        X = X.reshape(graph.number_of_nodes(), -1)
        X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
        feature_blocks = []
        for _ in range(self.order):
            X = A_tilde.dot(X)
            feature_blocks.append(X)
        feature_blocks = np.concatenate(feature_blocks, axis=1)
        feature_blocks = np.mean(feature_blocks, axis=0)
        return feature_blocks


    def fit(self, graphs):
        """
        Fitting a graph level FEATHER model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_feather(graph) for graph in graphs]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
