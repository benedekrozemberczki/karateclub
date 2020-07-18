import math
import numpy as np
import networkx as nx
from typing import List
import scipy.stats.mstats
import scipy.sparse as sparse
from karateclub.estimator import Estimator

class GeoScattering(Estimator):
    r"""An implementation of `"GeoScattering" <http://proceedings.mlr.press/v97/gao19e.html>`_
    from the ICML '19 paper "Geometric Scattering for Graph Data Analysis". The procedure
    uses scattering with wavelet transforms to create graph spectral descriptors. Moments of the
    wavelet transformed features are used as graph level features for the embedding.

    Args:
        order (int): Adjacency matrix powers. Default is 4.
        moments (int): Unnormalized moments considered. Default is 4.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, order: int=4, moments: int=4, seed: int=42):
        self.order = order
        self.moments = moments
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
        A_hat = sparse.identity(graph.number_of_nodes()) + D_inverse.dot(A)
        A_hat = 0.5*A_hat
        return A_hat


    def _calculate_wavelets(self, A_hat):
        """
        Calculating the wavelets of a normalized self-looped adjacency matrix.

        Arg types:
            * **A_hat** *(SciPy array)* - The normalized adjacency matrix.

        Return types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
        """
        Psi = [A_hat.power(2**power) - A_hat.power(2**(power+1)) for power in range(self.order+1)]
        return Psi


    def _create_node_feature_matrix(self, graph):
        """
        Calculating the node features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **X** *(NumPy array)* - The node features.
        """
        log_degree = np.array([math.log(graph.degree(node)+1) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        eccentricity = np.array([nx.eccentricity(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        clustering_coefficient = np.array([nx.clustering(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        X = np.concatenate([log_degree, eccentricity, clustering_coefficient], axis=1)
        return X


    def _get_zero_order_features(self, X):
        """
        Calculating the zero-th order graph features.

        Arg types:
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The zero-th order graph features.
        """
        features = []
        X = np.abs(X)
        for col in range(X.shape[1]):
            x = np.abs(X[:, col])
            for power in range(1, self.order+1):
                features.append(np.sum(np.power(x, power)))
        features = np.array(features).reshape(-1)
        return features


    def _get_first_order_features(self, Psi, X):
        """
        Calculating the first order graph features.

        Arg types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The first order graph features.
        """
        features = []
        X = np.abs(X)
        for col in range(X.shape[1]):
            x = np.abs(X[:, col])
            for psi in Psi:
                filtered_x = psi.dot(x)
                for q in range(1, self.moments):
                    features.append(np.sum(np.power(np.abs(filtered_x), q)))
        features = np.array(features).reshape(-1)
        return features


    def _get_second_order_features(self, Psi, X):
        """
        Calculating the second order graph features.

        Arg types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The second order graph features.
        """
        features = []
        X = np.abs(X)
        for col in range(X.shape[1]):
            x = np.abs(X[:, col])
            for i in range(self.order-1):
                for j in range(i+1, self.order):
                    psi_j = Psi[i]
                    psi_j_prime = Psi[j]     
                    filtered_x = np.abs(psi_j_prime.dot(np.abs(psi_j.dot(x))))
                    for q in range(1, self.moments):
                        features.append(np.sum(np.power(np.abs(filtered_x), q)))

        features = np.array(features).reshape(-1)
        return features


    def _calculate_geoscattering(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **features** *(Numpy vector)* - The embedding of a single graph.
        """
        A_hat = self._get_normalized_adjacency(graph)
        Psi = self._calculate_wavelets(A_hat)
        X = self._create_node_feature_matrix(graph)
        zero_order_features = self._get_zero_order_features(X)
        first_order_features = self._get_first_order_features(Psi, X)
        second_order_features = self._get_second_order_features(Psi, X)
        features = np.concatenate([zero_order_features, first_order_features, second_order_features], axis=0)
        return features


    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a Geometric-Scattering model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_geoscattering(graph) for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
