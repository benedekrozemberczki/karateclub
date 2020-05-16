import math
import numpy as np
import networkx as nx
import scipy.stats.mstats
import scipy.sparse as sparse
from karateclub.estimator import Estimator

class FeatherG(Estimator):
    r"""An implementation of `"GeoScattering" <http://proceedings.mlr.press/v97/gao19e.html>`_
    from the ICML '19 paper "Geometric Scattering for Graph Data Analysis". The procedure
    uses scattering with wavelet transforms to create graph spectral descriptors. Moments of the
    wavelet transformed features are used as graph level features for the embedding. 

    Args:
        order (int): Adjacency matrix powers. Default is 4.
        moments (int): Unnormalized moments considered. Default is 4.
    """
    def __init__(self, order=4, moments=4):
        self.order = order
        self.moments = moments


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
        clustering_coefficient = np.array([nx.clustering(graph,node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
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
        A_hat = self._get_normalized_adjacency(graph)
        X = self._create_node_feature_matrix(graph)

        return features


    def fit(self, graphs):
        """
        Fitting a FEATHER graph level model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._check_graphs(graphs)
        self._embedding = [self._calculate_feather(graph) for graph in graphs]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
