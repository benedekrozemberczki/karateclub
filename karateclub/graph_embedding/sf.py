import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from karateclub.estimator import Estimator

class SF(Estimator):
    r"""An implementation of `"SF" <A Simple Baseline Algorithm for Graph Classification>`_
    from the NeurIPS Relational Representation Learning Workshop '18 paper "A Simple Baseline Algorithm for Graph Classification".
    The procedure calculates the k lowest egeinvalues of the normalized Laplacian.
    If the graph has a lower number of eigenvalues than k the representation is padded with zeros.

    Args:
        dimensions (int): Number of lowest eigenvalues. Default is 128.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions=128, seed=42):
        self.dimensions = dimensions
        self.seed = seed

    def _calculate_sf(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of a single graph.
        """
        number_of_nodes = graph.number_of_nodes()
        L_tilde = nx.normalized_laplacian_matrix(graph, nodelist=range(number_of_nodes))
        if number_of_nodes <= self.dimensions:
            embedding = eigsh(L_tilde, k=number_of_nodes-1, which='LM',
                              ncv=10*self.dimensions, return_eigenvectors=False)

            shape_diff = self.dimensions - embedding.shape[0] - 1
            embedding = np.pad(embedding, (1, shape_diff), 'constant', constant_values=0)
        else:
            embedding = eigsh(L_tilde, k=self.dimensions, which='LM',
                              ncv=10*self.dimensions, return_eigenvectors=False)
        return embedding

    def fit(self, graphs):
        """
        Fitting an SF model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_sf(graph) for graph in graphs]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
