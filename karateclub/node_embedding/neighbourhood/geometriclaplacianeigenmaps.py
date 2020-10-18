import numpy as np
import networkx as nx
import scipy.sparse as sps
from karateclub.estimator import Estimator

class GLEE(Estimator):
    r"""An implementation of `"Geometric Laplacian Eigenmaps" <https://arxiv.org/abs/1905.09763>`_
    from the Journal of Complex Networks '20 paper "GLEE: Geometric Laplacian Eigenmap Embedding".
    The procedure extracts the eigenvectors corresponding to the largest eigenvalues 
    of the graph Laplacian. These vectors are used as the node embedding.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions: int=128, seed: int=42):

        self.dimensions = dimensions
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Geometric Laplacian EigenMaps model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        number_of_nodes = graph.number_of_nodes()
        L_tilde = nx.normalized_laplacian_matrix(graph, nodelist=range(number_of_nodes))
        _, self._embedding = sps.linalg.eigsh(L_tilde, k=self.dimensions+1,
                                                  which='LM', return_eigenvectors=True)


    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
