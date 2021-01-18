import numpy as np
import networkx as nx
import scipy.sparse as sps
from karateclub.estimator import Estimator

class SocioDim(Estimator):
    r"""An implementation of `"SocioDim" <https://dl.acm.org/doi/abs/10.1145/1557019.1557109>`_
    from the KDD '09 paper "Relational Learning via Latent Social Dimensions".
    The procedure extracts the eigenvectors corresponding to the largest eigenvalues 
    of the graph modularity matrix. These vectors are used as the node embedding.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions: int=128, seed: int=42):

        self.dimensions = dimensions
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Social Dimensions model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        number_of_nodes = graph.number_of_nodes()
        L_tilde = nx.modularity_matrix(graph, nodelist=range(number_of_nodes))
        _, self._embedding = sps.linalg.eigsh(L_tilde, k=self.dimensions,
                                              which='LM', return_eigenvectors=True)


    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
