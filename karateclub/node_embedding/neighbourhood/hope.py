import numpy as np
import networkx as nx
import scipy.sparse as sps
from karateclub.estimator import Estimator

class HOPE(Estimator):
    r"""An implementation of `"HOPE" <https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf>`_
    from the KDD '16 paper "Asymmetric Transitivity Preserving Graph Embedding". The procedure uses
    sparse SVD on the neighbourhood overlap matrix. The singular value rescaled left and right 
    singular vectors are used as the node embeddings after concatenation.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions=128, seed=42):

        self.dimensions = dimensions
        self.seed = seed


    def _create_target(self, graph):
        """
        Creating a target similarity matrix.
        """
        number_of_nodes = graph.number_of_nodes()
        A = nx.adjacency_matrix(graph, nodelist=range(number_of_nodes))
        S = sps.coo_matrix(A.dot(A), dtype=np.float32)
        return S

    def _do_rescaled_decomposition(self, S):
        """
        Decomposing the similarity matrix.
        """
        U, sigmas, Vt = sps.linalg.svds(S, k=int(self.dimensions/2))
        sigmas = np.diagflat(np.sqrt(sigmas))
        self._left_embedding = np.dot(U, sigmas)
        self._right_embedding = np.dot(Vt.T, sigmas)

    def fit(self, graph):
        """
        Fitting a HOPE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        S = self._create_target(graph)
        self._do_rescaled_decomposition(S)


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.concatenate([self._left_embedding, self._right_embedding], axis=1)
