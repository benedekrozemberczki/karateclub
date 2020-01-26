import numpy as np
import networkx as nx
from karateclub.estimator import Estimator

class SF(Estimator):
    r"""An implementation of `"SF" <A Simple Baseline Algorithm for Graph Classification>`_
    from the NeurIPS Relational Representation Learning Workshop '18 paper A Simple Baseline Algorithm for Graph Classification".
    The procedure calculates the k lowest egeinvalues of the normalized Laplacian.
    If the graph has a lowe umber of eigenvalues than k the representation is padded with zeros.

    Args:
        dimensions (int): Number of lowest eigenvalues. Defauls is 128.
    """
    def __init__(self, dimensions=128):

        self.dimensions = dimensions

    def _calculate_sf(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        L = nx.normalized_laplacian_matrix(graph).todense()
        fL = np.linalg.pinv(L)
        ones = np.ones(L.shape[0])
        S = np.outer(np.diag(fL), ones)+np.outer(ones, np.diag(fL))-2*fL
        hist, bin_edges = np.histogram(S.flatten(),
                                       bins=self.hist_bins,
                                       range=self.hist_range)
        return hist

    def fit(self, graphs):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._embedding = [self._calculate_fgsd(graph) for graph in graphs]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
