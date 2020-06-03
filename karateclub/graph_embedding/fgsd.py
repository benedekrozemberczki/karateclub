import numpy as np
import networkx as nx
from karateclub.estimator import Estimator

class FGSD(Estimator):
    r"""An implementation of `"FGSD" <https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs>`_
    from the NeurIPS '17 paper "Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs".
    The procedure calculates the Moore-Penrose spectrum of the normalized Laplacian.
    Using this spectrum the histogram of the spectral features is used as a whole graph representation. 

    Args:
        hist_bins (int): Number of histogram bins. Default is 200.
        hist_range (int): Histogram range considered. Default is 20.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, hist_bins=200, hist_range=20, seed=42):

        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)
        self.seed = seed

    def _calculate_fgsd(self, graph):
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
        Fitting a FGSD model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_fgsd(graph) for graph in graphs]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
