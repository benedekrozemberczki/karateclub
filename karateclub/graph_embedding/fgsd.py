import numpy as np
import networkx as nx
from tqdm import tqdm
from karateclub.estimator import Estimator

class FGSD(Estimator):
    r"""An implementation of `"FGSD" <https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs>`_
    from the NeurIPS '17 paper "Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurence matrix is decomposed in order
    to generate representations for the graphs.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 200.
        attributed (int): Number of nodes in diffusion. Default is 20.
    """
    def __init__(self, hist_bins=200, hist_range=20):

        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)

    def _calculate_fgsd(self, graph):
        L = nx.laplacian_matrix(graph).todense()
        fL = np.linalg.pinv(L)
        ones = np.ones(L.shape[0])
        S = np.outer(np.diag(fL), ones)+np.outer(ones, np.diag(fL))-2*fL
        hist, bin_edges = np.histogram(S.flatten(),bins=self.hist_bins,range=self.hist_range)
        return hist

    def fit(self, graphs):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._embedding = [self._calculate_fgsd(graph) for graph in tqdm(graphs)]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
