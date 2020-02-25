import numpy as np
import networkx as nx
from karateclub.estimator import Estimator

class NetLSD(Estimator):
    r"""An implementation of `"FGSD" <https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs>`_
    from the NeurIPS '17 paper "Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs".
    The procedure calculates the Moore-Penrose spectrum of the normalized Laplacian.
    Using this spectrum the histogram of the spectral features is used as a whole graph representation. 

    Args:
        hist_bins (int): Number of histogram bins. Default is 200.
        hist_range (int): Histogram range considered. Default is 20.
    """
    def __init__(self, hist_bins=200, hist_range=20):

        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)

    def _calculate_heat_kernel_trace(self, eivals):
        timescales = np.logspace(-2, 2, 250)
        nodes = eivals.shape[0]
        heat_kernel_trace = np.zeros(timescales.shape)
        for idx, t in enumerate(timescales):
            heat_kernel_trace[idx] = np.sum(np.exp(-t * eivals))
        heat_kernel_trace = heat_kernel_trace / nodes
        return heat_kernel_trace

    def _updown_linear_approx(self, eigvals_lower, eigvals_upper, nv):
        nal = len(eigvals_lower)
        nau = len(eigvals_upper)
        ret = np.zeros(nv)
        ret[:nal] = eigvals_lower
        ret[-nau:] = eigvals_upper
        ret[nal-1:-nau+1] = np.linspace(eigvals_lower[-1], eigvals_upper[0], nv-nal-nau+2)
        return ret

    def _eigenvalues_auto(self, mat):
        n_approx = 10
        nv = mat.shape[0]
        if 2*n_approx < nv:
            lo_eivals = sps.linalg.eigsh(mat, n_approx, which='SM', return_eigenvectors=False)[::-1]
            up_eivals = sps.linalg.eigsh(mat, n_approx, which='LM', return_eigenvectors=False)
            return self._updown_linear_approx(lo_eivals, up_eivals, nv)
        else:
            return sp.linalg.eigsh(mat, nv, which='SM', return_eigenvectors=False)


    def _calculate_netlsd(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        normalized_laplacian = sps.coo_matrix(nx.normalized_laplacian_matrix(graph, nodelist = range(graph.number_of_nodes())))
        eigen_values = self._calculate_eigenvalues_auto(normalized_laplacian)
        heat_kernel_trace = self._calculat_heat_kernel_trace(eigen_values)
        return heat_kernel_trace

    def fit(self, graphs):
        """
        Fitting a FGSD model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._embedding = [self._calculate_netlsd(graph) for graph in graphs]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
