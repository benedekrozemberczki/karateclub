"""Local Degree Profile based embedding."""

import numpy as np
from karateclub.estimator import Estimator

class LDP(Estimator):
    r"""An implementation of `"LDP" <https://arxiv.org/abs/1811.03508>`_ from the
    ICLR Representation Learning on Graphs and Manifolds Workshop '19 paper "A
    Simple Yet Effective Baseline for Non-Attributed Graph Classification". The
    procedure calculates histograms of degree profiles. These concatenated
    histograms form the graph representations.

    Args:
        bins (int): Number of histogram bins. Default is 32.
    """
    def __init__(self, bins: int=32):
        self.bins = bins

    def _calculate_ldp(self, graph):
        """
        Calculating the local degree profile features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of a single graph.
        """
        degrees = np.log(np.array([graph.degree[n] for n in range(graph.number_of_nodes())]))
        features = []
        for n in range(graph.number_of_nodes()):
            nebs = [neb for neb in graph.neighbors(n)]
            degs = degrees[nebs]

            features.append([np.min(degs),
                             np.max(degs),
                             np.std(degs),
                             np.mean(degs)])

        features = np.concatenate([degrees.reshape(-1, 1),
                                   np.array(features)], axis=1)
        embedding = []
        for i in range(features.shape[1]):
            x = features[:, i]
            emb = np.histogram(x, bins=self.bins, range=(0.0, 10.0))[0]
            embedding.append(emb)
        embedding = np.concatenate(embedding).reshape(-1)
        return embedding

    def fit(self, graphs):
        """
        Fitting an LDP model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._check_graphs(graphs)
        self._embedding = [self._calculate_ldp(graph) for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
