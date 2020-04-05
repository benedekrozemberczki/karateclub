import numpy as np
import networkx as nx
from karateclub.utils.walker import RandomWalker
from karateclub.estimator import Estimator

class GEMSEC(Estimator):
    r"""An implementation of `"GEMSEC" <https://arxiv.org/abs/1802.03997>`_
    from the ASONAM '19 paper "GEMSEC: Graph Embedding with Self Clustering".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        window_size (int): Matrix power order. Default is 5.
        learning_rate (float): Gradient descent learning rate. Default is 0.05.
        clusters (int): Number of cluster centers. Default is 10.
        gamma (float): Clustering cost weight coefficient. Default is 0.01.
    """
    def __init__(self, walk_number=10, walk_length=80, dimensions=128,
                 window_size=5, learning_rate=0.01, clusters=10, gamma=0.01):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.clusters = clusters
        self.gamma = gamma

    def fit(self, graph):
        """
        Fitting a GEMSEC model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)



    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
