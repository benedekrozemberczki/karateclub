import numpy as np
import networkx as nx
from scipy import sparse
from karateclub.estimator import Estimator

class FGSD(Estimator):
    r"""An implementation of `"FGSD" <https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs>`_
    from the NeurIPS '17 paper "Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurence matrix is decomposed in order
    to generate representations for the graphs.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Number of nodes in diffusion. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Deefault is 0.025.
        min_count (int): Minimal count of graph feature occurences. Default is 5.
    """
    def __init__(self, wl_iterations=2, attributed=False, dimensions=128, workers=4,
                 down_sampling=0.0001, epochs=10, learning_rate=0.025, min_count=5):

        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count

    def _calculate_fgsd(self, graph):
        pass

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
