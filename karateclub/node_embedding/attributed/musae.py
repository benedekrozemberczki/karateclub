import random
import numpy as np
import networkx as nx
from karateclub.estimator import Estimator
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker

class MUSAE(Estimator):
    r"""An implementation of `"MUSAE" <https://arxiv.org/pdf/1810.06768.pdf>`_
    from the Arxiv '19 paper "SINE: Scalable Incomplete Network Embedding". The 
    procedure implicitly factorizes a joint adjacency matrix power and feature matrix.
    The decomposition happens on truncated random walks and the adjacency matrix powers
    are pooled together.
       
    Args:
        walk_number (int): Number of random walks. Default is 5.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 32.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 3.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurences. Default is 1.
    """
    def __init__(self, walk_number=10, walk_length=80, dimensions=32, workers=4,
                 window_size=4, epochs=1, learning_rate=0.05, min_count=1):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count


    def _feature_transform(self, graph, X):
        features = {str(node): [] for node in graph.nodes()}
        nodes = X.row
        for i, node in enumerate(nodes):
            features[str(node)].append("feature_"+ str(X.col[i]))
        return features


        
    def fit(self, graph, X):
        """
        Fitting a SINE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO array)* - The matrix of node features.
        """
        self.walker = RandomWalker(self.walk_length, self.walk_number)
        self.walker.do_walks(graph)
        self.features = self._feature_transform(graph, X)


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = None
        return embedding
