import random
import numpy as np
import networkx as nx
from karateclub.estimator import Estimator
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker

class SINE(Estimator):
    r"""An implementation of `"SINE" <https://arxiv.org/pdf/1810.06768.pdf>`_
    from the ICDM '18 paper "SINE: Scalable Incomplete Network Embedding". The 
    procedure implicitly factorizes a joint adjacency matrix power and feature matrix.
    The decomposition happens on truncated random walks and the adjacency matrix powers
    are pooled together.
       
    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, walk_number=10, walk_length=80, dimensions=128, workers=4,
                 window_size=5, epochs=1, learning_rate=0.05, min_count=1, seed=42):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed


    def _feature_transform(self, graph, X):
        features = {str(node): [] for node in graph.nodes()}
        nodes = X.row
        for i, node in enumerate(nodes):
            features[str(node)].append("feature_"+ str(X.col[i]))
        return features

    def _select_walklets(self):
        self._walklets = []
        for walk in self._walker.walks:
            for power in range(1, self.window_size+1): 
                for step in range(power+1):
                    neighbors = [n for i, n in enumerate(walk[step:]) if i % power == 0]
                    neighbors = [n for n in neighbors for _ in range(0, 3)]
                    neighbors = [random.choice(self._features[val]) if i % 3 == 1 and self._features[val] else val for i, val in enumerate(neighbors)]
                    self._walklets.append(neighbors)
        del self._walker
        
        
    def fit(self, graph, X):
        """
        Fitting a SINE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO array)* - The matrix of node features.
        """
        self._set_seed()
        self._check_graph(graph)
        self._walker = RandomWalker(self.walk_length, self.walk_number)
        self._walker.do_walks(graph)
        self._features = self._feature_transform(graph, X)
        self._select_walklets()

        model = Word2Vec(self._walklets,
                         hs=0,
                         alpha=self.learning_rate,
                         iter=self.epochs,
                         size=self.dimensions,
                         window=1,
                         min_count=self.min_count,
                         workers=self.workers,
                         seed=self.seed)

        self.embedding = np.array([model[str(n)] for n in range(graph.number_of_nodes())])

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self.embedding
        return embedding
