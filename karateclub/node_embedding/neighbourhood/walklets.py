import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker

class Walklets(object):
    r"""An implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 32.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 4.
        epochs (int): Number of epochs.
        learning_rate (float): HogWild! learning rate.
        min_count (int): Minimal count of node occurences.
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


    def _select_walklets(self, walks, power):
        walklets = []
        for walk in walks:
            for step in range(power+1):
                neighbors = [n for i, n in enumerate(walk[step:]) if i % power == 0]
                walklets.append(neighbors)
        return walklets

    def fit(self, graph):
        """
        Fitting a DeepWalk model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        walker = RandomWalker(self.walk_number, self.walk_length)
        walker.do_walks(graph)
        num_of_nodes = graph.number_of_nodes()
        self._embedding = []
        for power in range(1, self.window_size+1):
            walklets = self._select_walklets(walker.walks, power)
            model = Word2Vec(walklets,
                             hs=0,
                             alpha=self.learning_rate,
                             iter=self.epochs,
                             size=self.dimensions,
                             window=1,
                             min_count=self.min_count,
                             workers=self.workers)

            embedding = np.array([model[str(n)] for n in range(num_of_nodes)])
            self._embedding.append(embedding)


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.concatenate(self._embedding,axis=1)
