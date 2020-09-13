import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker
from karateclub.estimator import Estimator

class DeepWalk(Estimator):
    r"""An implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, walk_number: int=10, walk_length: int=80, dimensions: int=128,
                 workers: int=4, window_size: int=5, epochs: int=1,
                 learning_rate: float=0.05, min_count: int=1, seed: int=42):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a DeepWalk model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)

        model = Word2Vec(walker.walks,
                         hs=1,
                         alpha=self.learning_rate,
                         iter=self.epochs,
                         size=self.dimensions,
                         window=self.window_size,
                         min_count=self.min_count,
                         workers=self.workers,
                         seed=self.seed)

        num_of_nodes = graph.number_of_nodes()
        self._embedding = [model[str(n)] for n in range(num_of_nodes)]


    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
