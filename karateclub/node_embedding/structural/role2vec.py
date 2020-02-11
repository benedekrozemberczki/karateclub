import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker
from karateclub.estimator import Estimator
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

class Role2Vec(Estimator):
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
        window_size (int): Matrix power order. Default is 2.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurences. Default is 1.
    """
    def __init__(self, walk_number=10, walk_length=80, dimensions=128, workers=4,
                 window_size=2, epochs=1, learning_rate=0.05, min_count=1, wl_iterations=2):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count

    def fit(self, graph):
        """
        Fitting a Role2vec model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        walker = RandomWalker(self.walk_number, self.walk_length)
        walker.do_walks(graph)
 
        hasher = WeisfeilerLehmanHashing(self.wl_iterations, attributed=False)
      
        node_features = hasher.get_node_features()

        documents = create_documents(walker.walks, node_features)

        model = Word2Vec(walker.walks,
                         hs=1,
                         alpha=self.learning_rate,
                         iter=self.epochs,
                         size=self.dimensions,
                         window=self.window_size,
                         min_count=self.min_count,
                         workers=self.workers)

        num_of_nodes = graph.number_of_nodes()
        self._embedding = [model[str(n)] for n in range(num_of_nodes)]


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
