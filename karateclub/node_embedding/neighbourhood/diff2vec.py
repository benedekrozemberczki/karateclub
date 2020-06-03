import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from karateclub.utils.diffuser import EulerianDiffuser
from karateclub.estimator import Estimator

class Diff2Vec(Estimator):
    r"""An implementation of `"Diff2Vec" <http://homepages.inf.ed.ac.uk/s1668259/papers/sequence.pdf>`_
    from the CompleNet '18 paper "Diff2Vec: Fast Sequence Based Embedding with Diffusion Graphs".
    The procedure creates diffusion trees from every source node in the graph. These graphs are linearized
    by a directed Eulerian walk, the walks are used for running the skip-gram algorithm the learn node
    level neighbourhood based embeddings.

    Args:
        diffusion_number (int): Number of diffusions. Default is 10.
        diffusion_cover (int): Number of nodes in diffusion. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, diffusion_number=10, diffusion_cover=80, dimensions=128, workers=4,
                 window_size=5, epochs=1, learning_rate=0.05, min_count=1, seed=42):

        self.diffusion_number = diffusion_number
        self.diffusion_cover = diffusion_cover
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph):
        """
        Fitting a Diff2Vec model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        diffuser = EulerianDiffuser(self.diffusion_number, self.diffusion_cover)
        diffuser.do_diffusions(graph)

        model = Word2Vec(diffuser.diffusions,
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


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
