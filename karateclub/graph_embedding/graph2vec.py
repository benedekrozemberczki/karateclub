import numpy as np
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from karateclub.estimator import Estimator

class Graph2Vec(Estimator):
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
        epochs (int): Number of epochs.
        learning_rate (float): HogWild! learning rate.
        min_count (int): Minimal count of node occurences.
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

    def fit(self, graphs):
        """
        Fitting a Diff2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        documents = [WeisfeilerLehmanHashing(graph, self.wl_iterations, self.attributed) for graph in graphs]
        documents = [TaggedDocument(words=doc.extracted_features, tags=[str(i)]) for i, doc in enumerate(documents)]

        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        epochs=self.epochs,
                        alpha=self.learning_rate)

        self._embedding = [model.docvecs[str(i)] for i, _ in enumerate(documents)]


    def get_embedding(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
