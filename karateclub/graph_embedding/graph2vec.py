import numpy as np
import networkx as nx
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

class Graph2Vec(Estimator):
    r"""An implementation of `"Graph2Vec" <https://arxiv.org/abs/1707.05005>`_
    from the MLGWorkshop '17 paper "Graph2Vec: Learning Distributed Representations of Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurence matrix is decomposed in order
    to generate representations for the graphs.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
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

    def fit(self, graphs):
        """
        Fitting a Graph2Vec model.

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
