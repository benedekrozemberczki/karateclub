import random
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class ASNE(Estimator):
    r"""An implementation of `"ASNE" <https://arxiv.org/abs/1705.04969>`_
    from the TKDE '18 paper "Attributed Social Network Embedding". The 
    procedure implicitly factorizes a concatenated adjacency matrix and feature matrix.
       
    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        epochs (int): Number of epochs. Default is 100.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions: int=128, workers: int=4,
                 epochs: int=100, down_sampling: float=0.0001,
                 learning_rate: float=0.05, min_count: int=1, seed: int=42):

        self.dimensions = dimensions
        self.workers = workers
        self.epochs = epochs
        self.down_sampling = down_sampling
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed


    def _feature_transform(self, graph, X):
        features = {node: ["neb_" + str(neb) for neb in graph.neighbors(node)] for node in graph.nodes()}
        nodes = X.row
        for i, node in enumerate(nodes):
            features[node].append("feature_"+ str(X.col[i]))
        return features
        
    def fit(self, graph: nx.classes.graph.Graph, X: coo_matrix):
        """
        Fitting an ASNE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO array)* - The matrix of node features.
        """
        self._set_seed()
        self._check_graph(graph)
        features = self._feature_transform(graph, X)
        documents = [TaggedDocument(words=features[node], tags=[str(node)]) for node in range(len(features))]

        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        iter=self.epochs,
                        alpha=self.learning_rate,
                        seed=self.seed)

        self._embedding = np.array([model.docvecs[str(i)] for i, _ in enumerate(documents)])



    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self._embedding
        return embedding
