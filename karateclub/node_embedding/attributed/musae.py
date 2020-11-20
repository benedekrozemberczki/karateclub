import random
import numpy as np
import networkx as nx
from typing import Union
from scipy.sparse import coo_matrix
from karateclub.estimator import Estimator
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from karateclub.utils.walker import RandomWalker

class MUSAE(Estimator):
    r"""An implementation of `"MUSAE" <https://arxiv.org/abs/1909.13021>`_
    from the Arxiv '19 paper "MUSAE: Multi-Scale Attributed Node Embedding". The
    procedure does attributed random walks to approximate the adjacency matrix power
    node feature matrix products. The matrices are decomposed implicitly by a Skip-Gram
    style optimizer. The individual embeddings are concatenated together to form a 
    multi-scale attributed node embedding. This way the feature distributions at different scales
    are separable.
       
    Args:
        walk_number (int): Number of random walks. Default is 5.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 32.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 3.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        down_sampling (float): Down sampling rate in the corpus. Default is 0.0001.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, walk_number=5, walk_length=80, dimensions=32, workers=4,
                 window_size=3, epochs=5, learning_rate=0.05, down_sampling=0.0001,
                 min_count=1, seed=42):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.down_sampling = down_sampling
        self.min_count = min_count
        self.seed = seed


    def _feature_transform(self, graph, X):
        features = {str(node): [] for node in graph.nodes()}
        nodes = X.row
        for i, node in enumerate(nodes):
            features[str(node)].append("feature_"+ str(X.col[i]))
        return features


    def _create_single_embedding(self, document_collections):
        model = Doc2Vec(document_collections,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        alpha=self.learning_rate,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        iter=self.epochs,
                        seed=self.seed)

        emb = np.array([model.docvecs[str(n)] for n in range(self.graph.number_of_nodes())])
        return emb


    def _create_documents(self, features):
        features_out = [TaggedDocument(words=[str(feat) for feat_elems in feature_set for feat in feat_elems], tags = [str(node)]) for node, feature_set in features.items()]
        return features_out

    def _setup_musae_features(self, approximation):
        features = {str(node): [] for node in self.graph.nodes()}
        for walk in self._walker.walks:
            for i in range(len(walk)-approximation):
                source = walk[i]
                target = walk[i+approximation]
                features[str(source)].append(self.features[str(target)] + [str(target)])
                features[str(target)].append(self.features[str(source)] + [str(source)])

        return self._create_documents(features)

    def _learn_musae_embedding(self):

        for approximation in range(self.window_size):

            features = self._setup_musae_features(approximation+1)
            embedding = self._create_single_embedding(features)
            self.embeddings.append(embedding)

    def _create_base_docs(self):
        features_out = [TaggedDocument(words=[str(feature) for feature in features], tags = [str(node)]) for node, features in self.features.items()]
        return features_out 

    def fit(self, graph: nx.classes.graph.Graph, X: Union[np.array, coo_matrix]):
        """
        Fitting a MUSAE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO array)* - The binary matrix of node features.
        """
        self._set_seed()
        self._check_graph(graph)
        self.graph = graph
        self._walker = RandomWalker(self.walk_length, self.walk_number)
        self._walker.do_walks(graph)
        self.features = self._feature_transform(graph, X)
        self._base_docs = self._create_base_docs()
        self.embeddings = [self._create_single_embedding(self._base_docs)]
        self._learn_musae_embedding()

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate(self.embeddings, axis=1)
        return embedding
