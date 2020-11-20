import numpy as np
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.walker import RandomWalker
from karateclub.estimator import Estimator
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

class Role2Vec(Estimator):
    r"""An implementation of `"Role2vec" <https://arxiv.org/abs/1802.02896>`_
    from the IJCAI '18 paper "Learning Role-based Graph Embeddings".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by multiplying the pooled adjacency power matrix with a 
    structural feature matrix (in this case Weisfeiler-Lehman features). This way
    one gets structural node embeddings.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 2.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        min_count (int): Minimal count of feature occurrences. Default is 10.
        wl_iterations (int): Number of Weisfeiler-Lehman hashing iterations. Default is 2.
        seed (int): Random seed value. Default is 42.
        erase_base_features (bool): Removing the base features. Default is False.
    """
    def __init__(self, walk_number: int=10, walk_length: int=80, dimensions: int=128, workers: int=4,
                 window_size: int=2, epochs: int=1, learning_rate: float=0.05, down_sampling: float=0.0001,
                 min_count: int=10, wl_iterations: int=2, seed: int=42, erase_base_features: bool=False):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.down_sampling = down_sampling
        self.min_count = min_count
        self.wl_iterations = wl_iterations
        self.seed = seed
        self.erase_base_features = erase_base_features

    def _transform_walks(self, walks):
        """
        Transforming the random walks.
        
        Arg types:
            * **walks** *(list of lists)* - Random walks with string ids.

        Return types:
            * *(list of lists)* - The random walks as integers.
        """
        return [[int(node) for node in walk] for walk in walks]

    def _create_documents(self, walks, features):
        """
        Accumulating the WL feature in neighbourhoods.
        
        Arg types:
            * **walks** *(list of lists)* - Random walks with string ids.

        Return types:
            * **new_features** *(list of TaggedDocument objects)* - The pooled features of nodes.
        """
        new_features = {node: [] for node, feature in features.items()}
        walks = self._transform_walks(walks)
        for walk in walks:
            for i in range(self.walk_length-self.window_size):
                for j in range(self.window_size):
                    source = walk[i]
                    target = walk[i+j]
                    new_features[source].append(features[target])
                    new_features[target].append(features[source])

        new_features = {node: [feature for features in new_features[node] for feature in features] for node, _ in new_features.items()}
        new_features = [TaggedDocument(words=feature, tags=[str(node)]) for node, feature in new_features.items()]
        return new_features


    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Role2vec model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)
 
        hasher = WeisfeilerLehmanHashing(graph=graph,
                                         wl_iterations=self.wl_iterations,
                                         attributed=False,
                                         erase_base_features=self.erase_base_features)
      
        node_features = hasher.get_node_features()
        documents = self._create_documents(walker.walks, node_features)

        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        dm=0,
                        workers=self.workers,
                        sample=self.down_sampling,
                        iter=self.epochs,
                        alpha=self.learning_rate,
                        seed=self.seed)

        self._embedding = [model.docvecs[str(i)] for i, _ in enumerate(documents)]

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
