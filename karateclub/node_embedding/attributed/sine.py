import random
import numpy as np
import networkx as nx
from karateclub.estimator import Estimator
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker

class SINE(Estimator):
    r"""An implementation of `"SINE" <https://shiruipan.github.io/publication/yang-binarized-2018/yang-binarized-2018.pdf>`_
    from the ICDM '18 paper "Binarized Attributed Network Embedding Class". The 
    procedure first calculates the truncated SVD of an adjacency - feature matrix
    product. This matrix is further decomposed by a binary CCD based technique. 
       
    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        svd_iterations (int): SVD iteration count. Default is 20.
        seed (int): Random seed. Default is 42.
        alpha (float): Kernel matrix inversion parameter. Default is 0.3. 
        iterations (int): Matrix decomposition iterations. Default is 100.
        binarization_iterations (int): Binarization iterations. Default is 20.
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


    def _feature_transform(self, graph, X):
        features = {str(node): [] for node in graph.nodes()}
        nodes = X.row
        for i, node in enumerate(nodes):
            features[str(node)].append("feature_"+ str(X.col[i]))
        return features

    def _select_walklets(self):
        self.walklets = []
        for walk in self.walker.walks:
            for power in range(1,self.window_size+1): 
                for step in range(power+1):
                    neighbors = [n for i, n in enumerate(walk[step:]) if i % power == 0]
                    neighbors = [n for n in neighbors for _ in (0, 3)]
                    #neighbors = [random.choice(self.features[val]) if i % 3 == 1 else val for i, val in enumerate(neighbors)]
                    print(neighbors)
                    self.walklets.append(neighbors)
        del self.walker
        
        
    def fit(self, graph, X):
        """
        Fitting a SINE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO array)* - The matrix of node features.
        """
        self.walker = RandomWalker(self.walk_length, self.walk_number)
        self.walker.do_walks(graph)
        self.features = self._feature_transform(graph, X)
        self._select_walklets()
        print(self.walklets)


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = None
        return embedding
