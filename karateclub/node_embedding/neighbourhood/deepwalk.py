import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker

class DeepWalk(object):

    def __init__(self, walk_number=10, walk_length=80, dimensions=128, workers=4,
                 window_size=10, epochs=1, learning_rate=0.05, min_count=1):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count


    def fit(self, graph):
        walker = RandomWalker(self.walk_number, self.walk_length)
        walker.do_walks(graph)

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
        return np.array(self._embedding)
