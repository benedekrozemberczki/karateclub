import numpy as np
from tqdm import tqdm
import networkx as nx

class NNSED(object):
    r"""An implementation of `"DANMF" <https://smartyfh.com/Documents/18DANMF.pdf>`_
    from the CIKM '18 paper "Deep Autoencoder-like Nonnegative Matrix Factorization for
    Community Detection". The procedure uses telescopic non-negative matrix factorization
    in order to learn a cluster memmbership distribution over nodes. The method can be 
    used in an overlapping and non-overlapping way.

    Args:
        layers (int): Autoencoder layer size. Default 32.
        iterations (int): Number of training epochs. Default 100.
        seed (int): Random seed for weight initializations. Default 42.
    """
    def __init__(self, dimensions=32, iterations=100, seed=42):
        self.dimensions = dimensions
        self.iterations = iterations
        self.seed = seed

    def _setup_target_matrix(self, graph):
        A = nx.adjacency_matrix(graph)
        return A

    def _setup_embeddings(self, graph):
        """
        Setup target matrix for pre-training process.
        """
        number_of_nodes = graph.shape[0]
        self.W = np.random.uniform(0, 1, size=(number_of_nodes, self.dimensions))
        self.Z = np.random.uniform(0, 1, size=(self.dimensions, number_of_nodes))

    def _update_W(self, A):
        pass

    def _update_Z(self, A):
        pass


    def get_embedding(self):
        r"""Getting the bottleneck layer embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = [self.W, self.Z.T]
        embedding = np.concatenate(embedding, axis=1)
        return embedding


    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            memberships (dict): Node cluster memberships.
        """
        embedding = [self.W, self.Z.T]
        index = np.argmax(self.embedding, axis=1)
        memberships = {int(i): int(index[i]) for i in range(len(index))}
        return memberships

    def fit(self, graph):
        """
        Fitting an NNSED clustering model.
        
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        print("\n\nTraining started. \n")

        A = self._setup_target_matrix(graph)
        self._setup_embeddings(A)
        for _ in tqdm(range(self.iterations), desc="Training pass: ", leave=True):
            self._update_W(A)
            self._update_Z(A)
