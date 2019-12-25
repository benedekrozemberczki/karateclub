import numpy as np
from tqdm import tqdm
import networkx as nx

class NNSED(object):
    r"""An implementation of `"NNSED"
    <http://www.bigdatalab.ac.cn/~shenhuawei/publications/2017/cikm-sun.pdf>`_
    from the CIKM '17 paper "A Non-negative Symmetric Encoder-Decoder Approach
    for Community Detection". The procedure uses non-negative matrix factorization
    in order to learn an unnormalized cluster membership distribution over nodes.
    The method can be used in an overlapping and non-overlapping way.

    Args:
        layers (int): Embedding layer size. Default is 32.
        iterations (int): Number of training epochs. Default 100.
        seed (int): Random seed for weight initializations. Default 42.
    """
    def __init__(self, dimensions=32, iterations=100, seed=42):
        self.dimensions = dimensions
        self.iterations = iterations
        self.seed = seed

    def _setup_target_matrix(self, graph):
        """
        Creating a sparse adjacency matrix.
        """
        A = nx.adjacency_matrix(graph)
        return A

    def _setup_embeddings(self, graph):
        """
        Setup the node embedding matrices.
        """
        number_of_nodes = graph.shape[0]
        self.W = np.random.uniform(0, 1, size=(number_of_nodes, self.dimensions))
        self.Z = np.random.uniform(0, 1, size=(self.dimensions, number_of_nodes))

    def _update_W(self, A):
        """
        Updating the vertical basis matrix.
        """
        enum = A.dot(self.Z.T)
        denom_1 = self.W.dot(np.dot(self.Z, self.Z.T))
        denom_2 = (A.dot(A.transpose())).dot(self.W)
        denom = denom_1 + denom_2
        self.W = self.W*(enum/denom)

    def _update_Z(self, A):
        """
        Updating the horizontal basis matrix.
        """
        enum = A.dot(self.W).T
        denom = np.dot(np.dot(self.W.T, self.W), self.Z) + self.Z
        self.Z = self.Z*(enum/denom)

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
        index = np.argmax(self.W, axis=1)
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
