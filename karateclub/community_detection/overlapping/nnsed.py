import numpy as np
import networkx as nx
from typing import Dict
from scipy import sparse
from karateclub.estimator import Estimator

class NNSED(Estimator):
    r"""An implementation of `"NNSED"
    <http://www.bigdatalab.ac.cn/~shenhuawei/publications/2017/cikm-sun.pdf>`_
    from the CIKM '17 paper "A Non-negative Symmetric Encoder-Decoder Approach
    for Community Detection". The procedure uses non-negative matrix factorization
    in order to learn an unnormalized cluster membership distribution over nodes.
    The method can be used in an overlapping and non-overlapping way.

    Args:
        layers (int): Embedding layer size. Default is 32.
        iterations (int): Number of training epochs. Default 10.
        seed (int): Random seed for weight initializations. Default 42.
        noise (float): Random noise for normalization stability. Default is 10**-6.
    """
    def __init__(self, dimensions: int=32, iterations: int=10,
                 seed: int=42, noise: float=10**-6):
        self.dimensions = dimensions
        self.iterations = iterations
        self.seed = seed
        self.noise = noise

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.

        Return types:
            * **A_hat** *Scipy array* - Normalized adjacency.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def _setup_embeddings(self, graph):
        """
        Setup the node embedding matrices.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        number_of_nodes = graph.shape[0]
        self._W = np.abs(np.random.normal(0, 10, size=(number_of_nodes, self.dimensions)))
        self._Z = np.abs(np.random.normal(0, 10, size=(self.dimensions, number_of_nodes)))

    def _update_W(self, A):
        """
        Updating the vertical basis matrix.

        Arg types:
            * **A** *(Scipy COO matrix)* - The normalized adjacency matrix.
        """
        enum = A.dot(self._Z.T)
        denom_1 = self._W.dot(self._Z).dot(self._Z.T)
        denom_2 = (A.dot(A.transpose())).dot(self._W)
        denom = denom_1 + denom_2 + self.noise
        self._W = self._W*(enum/denom)

    def _update_Z(self, A):
        """
        Updating the horizontal basis matrix.

        Arg types:
            * **A** *(Scipy COO matrix)* - The normalized adjacency matrix.
        """
        enum = (A.dot(self._W)).transpose()
        denom = np.dot(np.dot(self._W.T, self._W), self._Z) + self._Z + self.noise
        self._Z = self._Z*(enum/denom)

    def get_embedding(self) -> np.array:
        r"""Getting the bottleneck layer embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding =  self._Z.T
        return embedding


    def get_memberships(self) -> Dict[int, int]:
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        index = np.argmax(self._Z, axis=0)
        memberships = {int(i): int(index[i]) for i in range(len(index))}
        return memberships

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting an NNSED clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._set_seed()
        self._check_graph(graph)
        A = self._create_base_matrix(graph)
        self._setup_embeddings(A)
        for _ in range(self.iterations):
            self._update_W(A)
            self._update_Z(A)
