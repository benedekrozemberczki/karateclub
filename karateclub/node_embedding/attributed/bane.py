import numpy as np
import networkx as nx
from tqdm import tqdm
from numpy.linalg import inv
from sklearn.decomposition import TruncatedSVD

class BANE(object):
    """
    Binarized Attributed Network Embedding Class (ICDM 2018).
    """
    r"""An implementation of `"BANE" <https://arxiv.org/abs/1403.6652>`_
    from the ICDM '18 paper "Binarized Attributed Network Embedding Class".

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
    """
    def __init__(self, dimensions=32, svd_iterations=70, seed=42, alpha=0.3,
                 approximation_rounds=100, binarization_rounds=20):
        self.dimensions = dimensions
        self.svd_iterations = svd_iterations
        self.seed = seed
        self.alpha = alpha
        self.approximation_rounds = approximation_rounds
        self.binarization_rounds = binarization_rounds 


    def _create_target_matrix(self, graph):
        """
        Creating a normalized sparse adjacency matrix target. 
        """
        weighted_graph = nx.Graph()
        for (u, v) in graph.edges():
            weighted_graph.add_edge(u, v, weight=1.0/graph.degree(u))
            weighted_graph.add_edge(v, u, weight=1.0/graph.degree(v))
        P = nx.adjacency_matrix(weighted_graph,
                                nodelist=range(graph.number_of_nodes()))
        return P

    def fit(self, graph, X):
        """
        Creating a BANE embedding.
        1. Running SVD.
        2. Running power iterations and CDC.
        """
 
        self.P = self._create_target_matrix(graph)
        self.X = X
        print("\nFitting BANE model.\nBase SVD fitting started.")
        self._fit_base_SVD_model()
        print("SVD completed.\nFitting binary model.\n")
        self._binary_optimize()

    def _fit_base_SVD_model(self):
        """
        Reducing the dimensionality with SVD in the 1st step.
        """
        self.P = self.P.dot(self.X)
        self.model = TruncatedSVD(n_components=self.dimensions,
                                  n_iter=self.svd_iterations,
                                  random_state=self.seed)

        self.model.fit(self.P)
        self.P = self.model.fit_transform(self.P)

    def _update_G(self):
        """
        Updating the kernel matrix.
        """
        self.G = np.dot(self.B.transpose(), self.B)
        self.G = self.G + self.alpha*np.eye(self.dimensions)
        self.G = inv(self.G)
        self.G = self.G.dot(self.B.transpose()).dot(self.P)

    def _update_Q(self):
        """
        Updating the rescaled target matrix.
        """
        self.Q = self.G.dot(self.P.transpose()).transpose()

    def _update_B(self):
        """
        Updating the embedding matrix.
        """
        for _ in tqdm(range(self.approximation_rounds), desc="Inner approximation:"):
            for d in range(self.dimensions):
                sel = [x for x in range(self.dimensions) if x != d]
                self.B[:, d] = self.Q[:, d]-self.B[:, sel].dot(self.G[sel, :]).dot(self.G[:, d]).transpose()
                self.B[:, d] = np.sign(self.B[:, d])

    def _binary_optimize(self):
        """
        Starting 2nd optimization phase with power iterations and CCD.
        """
        self.B = np.sign(np.random.normal(size=(self.P.shape[0], self.dimensions)))
        for _ in tqdm(range(self.binarization_rounds), desc="Iteration", leave=True):
            self._update_G()
            self._update_Q()
            self._update_B()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self.B
        return embedding
