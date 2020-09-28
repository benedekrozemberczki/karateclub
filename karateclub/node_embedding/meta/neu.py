import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from karateclub.estimator import Estimator

class NEU(Estimator):
    r"""An implementation of `"NEU" <https://www.ijcai.org/Proceedings/2017/0544.pdf>`_
    from the IJCAI 17 paper "Fast Network Embedding Enhancement via High Order Proximity Approximation".
    The procedure uses an arbitrary embedding and augments it by higher order proximities wiht a recursive
    meta learning algorithm.

    Args:
        L1 (float): Weight of lower order proximities. Defauls is 0.5
        L2 (float): Weight of higer order proximities. Default is 0.25.
        T (int): Number of iterations. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, L1: float=0.5, L2: float=0.25, T: int=1, seed: int=42):
        self.iterations = T
        self.L1 = L1
        self.L2 = L2
        self.seed = seed

    def _normalize_embedding(self, original_embedding):
        r"""Normalizes matrix rows by their Frobenius norm.
        Args:
            original_embedding (Numpy array): An array containing an embedding

        Return types:
            normalized_embedding (Numpy array): An array containing a normalized embedding
        """
        norms = np.linalg.norm(original_embedding, axis=1)
        normalized_embedding = (original_embedding.T/norms).T
        return normalized_embedding

    def _update_embedding(self, graph, original_embedding):
        r"""Performs the Network Embedding Update on the original embedding.
        Args:
            original_embedding (Numpy array): An array containing an embedding.
            graph (NetworkX graph): The embedded graph.

        Return types:
            embedding (Numpy array): An array containing the updated embedding.
        """
        embedding = self._normalize_embedding(original_embedding)
        adjacency = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        normalized_adjacency = normalize(adjacency, norm='l1', axis=1)
        for _ in range(self.iterations):
            embedding = (embedding + 
                         self.L1*(normalized_adjacency @ embedding) + 
                         self.L2*(normalized_adjacency @ (normalized_adjacency @ embedding)))
        return embedding

    def fit(self, graph: nx.classes.graph.Graph, model: Estimator):
        r"""
        Fitting a model and performing NEU.

        Args:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **model** *(KC embedding model)* - Karate Club embedding.
        """
        self._set_seed()
        self._check_graph(graph)
        self.model = model
        self.model.fit(graph)
        original_embedding = self.model.get_embedding()
        self._embedding = self._update_embedding(graph, original_embedding)    

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
