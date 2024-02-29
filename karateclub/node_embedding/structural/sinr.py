from typing import Dict, List, Set
import networkx as nx
from karateclub.estimator import Estimator
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize
import numpy as np


class SINr(Estimator):
    r"""An implementation of `"SINr" <https://inria.hal.science/hal-03197434/>`_
    from the IDA '21 best paper "SINr: Fast Computing of Sparse Interpretable Node Representations is not a Sin!".
    The procedure computes community detection using Louvain algorithm, and calculates the distribution of edges of each node
    across communities.

    Args:
        gamma (int): modularity multi-resolution parameter. Default is 1. The higher it is, the more communities are detected, the higher the number of dimensions of the latent space uncovered.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        gamma: int = 1,
        seed: int = 42,
    ):

        self.gamma = gamma
        self.workers = workers
        self.seed = seed
        self.erase_base_features = erase_base_features


    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a SINr model model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        # Get the adjacency matrix of the graph
        adjacency = nx.adjacency_matrix(graph)
        norm_adjacency = normalize(adjacency, "l1") # Make rows of matrix sum at 1
        # Detect communities use louvain algorithm with the gamma resolution parameter
        communities = nx.community.louvain_communities(graph, resolution = self.gamma, seed = self.seed)
        # Get the community membership of the graph
        membership_matrix = self._get_matrix_membership(communities)
        
        self._embedding = norm_adjacency.dot(membership_matrix)
        
    def _get_matrix_membership(self, list_of_communities:List[Set[int]]):
        r"""Getting the membership matrix describing for each node (rows), to which community (columns) it belongs.

        Return types:
            * **Membership matrix** *(scipy sparse matrix csr)* - Size nodes, communities
        """
        row = list()
        col = list()
        data = list()
        for idx_c, community in enumerate(list_of_communities):
            for node in community:
                row.append(node)
                col.append(idx_c)
                data.append(1)
        return coo_matrix((data, (row, col)), shape=(len(row), len(list_of_communities))).tocsr()
        
        
    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding.toarray()
