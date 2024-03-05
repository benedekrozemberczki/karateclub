"""Implementation of SINr: Fast Computing of Sparse Interpretable Node Representations."""

from typing import List, Set, Optional
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np
from karateclub.estimator import Estimator


class SINr(Estimator):
    r"""An implementation of `"SINr" <https://inria.hal.science/hal-03197434/>`_
    from the IDA '21 best paper "SINr: Fast Computing of Sparse Interpretable Node Representations is not a Sin!".
    The procedure performs community detection using the Louvain algorithm, and computes the distribution of edges
    of each node across all communities.
    The algorithm is one of the fastest, because it mostly relies on Louvain community detection. 
    It thus runs in quasi-linear time. Regarding space complexity, the adjacency matrix and the community
    membership matrix need to be stored, it is also quasi-linear.

    Args:
        gamma (int): modularity multi-resolution parameter. Default is 1.
            The dimension parameter does not exist for SINr, gamma should be used instead:
            the number of dimensions of the embedding space is based on the number of communities uncovered.
            The higher gamma is, the more communities are detected, the higher the number of dimensions of
            the latent space are uncovered. For small graphs, setting gamma to 1 is usually sufficient.
            For bigger graphs, it is recommended to increase gamma (5 or 10 for example).
            For word co-occurrence graphs, to deal with word embedding, gamma is usually set to 50 in order to get many small communities.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        gamma: int = 1,
        seed: int = 42,
    ):
        self.gamma: int = gamma
        self.seed: int = seed
        self.number_of_nodes: Optional[int] = None
        self.number_of_communities: Optional[int] = None
        self._embedding: Optional[np.ndarray] = None

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a SINr model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        # Get the adjacency matrix of the graph
        adjacency = nx.adjacency_matrix(graph)
        norm_adjacency = normalize(adjacency, "l1")  # Make rows of matrix sum at 1
        # Detect communities use louvain algorithm with the gamma resolution parameter
        communities = nx.community.louvain_communities(
            graph, resolution=self.gamma, seed=self.seed
        )
        self.number_of_nodes = graph.number_of_nodes()
        self.number_of_communities = len(communities)
        # Get the community membership of the graph
        membership_matrix = self._get_matrix_membership(communities)
        # Computes the node-recall: for each node, the distribution of links across communities
        self._embedding = norm_adjacency.dot(membership_matrix)

    def _get_matrix_membership(self, list_of_communities: List[Set[int]]):
        r"""Getting the membership matrix describing for each node (rows), in which community (column) it belongs.

        Return types:
            * **Membership matrix** *(scipy sparse matrix csr)* - Size nodes, communities
        """
        # Since we will have a lot of zeros, we use a sparse matrix.
        # We build a CSR matrix.

        # A CSR matrix is composite of two arrays: the data array and the indices array.
        # The data array is a 1D array that contains all the non-zero values of the matrix.
        nodes_per_community = np.empty(self.number_of_nodes, dtype=np.uint32)
        # The indices array is a 1D array that contains the offsets of the start of each row of the matrix.
        communities_comulative_degrees = np.empty(self.number_of_communities + 1, dtype=np.uint32)
        offset: int = 0

        # For each community, we store the nodes that belong to it.
        for column_index, community in enumerate(list_of_communities):
            # We store the offset of the start of each row of the matrix.
            communities_comulative_degrees[column_index] = offset
            # We store the nodes that belong to the community.
            for node in community:
                nodes_per_community[offset] = node
                offset += 1

        assert offset == self.number_of_nodes

        # We set the offset of the end of the last row of the matrix
        # to the number of nodes, which is expected to be identical
        # to the offset of the start of the last row of the matrix.
        communities_comulative_degrees[-1] = self.number_of_nodes

        # And finally we can build the matrix.
        return csr_matrix(
            (
                np.ones(self.number_of_nodes, dtype=np.float32),
                nodes_per_community,
                communities_comulative_degrees,
            ),
            shape=(self.number_of_communities, self.number_of_nodes),
        ).T

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        if self._embedding is None:
            raise ValueError(
                "No embedding has been computed. "
                "Please call the fit method first."
            )

        return self._embedding.toarray()
