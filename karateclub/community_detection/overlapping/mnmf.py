import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from karateclub.estimator import Estimator


class MNMF(Estimator):
    r"""An implementation of `"M-NMF" <https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589/13763>`_
    from the AAAI '17 paper "Community Preserving Network Embedding".
    The procedure uses joint non-negative matrix factorization with modularity
    based regularization in order to learn a cluster membership distribution
    over nodes. The method can be used in an overlapping and non-overlapping way.

    Args:
        dimensions (int): Number of dimensions. Default is 128.
        clusters (int): Number of clusters. Default is 10.
        lambd (float): KKT penalty. Default is 0.2
        alpha (float): Clustering penalty. Default is 0.05.
        beta (float): Modularity regularization penalty. Default is 0.05.
        iterations (int): Number of power iterations. Default is 200.
        lower_control (float): Floating point overflow control. Default is 10**-15.
        eta (float): Similarity mixing parameter. Default is 5.0.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions=128, clusters=10, lambd=0.2, alpha=0.05,
                 beta=0.05, iterations=200, lower_control=10**-15, eta=5.0, seed=42):

        self.dimensions = dimensions
        self.clusters = clusters
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.lower_control = lower_control
        self.eta = eta
        self.seed = seed

    def _modularity_generator(self):
        """Calculating the sparse modularity matrix."""
        degs = nx.degree(self._graph)
        e_count = self._graph.number_of_edges()
        n_count = self._graph.number_of_nodes()
        modularity_mat_shape = (n_count, n_count)
        indices_1 = np.array([edge[0] for edge in self._graph.edges()] + [edge[1] for edge in self._graph.edges()])
        indices_2 = np.array([edge[1] for edge in self._graph.edges()] + [edge[0] for edge in self._graph.edges()])
        scores = [1.0-(float(degs[e[0]]*degs[e[1]])/(2*e_count)) for e in self._graph.edges()]
        scores = scores + [1.0-(float(degs[e[1]]*degs[e[0]])/(2*e_count)) for e in self._graph.edges()]
        mod_matrix = coo_matrix((scores, (indices_1, indices_2)), shape=modularity_mat_shape)
        return mod_matrix

    def _setup_matrices(self):
        """Creating parameter matrices and target matrices."""
        self._number_of_nodes = nx.number_of_nodes(self._graph)
        self._M = np.random.uniform(0, 1, (self._number_of_nodes, self.dimensions))
        self._U = np.random.uniform(0, 1, (self._number_of_nodes, self.dimensions))
        self._H = np.random.uniform(0, 1, (self._number_of_nodes, self.clusters))
        self._C = np.random.uniform(0, 1, (self.clusters, self.dimensions))
        self._B1 = nx.adjacency_matrix(self._graph, nodelist=range(self._graph.number_of_nodes()))
        self._B2 = self._modularity_generator()
        self._X = np.transpose(self._U)
        overlaps = self._B1.dot(self._B1)
        self._S = self._B1 + self.eta*self._B1*(overlaps)

    def _update_M(self):
        """Update matrix M."""
        enum = self._S.dot(self._U)
        denom = np.dot(self._M, np.dot(np.transpose(self._U), self._U))
        denom[denom < self.lower_control] = self.lower_control
        self._M = np.multiply(self._M, enum/denom)
        row_sums = self._M.sum(axis=1)
        self._M = self._M / row_sums[:, np.newaxis]

    def _update_U(self):
        """Update matrix U."""
        enum = self._S.dot(self._M)+self.alpha*np.dot(self._H, self._C)
        denom = np.dot(self._U, np.dot(np.transpose(self._M), self._M)+self.alpha*np.dot(np.transpose(self._C), self._C))
        denom[denom < self.lower_control] = self.lower_control
        self._U = np.multiply(self._U, enum/denom)
        row_sums = self._U.sum(axis=1)
        self._U = self._U / row_sums[:, np.newaxis]

    def _update_C(self):
        """Update matrix C."""
        enum = np.dot(np.transpose(self._H), self._U)
        denom = np.dot(self._C, np.dot(np.transpose(self._U), self._U))
        denom[denom < self.lower_control] = self.lower_control
        frac = enum/denom
        self._C = np.multiply(self._C, frac)
        row_sums = self._C.sum(axis=1)
        self._C = self._C / row_sums[:, np.newaxis]

    def _update_H(self):
        """Update matrix H."""
        B1H = self._B1.dot(self._H)
        B2H = self._B2.dot(self._H)
        HHH = np.dot(self._H, (np.dot(np.transpose(self._H), self._H)))
        UC = np.dot(self._U, np.transpose(self._C))
        rooted = np.square(2*self.beta*B2H)+np.multiply(16*self.lambd*HHH, (2*self.beta*B1H+2*self.alpha*UC+(4*self.lambd-2*self.alpha)*self._H))
        rooted[rooted < 0] = 0
        sqroot_1 = np.sqrt(rooted)
        enum = -2*self.beta*B2H+sqroot_1
        denom = 8*self.lambd*HHH
        denom[denom < self.lower_control] = self.lower_control
        rooted = enum/denom
        rooted[rooted < 0] = 0
        sqroot_2 = np.sqrt(rooted)
        self._H = np.multiply(self._H, sqroot_2)
        row_sums = self._H.sum(axis=1)
        self._H = self._H / row_sums[:, np.newaxis]

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        indices = np.argmax(self._H, axis=1)
        memberships = {i: membership for i, membership in enumerate(indices)}
        return memberships

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self._U
        return embedding

    def get_cluster_centers(self):
        r"""Getting the node embedding.

        Return types:
            * **centers** *(Numpy array)* - The cluster centers.
        """
        centers = self._C
        return centers

    def fit(self, graph):
        """
        Fitting an M-NMF clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._set_seed()
        self._check_graph(graph)
        self._graph = graph
        self._setup_matrices()
        for _ in range(self.iterations):
            self._update_M()
            self._update_U()
            self._update_C()
            self._update_H()
