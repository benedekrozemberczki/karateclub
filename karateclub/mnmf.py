import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.sparse import coo_matrix

class M_NMF:
    r"""An implementation of `"M-NMF" <https://smartyfh.com/Documents/18DANMF.pdf>`_
    from the AAAI '17 paper "Community Preserving Network Embedding".
    The procedure uses joint non-negative matrix factorization with modularity
    based regul;arization in order to learn a cluster memmbership distribution
    over nodes. The method can be used in an overlapping and non-overlapping way.

    Args:
        dimensions (int): Number of dimensions. Default is 128.
        clusters (int): Number of clusters. Default is 10.
        lambd (float): KKT penalty. Default is 0.2
        alpha (float): Clustering penalty. Default is 0.05.
        beta (float): Modularity regularization penalty. Default is 0.05.
        iteration_number (int): Number of power iterations. Default is 200.
        lower_control (float): Floating point overflow control. Default is 10**-15.
        eta (float): Similarity mixing parameter. Default is 5.0.
    """
    def __init__(self, dimensions=128, clusters=10, lambd=0.2, alpha=0.05,
                 beta=0.05, iteration_number=200, lower_control=10**-15, eta=5.0):

        self.dimensions = dimensions
        self.clusters = clusters
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.iteration_number = iteration_number
        self.lower_control = lower_control
        self.eta = eta

    def _model_init_print(self):
        print("Model initialization started.\n")

    def _optimization_print(self):
        print("Optimization started.\n")

    def _modularity_generator(self, graph):
        """Calculating the sparse modularity matrix."""
        degs = nx.degree(self.graph)
        e_count = self.graph.number_of_edges()
        n_count = self.graph.number_of_nodes()
        modularity_mat_shape = (n_count, n_count)
        indices_1 = np.array([edge[0] for edge in self.graph.edges()])
        indices_2 = np.array([edge[1] for edge in self.graph.edges()])
        scores = [1.0-(float(degs[e[0]]*degs[e[1]])/(2*e_count)) for e in self.graph.edges()]
        mod_matrix = coo_matrix((scores, (indices_1, indices_2)), shape=modularity_mat_shape)
        return mod_matrix

    def _setup_matrices(self):
        """Creating parameter matrices and target matrices."""
        self.number_of_nodes = nx.number_of_nodes(self.graph)
        self.M = np.random.uniform(0, 1, (self.number_of_nodes, self.dimensions))
        self.U = np.random.uniform(0, 1, (self.number_of_nodes, self.dimensions))
        self.H = np.random.uniform(0, 1, (self.number_of_nodes, self.clusters))
        self.C = np.random.uniform(0, 1, (self.clusters, self.dimensions))
        self.B1 = nx.adjacency_matrix(self.graph)
        self.B2 = self._modularity_generator(self.graph)
        self.X = np.transpose(self.U)
        overlaps = self.B1.dot(self.B1)
        self.S = self.B1 + self.eta*self.B1*(overlaps)

    def _update_M(self):
        """Update matrix M."""
        enum = self.S.dot(self.U)
        denom = np.dot(self.M, np.dot(np.transpose(self.U), self.U))
        denom[denom < self.lower_control] = self.lower_control
        self.M = np.multiply(self.M, enum/denom)
        row_sums = self.M.sum(axis=1)
        self.M = self.M / row_sums[:, np.newaxis]

    def _update_U(self):
        """Update matrix U."""
        enum = (self.S.transpose()).dot(self.M)+self.alpha*np.dot(self.H, self.C)
        denom = np.dot(self.U, np.dot(np.transpose(self.M), self.M)+self.alpha*np.dot(np.transpose(self.C), self.C))
        denom[denom < self.lower_control] = self.lower_control
        self.U = np.multiply(self.U, enum/denom)
        row_sums = self.U.sum(axis=1)
        self.U = self.U / row_sums[:, np.newaxis]

    def _update_C(self):
        """Update matrix C."""
        enum = np.dot(np.transpose(self.H), self.U)
        denom = np.dot(self.C, np.dot(np.transpose(self.U), self.U))
        denom[denom < self.lower_control] = self.lower_control
        frac = enum/denom
        self.C = np.multiply(self.C, frac)
        row_sums = self.C.sum(axis=1)
        self.C = self.C / row_sums[:, np.newaxis]

    def _update_H(self):
        """Update matrix H."""
        B1H = self.B1.dot(self.H)
        B2H = self.B2.dot(self.H)
        HHH = np.dot(self.H, (np.dot(np.transpose(self.H), self.H)))
        UC = np.dot(self.U, np.transpose(self.C))
        rooted = np.square(2*self.beta*B2H)+np.multiply(16*self.lambd*HHH, (2*self.beta*B1H+2*self.alpha*UC+(4*self.lambd-2*self.alpha)*self.H))
        rooted[rooted < 0] = 0
        sqroot_1 = np.sqrt(rooted)
        enum = -2*self.beta*B2H+sqroot_1
        denom = 8*self.lambd*HHH
        denom[denom < self.lower_control] = self.lower_control
        rooted = enum/denom
        rooted[rooted < 0] = 0
        sqroot_2 = np.sqrt(rooted)
        self.H = np.multiply(self.H, sqroot_2)
        row_sums = self.H.sum(axis=1)
        self.H = self.H / row_sums[:, np.newaxis]

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            memberships (dict): Node cluster memberships.
        """
        indices = np.argmax(self.H, axis=1)
        memberships = {i: membership for i, membership in enumerate(indices)}
        return memberships

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self.U
        return embedding

    def get_cluster_centers(self):
        r"""Getting the node embedding.

        Return types:
            * **centers** *(Numpy array)* - The cluster centers.
        """
        centers = self.C
        return centers

    def fit(self, graph):
        """
        Fitting an M-NMF clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._model_init_print()
        self.graph = graph
        self._setup_matrices()
        self._optimization_print()
        for _ in tqdm(range(self.iteration_number)):
            self._update_M()
            self._update_U()
            self._update_C()
            self._update_H()
