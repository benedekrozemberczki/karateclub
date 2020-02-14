import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm
from karateclub.estimator import Estimator

class SymmNMF(Estimator):

    r"""An implementation of `"Symm-NMF" <https://www.cc.gatech.edu/~hpark/papers/DaDingParkSDM12.pdf>`_
    from the SDM'12 paper "Symmetric Nonnegative Matrix Factorization for Graph Clustering". The procedure
    decomposed the second power od the normalized adjacency matrix with an ADMM based non-negative matrix
    factorization based technique. This results in a node embedding and each node is associated with an
    embedding factor in the created latent space.

    Args:
        dimensions (int): Number of dimensions. Default is 16.
        iterations (int): Number of power iterations. Default is 200.
        rho (float): ADMM tuning parameter. Default is 1.0.
    """
    def __init__(self, dimensions=16, iterations=200, rho=1.0):

        self.dimensions = dimensions
        self.iterations = iterations
        self.rho = rho

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
        self.H = np.abs(np.random.normal(0, 1, size=(number_of_nodes, self.dimensions)))
        self.H_gamma = np.zeros((number_of_nodes, self.dimensions))
        self.I = np.identity(self.dimensions)

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        index = np.argmax(self.W, axis=0)
        memberships = {int(i): int(index[i]) for i in range(len(index))}
        return memberships

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self.H
        return embedding

    def _do_admm_update(self, A_hat, step):
        """
        Doing a single ADMM update with the adjacency matrix.
        
        """
        H_covar = np.linalg.inv(self.H.T.dot(self.H) + self.rho*self.I)
        self.W = (A_hat.dot(A_hat.dot(self.H)) + self.rho*self.H - self.H_gamma).dot(H_covar)
        self.W = np.maximum(self.W, 0)
        W_covar = np.linalg.inv(self.W.T.dot(self.W) + self.rho*self.I)
        self.H = (A_hat.dot(A_hat.dot(self.W)) + self.rho*self.H + self.H_gamma).dot(W_covar)
        self.H = np.maximum(self.H, 0)
        self.H_gamma = self.H_gamma + step*self.rho*(self.W-self.H)

    def fit(self, graph):
        """
        Fitting a Symm-NMF clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        A_hat = self._create_base_matrix(graph)
        self._setup_embeddings(A_hat)
        for step in tqdm(range(self.iterations)):
            self._do_admm_update(A_hat, step)
