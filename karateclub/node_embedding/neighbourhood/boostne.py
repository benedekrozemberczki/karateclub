import math
import random
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import NMF
from karateclub.estimator import Estimator

class BoostNE(Estimator):
    r"""An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.

    Args:
        dimensions (int): Number of individual embedding dimensions. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 5.

    """
    def __init__(self, dimensions=4, iteration=10, order=5, alpha=0.01):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order
        self.alpha = alpha

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree[0] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.

        Return types:
            * **(A_hat, A_hat, A_hat)** *(Tuple of SciPy arrays)* - Normalized adjacencies.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return (A_hat, A_hat, A_hat)

    def _create_target_matrix(self, graph):
        """
        Creating a log transformed target matrix.

        Return types:
            * **target_matrix** *(SciPy array)* - The PMI matrix.
        """
        A_tilde, A_hat, A_accum = self._create_base_matrix(graph)
        for _ in range(self.order-1):
            A_tilde = sparse.coo_matrix(A_tilde.dot(A_hat))
            A_accum = A_accum + A_tilde
        A_accum = A_accum / self.order
        return A_accum


    def _sampler(self, index):
        """
        Anchor sampling procedure.
        :param index: Matrix axis row/column chosen for anchor sampling.
        :return sample: Chosen sampled row/column id.
        """
        row_weights = self.residuals.sum(axis=index)
        if len(row_weights.shape) > 1:
            row_weights = row_weights.reshape(-1)
        sums = np.sum(np.sum(row_weights))
        to_pick_from = row_weights.reshape(-1)
        to_pick_from = (to_pick_from/np.sum(to_pick_from)).tolist()[0]
        sample = self._binary_search(to_pick_from)
        return sample

    def _reweighting(self, X, chosen_row, chosen_column):
        """
        Rescaling the target matrix with the anchor row and column.
        :param X: The target matrix rescaled.
        :param chosen_row: Anchor row.
        :param chosen_column: Anchor column.
        :return X: The rescaled residual.
        """
        row_sims = X.dot(chosen_row.transpose())
        column_sims = chosen_column.transpose().dot(X)
        X = sparse.csr_matrix(row_sims).multiply(X)
        X = X.multiply(sparse.csr_matrix(column_sims))
        return X

    def _fit_and_score_NMF(self, new_residuals):
        """
        Factorizing a residual matrix, returning the approximate target and an embedding.
        :param new_residuals: Input target matrix.
        :return scores: Approximate target matrix.
        :return W: Embedding matrix.
        """
        model = NMF(n_components=self.dimensions,
                    init="random",
                    verbose=False,
                    alpha=self.alpha)

        W = model.fit_transform(new_residuals)
        H = model.components_

        sub_scores = np.sum(np.multiply(W[self.index_1, :], H[:, self.index_2].T), axis=1)
        scores = np.maximum(self.residuals.data-sub_scores, 0)
        scores = sparse.csr_matrix((scores, (self.index_1, self.index_2)),
                                   shape=self.shape,
                                   dtype=np.float32)
        return scores, W

    def _setup_base_model(self):
        self.shape = self.residuals.shape
        indices = self.residuals.nonzero()
        self.index_1 = indices[0]
        self.index_2 = indices[1]
        base_score, embedding = self._fit_and_score_NMF(self.residuals)
        self.embeddings = [embedding]


    def _binary_search(self, weights):
        running_totals = np.cumsum(weights)
        target_distance = np.random.uniform(0,1)
        low, high = 0, len(weights)
        while low < high:
            mid = int((low + high) / 2)
            distance = running_totals[mid]
            if distance < target_distance:
                low = mid + 1
            elif distance > target_distance:
                high = mid
            else:
                return mid
        return low

    def _single_boosting_round(self):
        """
        A method to perform anchor sampling, rescaling, factorization and scoring.
        """
        row = self._sampler(1)
        column = self._sampler(0)

        chosen_row = self.residuals[row, :]
        chosen_column = self.residuals[:, column]
        new_residuals = self._reweighting(self.residuals, chosen_row, chosen_column)
        scores, embedding = self._fit_and_score_NMF(new_residuals)
        self.embeddings.append(embedding)
        self.residuals = scores

    def fit(self, graph):
        """
        Fitting a GraRep model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self.residuals = self._create_target_matrix(graph)
        self._setup_base_model()
        for _ in range(self.iterations):
            self._single_boosting_round()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate(self.embeddings, axis=1)
        return embedding
