import math
import random
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import NMF
from karateclub.estimator import Estimator

class BoostNE(Estimator):
    r"""An implementation of `"BoostNE" <https://arxiv.org/abs/1808.08627>`_
    from the ASONAM '19 paper "Multi-Level Network Embedding with Boosted Low-Rank
    Matrix Approximation". The procedure uses non-negative matrix factorization 
    iteratively to decompose the residuals obtained by previous factorization models.
    The base target matrix is a pooled sum of adjacency matrix powers. 

    Args:
        dimensions (int): Number of individual embedding dimensions. Default is 8.
        iterations (int): Number of boosting iterations. Default is 16.
        order (int): Number of adjacency matrix powers. Default is 2.
        alpha (float): NMF regularization parameter. Default is 0.01.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, dimensions: int=8, iterations: int=16,
                 order: int=2, alpha: float=0.01, seed: int=42):
        self.dimensions = dimensions
        self.iterations = iterations
        self.order = order
        self.alpha = alpha
        self.seed = seed

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
            * **(A_hat, A_hat, A_hat)** *(Tuple of SciPy arrays)* - Normalized adjacency matrices.
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

        Arg types:
            * **index** *(int)* - The axis for marginalization.

        Return types:
            * **sample** *(int)* - Anchor point index.
        """
        row_weights = self._residuals.sum(axis=index)
        if len(row_weights.shape) > 1:
            row_weights = row_weights.reshape(-1)
        sums = np.sum(np.sum(row_weights))
        to_pick_from = row_weights.reshape(-1)
        to_pick_from = (to_pick_from/np.sum(to_pick_from)).tolist()[0]
        sample = self._binary_search(to_pick_from)
        return sample

    def _reweighting(self, X, chosen_row, chosen_col):
        """
        Re-scaling the target matrix with the anchor row and column.

        Arg types:
            * **X** *(COO Scipy matrix)* - The target matrix.
            * **chosen_row** *(int)* - The row anchor.
            * **choswen_col** *(int)* - The column anchor.

        Return types:
            * **X** *(COO Scipy matrix)* - The rescaled target matrix.
        """
        row_sims = X.dot(chosen_row.transpose())
        column_sims = chosen_col.transpose().dot(X)
        X = sparse.csr_matrix(row_sims).multiply(X)
        X = X.multiply(sparse.csr_matrix(column_sims))
        return X

    def _fit_and_score_NMF(self, new_residuals):
        """
        Factorizing a residual matrix, returning the approximate target, and an embedding.

        Arg types:
            * **new_residuals** *(COO Scipy matrix)* - The residual matrix.

        Return types:
            * **scores** *(COO Scipy matrix)* - The residual scores.
            * **W** *(Numpy array)* - The embedding matrix.
        """
        model = NMF(n_components=self.dimensions,
                    init="random",
                    verbose=False,
                    alpha=self.alpha)

        W = model.fit_transform(new_residuals)
        H = model.components_

        sub_scores = np.sum(np.multiply(W[self._index_1, :], H[:, self._index_2].T), axis=1)
        scores = np.maximum(self._residuals.data-sub_scores, 0)
        scores = sparse.csr_matrix((scores, (self._index_1, self._index_2)),
                                   shape=self._shape,
                                   dtype=np.float32)
        return scores, W

    def _setup_base_model(self):
        """
        Fitting NMF on the starting matrix.
        """
        self._shape = self._residuals.shape
        indices = self._residuals.nonzero()
        self._index_1 = indices[0]
        self._index_2 = indices[1]
        base_score, embedding = self._fit_and_score_NMF(self._residuals)
        self._embeddings = [embedding]


    def _binary_search(self, weights):
        """
        Weighted search procedure. Choosing a random index.

        Arg types:
            * **weights** *(Numpy array)* - The weights for choosing an index.

        Return types:
            * **low/mid** *(int)* - Sampled index.
        """
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
        chosen_row = self._residuals[row, :]
        chosen_column = self._residuals[:, column]
        new_residuals = self._reweighting(self._residuals, chosen_row, chosen_column)
        scores, embedding = self._fit_and_score_NMF(new_residuals)
        self._embeddings.append(embedding)
        self._residuals = scores

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a BoostNE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        self._residuals = self._create_target_matrix(graph)
        self._setup_base_model()
        for _ in range(self.iterations):
            self._single_boosting_round()

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate(self._embeddings, axis=1)
        return embedding
