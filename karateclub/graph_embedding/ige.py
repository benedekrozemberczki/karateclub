"""Invariant Graph Embedding model class."""

from typing import List
import numpy as np
import networkx as nx
import scipy.sparse as sps
from karateclub.estimator import Estimator

class IGE(Estimator):
    r"""An implementation of `"Invariant Graph Embedding" <https://graphreason.github.io/papers/16.pdf>`_
    from the ICML 2019 Workshop on Learning and Reasoning with Graph-Structured 
    Data paper "Invariant Embedding for Graph Classification". The procedure 
    computes a mixture of spectral and node embedding based features. Specifically,
    it uses scattering, eigenvalues and pooled node feature embeddings to create
    graph descriptors.

    Args:
        feature_embedding_dimensions (list): Feature embedding dimensions. Default is [3, 5]
        spectral_embedding_dimensions (list): Spectral embedding dimensions. Default is [10, 20].
        histogram_bins (list): Number of histogram bins. Default is [10, 20].
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, feature_embedding_dimensions: List[int]=[3, 5],
                 spectral_embedding_dimensions: List[int]=[10, 20],
                 histogram_bins: List[int]=[10, 20],
                 seed: int=42):
        self.feature_embedding_dimensions = feature_embedding_dimensions
        self.spectral_embedding_dimensions = spectral_embedding_dimensions
        self.histogram_bins = histogram_bins
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
        D_inverse = sps.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


    def _get_normalized_adjacency(self, graph):
        """
        Calculating the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **A_hat** *(SciPy array)* - The adjacency matrix of the graph.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def _get_embedding_features(self, graph, features):
        """
        Calculating the embedding features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.
            * **features** *(list)* - The list of graph feature NumPy arrays.

        Return types:
            * **features** *(list)* - The list of graph feature NumPy arrays.
        """
        number_of_nodes = graph.number_of_nodes()
        mat_eye = np.eye(self.max_deg + 1)
        degrees = [graph.degree[node] for node in graph.nodes()]
        sub_features = mat_eye[degrees]
        feature_dim = sub_features.shape[1]
        for emb_dim in self.feature_embedding_dimensions:
            emb_space_full = emb_dim * feature_dim

            embed_space = np.zeros((emb_space_full, number_of_nodes))
            P = self._get_normalized_adjacency(graph)
            Q = self._get_normalized_adjacency(graph)
            for i in range(emb_dim):
                P = P.dot(Q)
                embed_space[i * feature_dim:(i + 1) * feature_dim, :] = P.dot(sub_features).T

            features.append(np.mean(embed_space, axis=1).T)
        return features

    def _get_spectral_features(self, graph, features):
        """
        Calculating the spectral features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.
            * **features** *(list)* - The list of graph feature NumPy arrays.

        Return types:
            * **features** *(list)* - The list of graph feature NumPy arrays.
        """
        L = nx.laplacian_matrix(graph).asfptype()
        for emb_dim in self.spectral_embedding_dimensions:
            emb_eig = np.zeros(emb_dim)
            min_dim = min(graph.number_of_nodes()-1, emb_dim)
            eigenvalues = sps.linalg.eigsh(L, min_dim, which="SM",
                                           ncv=25*min_dim, return_eigenvectors=False)
            emb_eig[-min_dim:] = eigenvalues[:min_dim]
            features.append(emb_eig)
        return features

    def _get_histogram_features(self, graph, features):
        """
        Calculating the spectral histogram features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.
            * **features** *(list)* - The list of graph feature NumPy arrays.

        Return types:
            * **features** *(list)* - The list of graph feature NumPy arrays.
        """
        L = nx.laplacian_matrix(graph).asfptype()
        eigenvalues, eigenvectors = sps.linalg.eigsh(L)

        eigenvectors_norm = np.dot(np.diag(np.sqrt(1 / eigenvalues[1:])),
                                   eigenvectors.T[1:, :])
        sim = np.dot(eigenvectors_norm.T, eigenvectors_norm)
        sim = np.reshape(sim, (1, -1))
        for bins in self.histogram_bins:
            hist = np.histogram(sim, range=(-1, 1), bins=bins)[0]
            features.append(hist)
        return features

    def _calculate_invariant_embedding(self, graph):
        """
        Calculating features from generic embedding, spectral embeddings and histogram features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **features** *(NumPy array)* - The features of the graph.
        """
        features = []
        features = self._get_embedding_features(graph, features)
        features = self._get_spectral_features(graph, features)
        features = self._get_histogram_features(graph, features)
        features = np.concatenate(features).reshape(1, -1)
        return features

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting an Invariant Graph Embedding model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self.max_deg = max([max([graph.degree[n] for n in graph.nodes()]) for graph in graphs])
        self._embedding = [self._calculate_invariant_embedding(graph) for graph in graphs]
        self._embedding = np.concatenate(self._embedding)

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return self._embedding
