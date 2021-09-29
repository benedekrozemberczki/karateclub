import math
from typing import List, Tuple
import numpy as np
import networkx as nx
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from karateclub.estimator import Estimator


class WaveletCharacteristic(Estimator):
    r"""An implementation of `"WaveCharacteristic" <https://arxiv.org/abs/2005.07959>`_
    from the CIKM '21 paper "Graph Embedding via Diffusion-Wavelets-Based Node Feature 
    Distribution Characterization". The procedure uses characteristic functions of 
    node features with wavelet function weights to describe node neighborhoods. 
    These node level features are pooled by mean pooling to create graph level statistics.

    Args:
        order (int): Adjacency matrix powers. Default is 5.
        eval_points (int): Number of characteristic function evaluations. Default is 5.
        theta_max (float): Largest characteristic function time value. Default is 2.5.
        tau (float): Wave function heat - time diffusion. Default is 1.0.
        pooling (str): Pooling function appliead to the characteristic functions. Default is "mean".
    """

    def __init__(self, order: int=5, eval_points: int=25,
                 theta_max: float=2.5, tau: float=1.0, pooling: str="mean"):
        self.order = order
        self.eval_points = eval_points
        self.theta_max = theta_max
        self.tau = tau
        self.pooling = pooling


    def _create_D_inverse(self, graph):
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


    def _get_normalized_adjacency(self, graph):
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def _heat_diffusion_ind(self, graph):
        L = nx.laplacian_matrix(graph, nodelist=range(graph.number_of_nodes()))
        lamb, U = np.linalg.eigh(L.todense())
        heat = U.dot(np.diagflat(np.exp(- self.tau * lamb).flatten())).dot(U.T)
        return heat

    def _create_node_feature_matrix(self, graph):
        log_degree = np.array([math.log(graph.degree(node)+1) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        clustering_coefficient = np.array([nx.clustering(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        X = np.concatenate([log_degree, clustering_coefficient], axis=1)
        return X


    def _calculate_wavelet_characteristic(self, graph):

        A_tilde = self._get_normalized_adjacency(graph)
        X = self._create_node_feature_matrix(graph)
        theta = np.linspace(0.01, self.theta_max, self.eval_points)
        X = np.outer(X, theta)
        X = X.reshape(graph.number_of_nodes(), -1)
        X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
        feature_blocks = []
        A_tilde = A_tilde.toarray()
        A_tilde_p = np.copy(A_tilde)
        
        heat = self._heat_diffusion_ind(graph)
        diffusion = np.copy(heat)
        diffusion  = np.exp(np.sum(np.abs(diffusion[:, np.newaxis] - diffusion), axis=2))

        degree_vector = np.array([graph.degree(node) for node in range(graph.number_of_nodes())])
        D_rep = np.outer(degree_vector, np.ones((graph.number_of_nodes(),)))
        
        for _ in range(self.order):
            A_tilde_2 = np.copy(A_tilde)
            A_tilde_3 = np.copy(A_tilde)
            
            A_tilde_3[A_tilde_2>0] = diffusion[A_tilde_2>0]
            A_tilde_2[A_tilde_2>0] = D_rep[A_tilde_2>0]       
            
            A_tilde_2 = normalize(A_tilde_2, axis=1, norm='l1')
            A_tilde_3 = normalize(A_tilde_3, axis=1, norm='l1')

            X_1 = A_tilde_2.dot(X)
            X_2 = A_tilde_3.dot(X)
            
            feature_blocks.append(X_1)
            feature_blocks.append(X_2)
            A_tilde = A_tilde.dot(A_tilde_p)

        feature_blocks = np.concatenate(feature_blocks, axis=1)
        if self.pooling == "mean":
            feature_blocks = np.mean(feature_blocks, axis=0)
        elif self.pooling == "min":
            feature_blocks = np.min(feature_blocks, axis=0)
        elif self.pooling == "max":
            feature_blocks = np.max(feature_blocks, axis=0)
        else:
            raise ValueError("Wrong pooling function.")
        return feature_blocks


    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a Geometric-Scattering model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._check_graphs(graphs)
        self._embedding = [self._calculate_wavelet_characteristic(graph) for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
