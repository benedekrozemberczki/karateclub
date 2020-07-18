import numpy as np
import networkx as nx
from typing import List
import scipy.sparse as sps
from karateclub.estimator import Estimator

class NetLSD(Estimator):
    r"""An implementation of `"NetLSD" <https://arxiv.org/abs/1805.10712>`_
    from the KDD '18 paper "NetLSD: Hearing the Shape of a Graph". The procedure
    calculate the heat kernel trace of the normalized Laplacian matrix over a
    vector of time scales. If the matrix is large it switches to an approximation
    of the eigenvalues. 

    Args:
        scale_min (float): Time scale interval minimum. Default is -2.0.
        scale_max (float): Time scale interval maximum. Default is 2.0.
        scale_steps (int): Number of steps in time scale. Default is 250.
        scale_approximations (int): Number of eigenvalue approximations. Default is 200.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, scale_min: float=-2.0, scale_max: float=2.0,
                 scale_steps: int=250, approximations: int=200, seed: int=42):

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_steps = scale_steps
        self.approximations = approximations
        self.seed = seed
   
    def _calculate_heat_kernel_trace(self, eigenvalues):
        """
        Calculating the heat kernel trace of the normalized Laplacian.

        Arg types:
            * **eigenvalues** *(Numpy array)* - The eigenvalues of the graph.

        Return types:
            * **heat_kernel_trace** *(Numpy array)* - The heat kernel trace of the graph.
        """
        timescales = np.logspace(self.scale_min, self.scale_max, self.scale_steps)
        nodes = eigenvalues.shape[0]
        heat_kernel_trace = np.zeros(timescales.shape)
        for idx, t in enumerate(timescales):
            heat_kernel_trace[idx] = np.sum(np.exp(-t * eigenvalues))
        heat_kernel_trace = heat_kernel_trace / nodes
        return heat_kernel_trace

    def _updown_linear_approx(self, eigenvalues_lower, eigenvalues_upper, number_of_nodes):
        """
        Approximating the eigenvalues of the normalized Laplacian.

        Arg types:
            * **eigenvalues_lower** *(Numpy array)* - The smallest eigenvalues of the graph.
            * **eigenvalues_upper** *(Numpy array)* - The largest eigenvalues of the graph.
            * **number_of_nodes** *(int)* - The number of nodes in the graph.

        Return types:
            * **eigenvalues** *(Numpy array)* - The eigenvalues of the graph.
        """
        nal = len(eigenvalues_lower)
        nau = len(eigenvalues_upper)
        eigenvalues = np.zeros(number_of_nodes)
        eigenvalues[:nal] = eigenvalues_lower
        eigenvalues[-nau:] = eigenvalues_upper
        eigenvalues[nal-1:-nau+1] = np.linspace(eigenvalues_lower[-1], eigenvalues_upper[0], number_of_nodes-nal-nau+2)
        return eigenvalues

    def _calculate_eigenvalues(self, laplacian_matrix):
        """
        Calculating the eigenvalues of the normalized Laplacian.

        Arg types:
            * **laplacian_matrix** *(SciPy COO matrix)* - The graph to be decomposed.

        Return types:
            * **eigenvalues** *(Numpy array)* - The eigenvalues of the graph.
        """
        number_of_nodes = laplacian_matrix.shape[0]
        if 2*self.approximations< number_of_nodes:
            lower_eigenvalues = sps.linalg.eigsh(laplacian_matrix, self.approximations, which="SM", ncv=5*self.approximations, return_eigenvectors=False)[::-1]
            upper_eigenvalues = sps.linalg.eigsh(laplacian_matrix, self.approximations, which="LM", ncv=5*self.approximations, return_eigenvectors=False)
            eigenvalues = self._updown_linear_approx(lower_eigenvalues, upper_eigenvalues, number_of_nodes)
        else:
            eigenvalues = sps.linalg.eigsh(laplacian_matrix, number_of_nodes-2, which="LM", return_eigenvectors=False)
        return eigenvalues


    def _calculate_netlsd(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        graph.remove_edges_from(nx.selfloop_edges(graph))
        laplacian = sps.coo_matrix(nx.normalized_laplacian_matrix(graph, nodelist = range(graph.number_of_nodes())), dtype=np.float32)
        eigen_values = self._calculate_eigenvalues(laplacian)
        heat_kernel_trace = self._calculate_heat_kernel_trace(eigen_values)
        return heat_kernel_trace

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a NetLSD model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_netlsd(graph) for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
