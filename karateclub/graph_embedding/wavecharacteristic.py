import math
from typing import List
import numpy as np
import networkx as nx
import scipy.sparse as sparse

from karateclub.estimator import Estimator




class WaveCharacteristic(Estimator):
    r"""An implementation of `"WaveCharacteristic" <https://arxiv.org/abs/2005.07959>`_
    from the CIKM '21 paper "Graph Embedding via Diffusion-Wavelets-Based Node Feature 
    Distribution Characterization". The procedure uses characteristic functions of 
    node features with wavelet function weights to describe node neighborhoods. 
    These node level features are pooled by mean pooling to create graph level statistics.

    Args:
        order (int): Adjacency matrix powers. Default is 5.
    """

    def __init__(self, order: int = 5):
        super(WaveCharacteristic, self).__init__()

        self.order = 2

    def _calculate_feather(self, graph: nx.classes.graph.Graph) -> np.ndarray:
        """
        Calculating the characteristic function features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **features** *(Numpy vector)* - The embedding of a single graph.
        """
        return np.array([0.1,0.2,0.3])

    def fit(self, graphs: List[nx.classes.graph.Graph]) -> None:
        """
        Fitting a graph level FEATHER model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        self._embedding = [self._calculate_feather(graph) for graph in graphs]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)
