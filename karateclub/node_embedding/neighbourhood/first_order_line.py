import numpy as np
import networkx as nx
from karateclub.estimator import Estimator
from tqdm.auto import trange


class FirstOrderLINE(Estimator):
    r"""An implementation of `"First-order LINE" <https://arxiv.org/abs/1503.03578>`_
    from the paper "LINE: Large-scale Information Network Embedding".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        epochs (int): Number of epochs. Default is 100.
        mini_batch_size (int): Number of samples in each mini-batch. Default is 128.
        learning_rate (float): learning rate. Default is 0.05.
        learning_rate_decay (float): learning rate decay per epoch. Default is 0.999.
        verbose (bool): whether to show a loading bar using TQDM while queering.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        dimensions: int = 128,
        epochs: int = 100,
        mini_batch_size: int = 128,
        learning_rate: float = 0.05,
        learning_rate_decay: float = 0.999,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.mini_batch_size = mini_batch_size
        self.dimensions = dimensions
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.seed = seed
        self.verbose = verbose
        self.embedding = None

    def _update(
        self,
        embedding: np.ndarray,
        edges: np.ndarray,
        gradient: np.ndarray,
        label: int
    ) -> np.ndarray:
        """Updates the provided gradient."""
        src_embedding = embedding[edges[:, 0]]
        dst_embedding = embedding[edges[:, 1]]
        dots = (src_embedding * dst_embedding).sum(axis=1)
        activations = (1.0 / (1.0 + np.exp(-dots)) - label)
        gradient[edges[:, 0]] += activations.reshape(-1, 1) * dst_embedding
        gradient[edges[:, 1]] += activations.reshape(-1, 1) * src_embedding

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a LINE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()

        number_of_nodes = graph.number_of_nodes()

        edges = np.array(graph.edges(data=False))
        number_of_edges = edges.shape[0]

        self.embedding = np.random.uniform(size=(number_of_nodes, self.dimensions))

        for epoch in trange(self.epochs, desc="Epochs", disable=not self.verbose):
            for _ in range(number_of_edges // self.mini_batch_size):
                gradient = np.zeros_like(self.embedding)

                positive_batch = edges[
                    np.random.randint(
                        number_of_edges,
                        size=self.mini_batch_size//2
                    )
                ]
                self._update(self.embedding, positive_batch, gradient, label=1)
                
                negative_batch = np.random.randint(
                    number_of_nodes,
                    size=(self.mini_batch_size//2, 2)
                )
                self._update(self.embedding, negative_batch, gradient, label=0)

                self.embedding -= (self.learning_rate * self.learning_rate_decay**epoch)*gradient


    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self.embedding