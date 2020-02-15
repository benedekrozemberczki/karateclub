import random
import networkx as nx
from karateclub.estimator import Estimator

class SCD(Estimator):
    r"""An implementation of `"SCD" <https://arxiv.org/abs/0709.2938>`_ from the
    WWW '14 paper "High Quality, Scalable and Parallel Community Detection for 
    Large Real Graphs". The procedure optimizes greedily the weighted community
    clustering indirectly.

    Args:
        seed (int): Random seed. Default is 42.
        rounds (int): Propagation iterations. Default is 100.
    """
    def __init__(self, seed=42, iterations=100):
        self.seed = seed
        self.iterations = iterations

    def fit(self, graph):
        """
        Fitting a Label Propagation clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        memberships = self.labels
        return memberships
