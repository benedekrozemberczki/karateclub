import math
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


    def _create_initial_partition(self):
        self.clustering_coefficient = nx.clustering(self.graph)
        self.cc_pairs = [((node_cc**0.5)*self.graph.degree(node), node) for node, node_cc in self.clustering_coefficient.items()]
        self.cc_pairs = sorted(self.cc_pairs, key=lambda tup: tup[0])[::-1]
        self._do_initial_assignments()

    def _do_initial_assignments(self):
        self.cluster_memberships = {}
        neighbor_memberships = {}
        cluster_index = 0 
        for pair in self.cc_pairs:
            if pair[1] in neighbor_memberships:
                self.cluster_memberships[pair[1]] = neighbor_memberships[pair[1]]
            else:
                self.cluster_memberships[pair[1]] = cluster_index
                for neighbor in self.graph.neighbors(pair[1]):
                    if neighbor not in neighbor_memberships:
                        neighbor_memberships[neighbor] = cluster_index
                cluster_index = cluster_index + 1

    def _do_refinements(self):
        pass


    def fit(self, graph):
        """
        Fitting a Label Propagation clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        self._create_initial_partition()
        self.nodes = [node for node in self.graph.nodes()]
        for _ in range(self.iterations):
            random.shuffle(self.nodes)
            self._do_refinement()

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        memberships = self.cluster_memberships
        return memberships
