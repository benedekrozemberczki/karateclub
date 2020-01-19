import random
import networkx as nx
from karateclub.estimator import Estimator

class LabelPropagation(Estimator):
    r"""An implementation of `"Label Propagation Clustering" <https://arxiv.org/abs/0709.2938>`_
    from the Physical Review '07 paper "Near Linear Time Algorithm to Detect Community Structures
    in Large-Scale Networks". The tool executes a series of label propagations wiht unique labels.
    The final labels are used as cluster memberships.

    Args:
        seed (int): Random seed. Default is 42.
        rounds (int): Propagation iterations. Default is 100.
    """
    def __init__(self, seed=42, iterations=100):
        self.seed = seed
        self.iterations = iterations

    def _make_a_pick(self, neighbors):
        """
        Choosing a neighbor from a propagation source node.

        Arg types:
            * **neigbors** *(list)* - Neighboring nodes.
        """
        scores = {}
        for neighbor in neighbors:
            neighbor_label = self.labels[neighbor]
            if neighbor_label in scores.keys():
                scores[neighbor_label] = scores[neighbor_label] + 1
            else:
                scores[neighbor_label] = 1
        top = [key for key, val in scores.items() if val == max(scores.values())]
        return random.sample(top, 1)[0]

    def _do_a_propagation(self):
        """
        Doing a propagation round.
        """
        random.shuffle(self.nodes)
        new_labels = {}
        for node in self.nodes:
            neighbors = [neb for neb in nx.neighbors(self.graph, node)]
            pick = self._make_a_pick(neighbors)
            new_labels[node] = pick
        self.labels = new_labels

    def fit(self, graph):
        """
        Fitting a Label Propagation clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        self.nodes = [node for node in self.graph.nodes()]
        self.labels = {node: i for i, node in enumerate(self.graph.nodes())}
        random.seed(self.seed)
        for _ in range(self.iterations):
            self._do_a_propagation()

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        memberships = self.labels
        return memberships
