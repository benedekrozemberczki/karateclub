"""Ego Net Splitting implementation."""

import community
import networkx as nx
from tqdm import tqdm

class EgoNetSplitter(object):
    """
    An implementation of "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters". 
    Paper: https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf
    Video: https://www.youtube.com/watch?v=xMGZo-F_jss
    Slides: https://epasto.org/papers/kdd2017-Slides.pdf
    """
    def __init__(self, resolution = 1.0):
        """
        Args:
            resolution (float): Resolution parameter of Python Louvain. Default 0.1.
        """
        self.resolution = resolution

    def _create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.
        :param node: Node ID for egonet (ego node).
        """
        ego_net_minus_ego = self.graph.subgraph(self.graph.neighbors(node))
        components = {i: nodes for i, nodes in enumerate(nx.connected_components(ego_net_minus_ego))}
        new_mapping = {}
        personalities = []
        for k, v in components.items():
            personalities.append(self.index)
            for other_node in v:
                new_mapping[other_node] = self.index 
            self.index = self.index +1
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def _create_egonets(self):
        """
        Creating an egonet for each node.
        """
        self.components = {}
        self.personalities = {}
        self.index = 0
        print("Creating egonets.")
        for node in tqdm(self.graph.nodes()):
            self._create_egonet(node)

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {persona: node for node in self.graph.nodes() for persona in self.personalities[node]}

    def _create_persona_graph(self):
        """
        Create a persona graph using the egonet components.
        """
        print("Creating the persona graph.")
        self.persona_graph_edges = [(self.components[edge[0]][edge[1]], self.components[edge[1]][edge[0]]) for edge in tqdm(self.graph.edges())]
        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)

    def _create_partitions(self):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        print("Clustering the persona graph.")
        self.partitions = community.best_partition(self.persona_graph, resolution=self.resolution)
        self.overlapping_partitions = {node: [] for node in self.graph.nodes()}
        for node, membership in self.partitions.items():
            self.overlapping_partitions[self.personality_map[node]].append(membership)

    def fit(self, graph):
        """
        Fitting an EgoNet clustering model.
        """
        self.graph = graph
        self._create_egonets()
        self._map_personalities()
        self._create_persona_graph()
        self._create_partitions()
