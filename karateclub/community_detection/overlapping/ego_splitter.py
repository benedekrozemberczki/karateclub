import community
import networkx as nx
from typing import Dict, Optional
from karateclub.estimator import Estimator


class EgoNetSplitter(Estimator):
    r"""An implementation of `"Ego-Splitting" <https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf>`_
    from the KDD '17 paper "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters". The tool first creates
    the ego-nets of nodes. A persona-graph is created which is clustered by the Louvain method. The resulting overlapping
    cluster memberships are stored as a dictionary.

    Args:
        resolution (float): Resolution parameter of Python Louvain. Default 1.0.
        local_resolution (float): Local resolution parameter of Python Louvain. Default 2.0.
        seed (int): Random seed value. Default is 42.
        weight (str): the key in the graph to use as weight. Default to 'weight'. Specify None to force using an unweighted version of the graph.
    """

    def __init__(
            self,
            resolution: float = 1.0,
            local_resolution: float = 2.0,
            seed: int = 42,
            weight: Optional[str] = "weight"
    ):
        self.resolution = resolution
        self.local_resolution = local_resolution
        self.seed = seed
        self.weight = weight

    def _create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it (using Louvain method).

        Arg types:
            * **node** *(int)* - Node ID for ego-net (ego node).
        """
        neighbors = list(self.graph.neighbors(node))
        ego_net = self.graph.subgraph(neighbors).copy()

        if self.weight is None:
            partition = community.best_partition(
                ego_net,
                resolution=self.local_resolution,
                random_state=self.seed
            )
        else:
            partition = community.best_partition(
                ego_net,
                resolution=self.local_resolution,
                weight=self.weight,
                random_state=self.seed
            )

        community_mapping = {}
        personalities = []
        for comm_id in set(partition.values()):
            personalities.append(self.index)
            community_mapping[comm_id] = self.index
            self.index += 1

        self.components[node] = {
            n: community_mapping[partition[n]] for n in ego_net.nodes()
        }
        self.personalities[node] = personalities

    def _create_egonets(self):
        """
        Creating an ego-net for each node.
        """
        self.components = {}
        self.personalities = {}
        self.index = 0
        for node in self.graph.nodes():
            self._create_egonet(node)

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {
            p: n for n in self.graph.nodes() for p in self.personalities[n]
        }

    def _get_new_edge_ids(self, edge):
        """
        Getting the new edge identifiers.

        Arg types:
            * **edge** *(list of ints)* - Edge being mapped to the new identifiers.
        """
        if self.weight is None or edge[2] is None:
            return (
                self.components[edge[0]][edge[1]],
                self.components[edge[1]][edge[0]],
            )
        else:
            return (
                self.components[edge[0]][edge[1]],
                self.components[edge[1]][edge[0]],
                {self.weight: edge[2]},
            )

    def _create_persona_graph(self):
        """
        Create a persona graph using the ego-net components.
        """
        if self.weight is None:
            self.persona_graph_edges = [
                self._get_new_edge_ids(edge) for edge in self.graph.edges()
            ]
        else:
            self.persona_graph_edges = [
                self._get_new_edge_ids(edge)
                for edge in self.graph.edges(data=self.weight)
            ]

        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)

    def _create_partitions(self):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        if self.weight is None:
            self.partitions = community.best_partition(
                self.persona_graph, resolution=self.resolution
            )
        else:
            self.partitions = community.best_partition(
                self.persona_graph, resolution=self.resolution, weight=self.weight
            )

        self.overlapping_partitions = {node: set() for node in self.graph.nodes()}
        for persona_id, cluster_id in self.partitions.items():
            original_node = self.personality_map[persona_id]
            self.overlapping_partitions[original_node].add(cluster_id)

        self.overlapping_partitions = {
            node: sorted(list(clusters))
            for node, clusters in self.overlapping_partitions.items()
        }

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting an Ego-Splitter clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        self.graph = graph
        self._create_egonets()
        self._map_personalities()
        self._create_persona_graph()
        self._create_partitions()

    def get_memberships(self) -> Dict[int, int]:
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dictionary of lists)* - Cluster memberships.
        """
        return self.overlapping_partitions

    def get_clusters(self) -> Dict[int, list[int]]:
        r"""get clustering results, show how many clusters there are and which clients are included in each cluster.

        Return types:
            * **clusters** *(dictionary)* - Clustering results, where the key is the cluster ID and the value is the list of nodes included in that cluster.
        """
        clusters = {}
        for node_id, cluster_ids in self.overlapping_partitions.items():
            for cluster_id in cluster_ids:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node_id)
        return clusters