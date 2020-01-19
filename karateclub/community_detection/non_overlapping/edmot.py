import community
import networkx as nx
from karateclub.estimator import Estimator

class EdMot(Estimator):
    r"""An implementation of `"Edge Motif Clustering" <https://arxiv.org/abs/1906.04560>`_
    from the KDD '19 paper "EdMot: An Edge Enhancement Approach for Motif-aware Community Detection". The tool first creates
    the graph of higher order motifs. This graph is clustered by the Louvain method. The resulting
    cluster memberships are stored as a dictionary.

    Args:
        component_count (int): Number of extracted motif hypergraph components. Default is 2.
        cutoff (int): Motif edge cut-off value. Default is 10.
    """
    def __init__(self, component_count=2, cutoff=10):
        self.component_count = component_count
        self.cutoff = cutoff

    def _overlap(self, node_1, node_2):
        """
        Calculating the neighbourhood overlap for a pair of nodes.

        Arg types:
            * **node_1** *(int)* - Source node 1.
            * **node_2** *(int)* - Source node 2.
        Return types:
            * **overlap** *(int)* - Neighbourhood overlap score.
        """
        nodes_1 = self.graph.neighbors(node_1)
        nodes_2 = self.graph.neighbors(node_2)
        overlap = len(set(nodes_1).intersection(set(nodes_2)))
        return overlap

    def _calculate_motifs(self):
        """
        Enumerating pairwise motif counts.
        """
        edges = [e for e in self.graph.edges() if self._overlap(e[0], e[1]) >= self.cutoff]
        self.motif_graph = nx.from_edgelist(edges)

    def _extract_components(self):
        """
        Extracting connected components from motif graph.
        """
        components = [c for c in sorted(nx.connected_components(self.motif_graph), key=len, reverse=True)]
        important_components = components[:self.component_count]
        self.blocks = [list(nodes) for nodes in important_components]

    def _fill_blocks(self):
        """
        Filling the dense blocks of the adjacency matrix.
        """
        new_edges = [(n_1, n_2) for nodes in self.blocks for n_1 in nodes for n_2 in nodes]
        new_graph = nx.from_edgelist(new_edges)
        self.graph = nx.disjoint_union(self.graph, new_graph)

    def fit(self, graph):
        """
        Fitting an Edge Motif clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        self._calculate_motifs()
        self._extract_components()
        self._fill_blocks()
        self.partition = community.best_partition(self.graph)

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dictionary of ints)* - Cluster memberships.
        """
        return self.partition
