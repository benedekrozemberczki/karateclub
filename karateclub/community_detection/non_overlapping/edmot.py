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
        cutoff (int): Motif edge cut-off value. Default is 50.
    """
    def __init__(self, component_count=2, cutoff=50):
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
        components = [c for c in nx.connected_components(self.motif_graph)]
        components = [[len(c), c] for c in components]
        components.sort(key=lambda x: x[0], reverse=True)
        important_components = [components[comp][1] for comp
                                in range(self.component_count if len(components)>=self.component_count else len(components))]
        self.blocks = [list(graph) for graph in important_components]

    def _fill_blocks(self):
        """
        Filling the dense blocks of the adjacency matrix.
        """
        new_edges = [(n_1, n_2) for nodes in self.blocks for n_1 in nodes for n_2 in nodes if n_1!= n_2]
        self.graph.add_edges_from(new_edges)  

    def fit(self, graph):
        """
        Fitting an Edge Motif clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._check_graph(graph)
        self.graph = graph
        self._calculate_motifs()
        self._extract_components()
        self._fill_blocks()
        self.partition = community.best_partition(self.graph, randomize=True)

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dictionary of ints)* - Cluster memberships.
        """
        return self.partition
