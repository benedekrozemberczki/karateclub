import networkx as nx

"""General Estimator base class."""

class Estimator(object):
    """Estimator base class with constructor and public methods."""

    def __init__(self):
        """Creatinng an estimator."""
        pass

    def fit(self):
        """Fitting a model."""
        pass

    def get_embedding(self):
        """Getting the embeddings (graph or node level)."""
        return None

    def get_memberships(self):
        """Getting the membership dictionary."""
        return None

    def get_cluster_centers(self):
        """Getting the cluster centers."""
        return None

    def _check_networkx_graph(self, graph):
        try:
            if not isinstance(graph, nx.classes.graph.Graph):
                raise TypeError("This is not a NetworkX graph. Please see requirements.")
        except:
                exit("This is not a NetworkX graph. Please see requirements.")
   

    def _check_connectivity(self, graph):
        """Checking the connected nature of a single graph."""
        try:
            connected = nx.is_connected(graph)
            if not connected:
                raise ValueError("Graph is not connected. Please see requirements.")
        except:
            exit("Graph is not connected. Please see requirements.")


    def _check_directedness(self, graph):
        """Checking the undirected nature of a single graph."""
        try:
            directed = nx.is_directed(graph)
            if directed:
                raise ValueError("Graph is directed. Please see requirements.")
        except:
            exit("Graph is directed. Please see requirements.")


    def _check_indexing(self, graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])
        try:
           if numeric_indices != node_indices:
               raise ValueError("The node indexing is wrong. Please see requirements.")
        except:
           exit("The node indexing is wrong. Please see requirements.")     


    def _check_graph(self, graph):
        """Check the Karate Club assumptions about the graph."""
        self._check_connectivity(graph)
        self._check_directedness(graph)
        self._check_indexing(graph)


    def _check_graphs(self, graphs):
        """Check the Karate Club assumptions for a list of graphs."""
        try:
            for graph in graphs:
                self._check_graph(graph)
        except:
            exit("The graph list does not satisfy requirements.")

