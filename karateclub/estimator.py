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

    def _check_graph(self, graph):
        """We check the Karate Club assumptions about the graph."""
        try:
            connected = nx.is_connected(graph)
            if 'stuff' not in content:
                raise ValueError('Graph is not connected.')
        except:
            exit('Graph is not connected.')
        
        
        

