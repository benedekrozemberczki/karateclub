import hashlib
import networkx as nx

class WeisfeilerLehmanHashing(object):
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        features (dict of strings): Feature hash map.
        iterations (int): Number of WL iterations.
    """
    def __init__(self, graph, wl_iterations, attributed):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.graph = graph
        self.attributed = attributed
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        """
        Creating the features.
        """
        if self.attributed:
            self.features = nx.get_node_attributes(self.graph, 'feature')
        else:
            self.features = {node: self.graph.degree(node) for node in self.graph.nodes()}

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {k: self.extracted_features[k] + [v] for k, v in new_features.items()}
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()

    def get_node_features(self):
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self):
        """
        Return the graph level features.
        """
        return [feature for node, features in self.extracted_features.items() for feature in features]
