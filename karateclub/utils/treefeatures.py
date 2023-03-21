import hashlib
import networkx as nx
from tqdm.auto import tqdm
from typing import List, Dict, Optional


class WeisfeilerLehmanHashing(object):
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        use_node_attribute (Optional[str]): Optional attribute name to be used.
        erase_base_feature (bool): Deleting the base features.
    """

    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        wl_iterations: int,
        use_node_attribute: Optional[str],
        erase_base_features: bool,
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.graph = graph
        self.use_node_attribute = use_node_attribute
        self.erase_base_features = erase_base_features
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        """
        Creating the features.
        """
        if self.use_node_attribute is not None:
            # We retrieve the features of the nodes with the attribute name
            # `feature` and assign them into a dictionary with structure:
            # {node_a_name: feature_of_node_a}
            # Nodes without this feature will not appear in the dictionary.
            features = nx.get_node_attributes(self.graph, self.use_node_attribute)

            # We check whether all nodes have the requested feature
            if len(features) != self.graph.number_of_nodes():
                missing_nodes = []
                # We find up to five missing nodes so to make
                # a more informative error message.
                for node in tqdm(
                    self.graph.nodes,
                    total=self.graph.number_of_nodes(),
                    leave=False,
                    dynamic_ncols=True,
                    desc="Searching for missing nodes"
                ):
                    if node not in features:
                        missing_nodes.append(node)
                    if len(missing_nodes) > 5:
                        break
                raise ValueError(
                    (
                        "We expected for ALL graph nodes to have a node "
                        "attribute name `{}` to be used as part of "
                        "the requested embedding algorithm, but only {} "
                        "out of {} nodes has the correct attribute. "
                        "Consider checking for typos and missing values, "
                        "and use some imputation technique as necessary. "
                        "Some of the nodes without the requested attribute "
                        "are: {}"
                    ).format(
                        self.use_node_attribute,
                        len(features),
                        self.graph.number_of_nodes(),
                        missing_nodes
                    )
                )
            # If so, we assign the feature set.
            self.features = features
        else:
            self.features = {
                node: self.graph.degree(node) for node in self.graph.nodes()
            }
        self.extracted_features = {k: [str(v)]
                                   for k, v in self.features.items()}

    def _erase_base_features(self):
        """
        Erasing the base features
        """
        for k, v in self.extracted_features.items():
            del self.extracted_features[k][0]

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + \
                sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features.
        """
        return [
            feature
            for node, features in self.extracted_features.items()
            for feature in features
        ]
