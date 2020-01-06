import io
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from six.moves import urllib
from scipy.sparse import coo_matrix

def pandas_reader(bytes):
    """
    """
    tab = pd.read_csv(io.BytesIO(bytes),
                      encoding="utf8",
                      sep=",",
                      dtype={"switch": np.int32})
    return tab

class GraphReader(object):
    r"""Class to read benchmark datasets for the community detection or node embedding task.

    Args:
        dataset (str): Dataset of interest  on of facebook/wikipedia/github/twitch. Default is 'wikipedia'.
    """
    def __init__(self, dataset="wikipedia"):
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/node_level/"

    def _dataset_reader(self, end):
        """
        Reading the dataset from the web.
        """
        path = os.path.join(self.base_url, self.dataset, end)
        data = urllib.request.urlopen(path)
        return data
   
    def get_graph(self):
        r"""Getting the graph.

        Return types:
            * **graph** *(NetworkX graph)* - Graph of interest.
        """
        data = self._dataset_reader("edges.csv")
        data = pandas_reader(data.read())
        graph = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")
        return graph

    def get_features(self):
        r"""Getting the node features Scipy matrix.

        Return types:
            * **features** *(COO Scipy array)* - Node feature matrix.
        """
        data = self._dataset_reader("features.json")
        data = json.loads(data.read().decode())
        row = [int(k) for k, v in data.items() for val in v]
        col = [val for k, v in data.items() for val in v]
        values = np.ones(len(row))
        node_count = max(row) + 1
        feature_count = max(col) + 1
        shape = (node_count, feature_count)
        features = coo_matrix((values, (row, col)), shape=shape)
        return features

    def get_target(self):
        r"""Getting the class membership of nodes.

        Return types:
            * **target** *(Numpy array)* - Class membership vector.
        """
        data = self._dataset_reader("target.csv")
        data = pandas_reader(data.read())
        target = np.array(data["target"])
        return target
