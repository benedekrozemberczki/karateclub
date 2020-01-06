import os
import numpy as np
import pandas as pd
import networkx as nx
from six.moves import urllib
import io
import json
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

    def __init__(self, dataset):
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/node_level/"

    def _dataset_reader(self, end):
        path = os.path.join(self.base_url, self.dataset, end)
        data = urllib.request.urlopen(path)
        return data
   
    def get_graph(self):
        data = self._dataset_reader("edges.csv")
        data = pandas_reader(data.read())
        graph = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")

        return graph

    def get_features(self):
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
        data = self._dataset_reader("target.csv")
        data = pandas_reader(data.read())
        target = np.array(data["target"])
        return target
