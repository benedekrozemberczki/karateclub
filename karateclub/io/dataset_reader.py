import os
import numpy as np
import pandas as pd
import networkx as nx
from six.moves import urllib
import io

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

    def get_dataset(self):
        graph = self._get_graph()
        features = self._get_features()
        target = self._get_target()
        return (graph, features, target) 
   
    def _get_graph(self):
        graph_path = os.path.join(self.base_url, self.dataset, "edges.csv")
        data = urllib.request.urlopen(graph_path)
        data = pandas_reader(data.read())
        graph = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")
        return graph

    def _get_features(self):
        features_path = os.path.join(self.base_url, self.dataset, "features.json")

    def _get_target(self):
        target_path = os.path.join(self.base_url, self.dataset, "target.csv")
        data = urllib.request.urlopen(target_path)
        data = pandas_reader(data.read())
        target = np.array(data["target"])
        print(target)
        return target
