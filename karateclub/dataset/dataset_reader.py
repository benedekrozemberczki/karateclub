import io
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import List
from six.moves import urllib
from scipy.sparse import coo_matrix


class GraphReader(object):
    r"""Class to read benchmark datasets for the community detection or node embedding task.

    Args:
        dataset (str): Dataset of interest, one of: 
            (:obj:`"facebook"`, :obj:`"twitch"`, :obj:`"wikipedia"`, :obj:`"github"`, :obj:`"lastfm"`, :obj:`"deezer"`). Default is 'wikipedia'.
    """
    def __init__(self, dataset: str="wikipedia"):
        assert dataset in ["wikipedia", "twitch", "github", "facebook", "lastfm", "deezer"], "Wrong dataset."
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/node_level/"

    def _pandas_reader(self, bytes):
        """
        Reading bytes as a Pandas dataframe.
        """
        tab = pd.read_csv(io.BytesIO(bytes),
                          encoding="utf8",
                          sep=",",
                          dtype={"switch": np.int32})
        return tab

    def _dataset_reader(self, end):
        """
        Reading the dataset from the web.
        """
        path = self.base_url + self.dataset + "/" + end
        data = urllib.request.urlopen(path).read()
        data = self._pandas_reader(data)
        return data
   
    def get_graph(self) -> nx.classes.graph.Graph:
        r"""Getting the graph.

        Return types:
            * **graph** *(NetworkX graph)* - Graph of interest.
        """
        data = self._dataset_reader("edges.csv")
        graph = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")
        return graph

    def get_features(self) -> coo_matrix:
        r"""Getting the node features Scipy matrix.

        Return types:
            * **features** *(COO Scipy array)* - Node feature matrix.
        """
        data = self._dataset_reader("features.csv")
        row = np.array(data["node_id"])
        col = np.array(data["feature_id"])
        values = np.array(data["value"])
        node_count = max(row) + 1
        feature_count = max(col) + 1
        shape = (node_count, feature_count)
        features = coo_matrix((values, (row, col)), shape=shape)
        return features

    def get_target(self) -> np.array:
        r"""Getting the class membership of nodes.

        Return types:
            * **target** *(Numpy array)* - Class membership vector.
        """
        data = self._dataset_reader("target.csv")
        target = np.array(data["target"])
        return target

class GraphSetReader(object):
    r"""Class to read benchmark datasets for the graph level embedding task.

    Args:
        dataset (str): Dataset of interest one of reddit10k. Default is 'reddit10k'.
    """
    def __init__(self, dataset: str="reddit10k"):
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/graph_level/"

    def _pandas_reader(self, bytes):
        """
        Reading bytes as a Pandas dataframe.
        """
        tab = pd.read_csv(io.BytesIO(bytes),
                          encoding="utf8",
                          sep=",",
                          dtype={"switch": np.int32})
        return tab

    def _dataset_reader(self, end):
        """
        Reading the dataset from the web.
        """
        path = self.base_url + self.dataset + "/" + end
        data = urllib.request.urlopen(path).read()
        return data

    def get_graphs(self) -> List[nx.classes.graph.Graph]:
        r"""Getting the graphs.

        Return types:
            * **graphs** *(List of NetworkX graphs)* - Graphs of interest.
        """
        graphs = self._dataset_reader("graphs.json")
        graphs = json.loads(graphs.decode())
        graphs = [nx.from_edgelist(graphs[str(i)]) for i in range(len(graphs))]
        return graphs

    def get_target(self) -> np.array:
        r"""Getting the class membership of graphs.

        Return types:
            * **target** *(Numpy array)* - Class membership vector.
        """
        data = self._dataset_reader("target.csv")
        data_tab = self._pandas_reader(data)
        target = np.array(data_tab["target"])
        return target
