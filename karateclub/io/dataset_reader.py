import numpy as np
import pandas as pd
import networkx as nx


class LargeGraphReader(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/tree/master/dataset/node_level/"

    def get_dataset():
        graph = self.get_graph()
        features = self.get_features()
        target = self.get_target()
        return (graph, features, target)    
    def _get_graph():
        pass
    def _get_features():
        pass
    def _get_target():
        pass
