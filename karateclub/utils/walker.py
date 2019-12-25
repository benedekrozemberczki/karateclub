import random
from tqdm import tqdm
import networkx as nx

class RandomWalker:
    """
    Class to do fast first-order random walks.
    """
    def __init__(self, walk_length, walk_number):
        """
        Constructor for FirstOrderRandomWalker.
        :param graph: Nx graph object.
        :param args: Arguments object.
        """
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.walks = []

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        for _ in range(self.walk_length-1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs) > 0:
                walk = walk + random.sample(nebs, 1)
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self, graph):
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        :param graph: NetworkX graph.
        :return : Random walks.
        """
        self.graph = graph
        for node in tqdm(self.graph.nodes()):
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
        return self.walks
