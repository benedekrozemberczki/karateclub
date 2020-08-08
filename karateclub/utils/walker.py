import random
import numpy as np
import networkx as nx

class RandomWalker:
    """
    Class to do fast first-order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
    """
    def __init__(self, walk_length: int, walk_number: int):
        self.walk_length = walk_length
        self.walk_number = walk_number

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the random walk.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
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

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph
        for node in self.graph.nodes():
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)

class BiasedRandomWalker:
    """
    Class to do biased second order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
        p (float): Return parameter (1/p transition probability) to move towards from previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
    """
    def __init__(self, walk_length: int, walk_number: int, p: float, q: float):
        self.walk_length = walk_length
        self.walk_number = walk_number
        try: 
            _ = 1/p
        except ZeroDivisionError:
            raise ValueError("The value of p is too small or zero to be used in 1/p.")
        self.p = p
        try: 
            _ = 1/q
        except ZeroDivisionError:
            raise ValueError("The value of q is too small or zero to be used in 1/q.")
        self.q = q

    def do_walk(self, node):
        """
        Doing a single truncated second order random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the random walk.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        walk = [node]
        previous_node = None
        previous_node_neighbors = []
        for _ in range(self.walk_length-1):
            current_node = walk[-1]
            current_node_neighbors = np.array(list(self.graph.neighbors(current_node)))
            probability = np.array([1/self.q] * len(current_node_neighbors), dtype=float)
            probability[current_node_neighbors==previous_node] = 1/self.p
            probability[(np.isin(current_node_neighbors, previous_node_neighbors))] = 1
            norm_probability = probability/sum(probability)
            selected = np.random.choice(current_node_neighbors, 1, p=norm_probability)[0]
            walk.append(selected)
            previous_node_neighbors = current_node_neighbors
            previous_node = current_node
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self, graph):
        """
        Doing a fixed number of truncated random walk from every node in the graph.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph
        for node in self.graph.nodes():
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
