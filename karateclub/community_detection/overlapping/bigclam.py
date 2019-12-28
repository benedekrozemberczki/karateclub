import numpy as np
from tqdm import tqdm
import networkx as nx

class BigClam(object):
    r"""An implementation of `"DANMF" <https://smartyfh.com/Documents/18DANMF.pdf>`_
    from the CIKM '18 paper "Deep Autoencoder-like Nonnegative Matrix Factorization for
    Community Detection". The procedure uses telescopic non-negative matrix factorization
    in order to learn a cluster memmbership distribution over nodes. The method can be 
    used in an overlapping and non-overlapping way.

    Args:
        layers (list): Autoencoder layer sizes in a list of integers. Default [32, 8].
        pre_iterations (int): Number of pre-training epochs. Default 100.
        iterations (int): Number of training epochs. Default 100.
        seed (int): Random seed for weight initializations. Default 42.
        lamb (float): Regularization parameter. Default 0.01.
    """
    def __init__(self, dimensions=32, iterations=10):
        self.dimensions = dimensions
        self.iterations = iterations
