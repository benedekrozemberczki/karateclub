import numpy as np
import networkx as nx

class SINE(Estimator):
    r"""An implementation of `"SINE" <https://shiruipan.github.io/publication/yang-binarized-2018/yang-binarized-2018.pdf>`_
    from the ICDM '18 paper "Binarized Attributed Network Embedding Class". The 
    procedure first calculates the truncated SVD of an adjacency - feature matrix
    product. This matrix is further decomposed by a binary CCD based technique. 
       
    Args:
        dimensions (int): Number of embedding dimensions. Default is 32.
        svd_iterations (int): SVD iteration count. Default is 20.
        seed (int): Random seed. Default is 42.
        alpha (float): Kernel matrix inversion parameter. Default is 0.3. 
        iterations (int): Matrix decomposition iterations. Default is 100.
        binarization_iterations (int): Binarization iterations. Default is 20.
    """
    def __init__(self, dimensions=32, svd_iterations=20, seed=42, alpha=0.3,
                 iterations=100, binarization_iterations=20):
        self.dimensions = dimensions
        self.svd_iterations = svd_iterations
        self.seed = seed
        self.alpha = alpha
        self.iterations = iterations
        self.binarization_iterations = binarization_iterations

    def fit(self, graph, X):
        """
        Fitting a SINE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Scipy COO array)* - The matrix of node features.
        """
        pass


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = None
        return embedding
