

class SymmNMF(Estimator):

    r"""An implementation of `"Symm-NMF" <https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589/13763>`_
    from the SDM'12 paper "Community Preserving Network Embedding".


    Args:
        dimensions (int): Number of dimensions. Default is 128.
        iterations (int): Number of power iterations. Default is 200.
        rho (float): ADMM tuning parameter. Default is 1.0.
    """
    def __init__(self, dimensions=128, iterations=200, rho=1.0):

        self.dimensions = dimensions
        self.iterations = iterations
        self.rho = rho

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        indices = np.argmax(self.H, axis=1)
        memberships = {i: membership for i, membership in enumerate(indices)}
        return memberships

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self.U
        return embedding

    def fit(self, graph):
        """
        Fitting an M-NMF clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
