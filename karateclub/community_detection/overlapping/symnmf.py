

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

