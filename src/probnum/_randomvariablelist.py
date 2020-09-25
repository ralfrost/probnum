import numpy as np
<<<<<<< HEAD
=======
import probnum.random_variables as pnrvs
>>>>>>> 41233c8941df9990fc2e8b0e78ddab6176f65786


class _RandomVariableList(list):
    """
    List of RandomVariables with convenient access to means, covariances, etc.

    Parameters
    ----------
    rv_list : :obj:`list` of :obj:`RandomVariable`
    """

    def __init__(self, rv_list):
        if not isinstance(rv_list, list):
            raise TypeError("RandomVariableList expects a list.")
<<<<<<< HEAD
        super().__init__(rv_list)

    def mean(self):
        return np.stack([rv.mean for rv in self])

    def cov(self):
        return np.stack([rv.cov for rv in self])

    def var(self):
        return np.stack([rv.var for rv in self])

=======

        # First element as a proxy for checking all elements
        if not isinstance(rv_list[0], pnrvs.RandomVariable):
            raise TypeError("RandomVariableList expects RandomVariable elements")
        super().__init__(rv_list)

    @property
    def mean(self):
        return np.stack([rv.mean for rv in self])

    @property
    def cov(self):
        return np.stack([rv.cov for rv in self])

    @property
    def var(self):
        return np.stack([rv.var for rv in self])

    @property
>>>>>>> 41233c8941df9990fc2e8b0e78ddab6176f65786
    def std(self):
        return np.stack([rv.std for rv in self])

    def __getitem__(self, idx):
        """Make sure to wrap the result into a _RandomVariableList if necessary"""
        result = super().__getitem__(idx)
        if isinstance(result, list):
            result = _RandomVariableList(result)
        return result
