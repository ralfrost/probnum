"""
Normal distribution.

This module implements the Gaussian distribution in its univariate,
multi-variate, matrix-variate and operator-variate forms.
"""
import operator
import warnings

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util
import scipy.linalg

from probnum.linalg import linops
from probnum.prob.distributions.distribution import Distribution
from probnum.prob.distributions.dirac import Dirac


class Normal(Distribution):
    """
    The normal distribution.

    The Gaussian distribution is ubiquitous in probability theory, since
    it is the final and stable or equilibrium distribution to which
    other distributions gravitate under a wide variety of smooth
    operations, e.g., convolutions and stochastic transformations.
    One example of this is the central limit theorem. The Gaussian
    distribution is also attractive from a numerical point of view as it
    is maintained through many transformations (e.g. it is stable).

    Parameters
    ----------
    mean : float or array-like or LinearOperator
        Mean of the normal distribution.

    cov : float or array-like or LinearOperator
        (Co-)variance of the normal distribution.

    random_state : None or int or :class:`~numpy.random.RandomState` instance, optional
        This parameter defines the RandomState object to
        use for drawing realizations from this
        distribution. Think of it like a random seed.
        If None (or np.random), the global
        np.random state is used. If integer, it is used to
        seed the local
        :class:`~numpy.random.RandomState` instance.
        Default is None.

    See Also
    --------
    Distribution : Class representing general probability distributions.

    Examples
    --------
    >>> from probnum.prob import Normal
    >>> N = Normal(mean=0.5, cov=1.0)
    >>> N.parameters
    {'mean': 0.5, 'cov': 1.0}
    """

    def __new__(cls, mean=0.0, cov=1.0, random_state=None):
        """
        Factory method for normal subclasses.

        Checks shape/type of mean and cov and returns the corresponding
        type of Normal distribution:
            * _UnivariateNormal
            * _MultivariateNormal
            * _MatrixvariateNormal
            * _SymmetricKroneckerIdenticalFactorsNormal
            * _OperatorvariateNormal
        If neither applies, a ValueError is raised.
        """
        if cls is Normal:
            if _both_are_univariate(mean, cov):
                return _UnivariateNormal(mean, cov, random_state)
            elif _both_are_multivariate(mean, cov):
                return _MultivariateNormal(mean, cov, random_state)
            elif _both_are_matrixvariate(mean, cov):
                return _MatrixvariateNormal(mean, cov, random_state)
            elif _both_are_symmkronidfactors(mean, cov):
                return _SymmetricKroneckerIdenticalFactorsNormal(
                    mean, cov, random_state
                )
            elif _both_are_operatorvariate(mean, cov):
                return _OperatorvariateNormal(mean, cov, random_state)
            else:
                errmsg = (
                    "Cannot instantiate normal distribution with mean of "
                    + "type {} and ".format(mean.__class__.__name__)
                    + "kernels of "
                    + "type {}.".format(cov.__class__.__name__)
                )
                raise ValueError(errmsg)
        else:
            return super().__new__(cls)

    def __init__(self, mean=0.0, cov=1.0, random_state=None):
        super().__init__(
            parameters={"mean": mean, "cov": cov},
            dtype=float,
            random_state=random_state,
        )

    def mean(self):
        return self.parameters["mean"]

    def cov(self):
        return self.parameters["cov"]

    def var(self):
        raise NotImplementedError

    # Binary arithmetic

    def __add__(self, other):
        """
        Addition of Gaussian random variables.
        """
        if isinstance(other, Dirac):
            return Normal(
                mean=self.mean() + other.mean(),
                cov=self.cov(),
                random_state=self.random_state,
            )
        elif isinstance(other, type(self)):
            if self.random_state is not None and other.random_state is not None:
                warnings.warn(
                    "When adding random variables with set random states only the first is preserved."
                )
            try:
                return Normal(
                    mean=self.mean() + other.mean(),
                    cov=self.cov() + other.cov(),
                    random_state=self.random_state,
                )
            except ValueError:
                return NotImplemented
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        Subtraction of Gaussian random variables.
        """
        if isinstance(other, Dirac):
            return self + (-other)
        elif isinstance(other, type(self)):
            if self.random_state is not None and other.random_state is not None:
                warnings.warn(
                    "When adding random variables with set random states only the first is preserved."
                )
            try:
                return Normal(
                    mean=self.mean() - other.mean(),
                    cov=self.cov() + other.cov(),
                    random_state=self.random_state,
                )
            except ValueError:
                return NotImplemented
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            if delta == 0:
                return Dirac(support=0 * self.mean(), random_state=self.random_state)
            else:
                return Normal(
                    mean=self.mean() * delta,
                    cov=self.cov() * delta ** 2,
                    random_state=self.random_state,
                )
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if other == 0:
            raise ZeroDivisionError("Division by zero not supported.")
        else:
            if isinstance(other, Dirac):
                return self * operator.inv(other)
            else:
                return NotImplemented

    def __pow__(self, power, modulo=None):
        return NotImplemented

    # Binary arithmetic with reflected operands

    def __radd__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return self + other
        else:
            return NotImplemented

    def __rsub__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return operator.neg(self) + other
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return self * other
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            newmean = delta @ self.mean()
            newcov = delta @ (self.cov() @ delta.transpose())
            return Normal(mean=newmean, cov=newcov, random_state=self.random_state)
        return NotImplemented

    def __rtruediv__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return operator.inv(self) * other
        else:
            return NotImplemented

    def __rpow__(self, power, modulo=None):
        return NotImplemented

    # Unary arithmetic

    def __neg__(self):
        """
        Negation of r.v.
        """
        try:
            return Normal(
                mean=-self.mean(), cov=self.cov(), random_state=self.random_state
            )
        except Exception:
            return NotImplemented

    def __pos__(self):
        try:
            return Normal(
                mean=operator.pos(self.mean()),
                cov=self.cov(),
                random_state=self.random_state,
            )
        except Exception:
            return NotImplemented

    def __abs__(self):
        try:
            # todo: add absolute moments of normal (see: https://arxiv.org/pdf/1209.4340.pdf)
            return Distribution(
                parameters={},
                sample=lambda size: operator.abs(self.sample(size=size)),
                random_state=self.random_state,
            )
        except Exception:
            return NotImplemented

    def __invert__(self):
        try:
            return Distribution(
                parameters={},
                sample=lambda size: operator.abs(self.sample(size=size)),
                random_state=self.random_state,
            )
        except Exception:
            return NotImplemented


def _both_are_univariate(mean, cov):
    """
    Checks whether mean and kernels correspond to the
    UNIVARIATE normal distribution.
    """
    both_are_scalars = np.isscalar(mean) and np.isscalar(cov)
    mean_shape_dim1 = np.shape(mean) in [(1, 1), (1,), ()]
    cov_shape_dim1 = np.shape(cov) in [(1, 1), (1,), ()]
    both_in_dim1shapes = mean_shape_dim1 and cov_shape_dim1
    return both_are_scalars or both_in_dim1shapes


def _both_are_multivariate(mean, cov):
    """
    Checks whether mean and kernels correspond to the
    MULTI- or MATRIXVARIATE normal distribution.
    """
    mean_is_multivar = isinstance(mean, (np.ndarray, scipy.sparse.spmatrix,))
    cov_is_multivar = isinstance(cov, (np.ndarray, scipy.sparse.spmatrix,))
    return mean_is_multivar and cov_is_multivar and len(mean.shape) == 1


def _both_are_matrixvariate(mean, cov):
    """
    Checks whether mean and kernels correspond to the
    MULTI- or MATRIXVARIATE normal distribution.
    """
    mean_is_multivar = isinstance(mean, (np.ndarray, scipy.sparse.spmatrix,))
    cov_is_multivar = isinstance(cov, (np.ndarray, scipy.sparse.spmatrix,))
    return mean_is_multivar and cov_is_multivar and len(mean.shape) > 1


def _both_are_symmkronidfactors(mean, cov):
    """
    Checks whether mean OR (!) kernels correspond to the
    OPERATORVARIATE normal distribution.
    """
    mean_is_opvariate = isinstance(mean, scipy.sparse.linalg.LinearOperator)
    cov_is_opvariate = isinstance(cov, scipy.sparse.linalg.LinearOperator)
    if mean_is_opvariate or cov_is_opvariate:
        return isinstance(cov, linops.SymmetricKronecker) and cov._ABequal
    else:
        return False


def _both_are_operatorvariate(mean, cov):
    """
    Checks whether mean OR (!) kernels correspond to the
    OPERATORVARIATE normal distribution.
    """
    mean_is_opvariate = isinstance(mean, scipy.sparse.linalg.LinearOperator)
    cov_is_opvariate = isinstance(cov, scipy.sparse.linalg.LinearOperator)
    return mean_is_opvariate or cov_is_opvariate


class _UnivariateNormal(Normal):
    """
    The univariate normal distribution.
    """

    def __init__(self, mean=0.0, cov=1.0, random_state=None):
        super().__init__(mean=float(mean), cov=float(cov), random_state=random_state)

    def var(self):
        return self.cov()

    def pdf(self, x):
        return scipy.stats.norm.pdf(x, loc=self.mean(), scale=self.std())

    def logpdf(self, x):
        return scipy.stats.norm.logpdf(x, loc=self.mean(), scale=self.std())

    def cdf(self, x):
        return scipy.stats.norm.cdf(x, loc=self.mean(), scale=self.std())

    def logcdf(self, x):
        return scipy.stats.norm.logcdf(x, loc=self.mean(), scale=self.std())

    def sample(self, size=()):
        return scipy.stats.norm.rvs(
            loc=self.mean(), scale=self.std(), size=size, random_state=self.random_state
        )

    def reshape(self, newshape):
        if np.prod(newshape) != 1:
            raise ValueError(
                f"Cannot reshape distribution with shape {self.shape} into shape {newshape}."
            )
        self.parameters["mean"] = np.reshape(self.parameters["mean"], newshape=newshape)
        self.parameters["cov"] = np.reshape(self.parameters["cov"], newshape=newshape)
        self._shape = newshape


class _MultivariateNormal(Normal):
    """
    The multivariate normal distribution.
    """

    def __init__(self, mean, cov, random_state=None):
        """
        Checks if mean and kernels have matching shapes before
        initialising.
        """
        meandim = np.prod(mean.shape)
        if len(cov.shape) != 2:
            raise ValueError("Covariance must be a 2D matrix " "or linear operator.")
        if meandim != cov.shape[0] or meandim != cov.shape[1]:
            raise ValueError(
                "Shape mismatch of mean and kernels. Total "
                "number of elements of the mean must match the "
                "first and second dimension of the kernels."
            )
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def var(self):
        return np.diag(self.cov())

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean(), cov=self.cov())

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(
            x, mean=self.mean(), cov=self.cov()
        )

    def cdf(self, x):
        return scipy.stats.multivariate_normal.cdf(x, mean=self.mean(), cov=self.cov())

    def logcdf(self, x):
        return scipy.stats.multivariate_normal.logcdf(
            x, mean=self.mean(), cov=self.cov()
        )

    def sample(self, size=()):
        return scipy.stats.multivariate_normal.rvs(
            mean=self.mean(), cov=self.cov(), size=size, random_state=self.random_state
        )

    def reshape(self, newshape):
        raise NotImplementedError

    # Arithmetic Operations

    def __matmul__(self, other):
        """
        Only works if other is a Dirac, which implies
        matrix-multiplication with some constant.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            newmean = self.mean() @ delta
            newcov = delta.T @ (self.cov() @ delta)
            if np.isscalar(newmean) and np.isscalar(newcov):
                return _UnivariateNormal(
                    mean=newmean, cov=newcov, random_state=self.random_state
                )
            else:
                return _MultivariateNormal(
                    mean=newmean, cov=newcov, random_state=self.random_state
                )
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        Only works if other is a Dirac, which implies
        matrix-multiplication with some constant.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            newmean = delta @ self.mean()
            newcov = delta @ (self.cov() @ delta.T)
            if np.isscalar(newmean) and np.isscalar(newcov):
                return _UnivariateNormal(
                    mean=newmean, cov=newcov, random_state=self.random_state
                )
            else:
                return _MultivariateNormal(
                    mean=newmean, cov=newcov, random_state=self.random_state
                )
        else:
            return NotImplemented


class _MatrixvariateNormal(Normal):
    """
    The matrix-variate normal distribution.
    """

    def __init__(self, mean, cov, random_state=None):
        """
        Checks if mean and kernels have matching shapes before
        initialising.
        """
        _mean_dim = np.prod(mean.shape)
        if len(cov.shape) != 2:
            raise ValueError("Covariance must be a 2D matrix.")
        if _mean_dim != cov.shape[0] or _mean_dim != cov.shape[1]:
            raise ValueError(
                "Shape mismatch of mean and kernels. Total "
                "number of elements of the mean must match the "
                "first and second dimension of the kernels."
            )
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def var(self):
        return np.diag(self.cov())

    def pdf(self, x):
        pdf_ravelled = scipy.stats.multivariate_normal.pdf(
            x.ravel(), mean=self.mean().ravel(), cov=self.cov()
        )
        return pdf_ravelled.reshape(newshape=self.shape)

    def logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError

    
    def sample(self, size=()):
        ravelled = scipy.stats.multivariate_normal.rvs(
            mean=self.mean().ravel(),
            cov=self.cov(),
            size=size,
            random_state=self.random_state,
        )
        return ravelled.reshape(self.shape)

    def reshape(self, newshape):
        if np.prod(newshape) != np.prod(self.shape):
            raise ValueError(
                f"Cannot reshape distribution with shape {self.shape} into shape {newshape}."
            )
        self.parameters["mean"] = np.reshape(self.parameters["mean"], newshape=newshape)
        self.parameters["cov"] = np.reshape(self.parameters["cov"], newshape=newshape)
        self._shape = newshape

    # Arithmetic Operations
    # TODO: implement special rules for matrix-variate RVs and Kronecker
    #  structured covariances (see e.g. p.64 Thm. 2.3.10 of Gupta:
    #  Matrix-variate Distributions)

    def __matmul__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            raise NotImplementedError
        return NotImplemented


class _OperatorvariateNormal(Normal):
    """
    A normal distribution over finite dimensional linear operators.
    """

    def __init__(self, mean, cov, random_state=None):
        """
        Checks shapes of mean and cov depending on the type
        of operator that cov is before initialising.
        """
        self._mean_dim = np.prod(mean.shape)
        if isinstance(cov, linops.Kronecker):
            _check_shapes_if_kronecker(mean, cov)
        elif isinstance(cov, linops.SymmetricKronecker):
            _check_shapes_if_symmetric_kronecker(mean, cov)
        elif self._mean_dim != cov.shape[0] or self._mean_dim != cov.shape[1]:
            raise ValueError("Shape mismatch of mean and kernels.")
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def var(self):
        return linops.Diagonal(Op=self.cov())

    def _params_todense(self):
        """Returns the mean and kernels of a distribution as dense matrices."""
        if isinstance(self.mean(), linops.LinearOperator):
            mean = self.mean().todense()
        else:
            mean = self.mean()
        if isinstance(self.cov(), linops.LinearOperator):
            cov = self.cov().todense()
        else:
            cov = self.cov()
        return mean, cov

    def _cov_cholesky(self, zeros=True):
        '''Returns the lower left cholesky factors of the covariance. Set zeros to True if the upper part of the matrices should be zeros.'''
        cov = self.cov()
        if zeros:
            return (scipy.linalg.cholesky(cov.A.todense(),lower=True), scipy.linalg.cholesky(cov.B.todense(),lower=True))
        else:
            (L_V, lower)=scipy.linalg.cho_factor(cov.A.todense(), lower=True)    
            (L_W, lower)=scipy.linalg.cho_factor(cov.B.todense(), lower=True)    
            return (L_V, L_W)

    def pdf(self, x):
        cov_factors = self._cov_cholesky(zeros=False)
        return np.exp( self._calculate_logpdf(cov_factors, x) )

    def logpdf(self, x):
        cov_factors = self._cov_cholesky(zeros=False)
        return self._calculate_logpdf(cov_factors, x)

    def _calculate_logpdf(self, cov_factors, x):
        (L_V,L_W) = cov_factors

        #test
        print(np.allclose(np.tril(L_V)@np.tril(L_V).T-self.cov().A,zeros(self.cov().A.shape)))
        (m,n) = self.mean().shape
        dev = x-self.mean()
        ln_2pi = np.log(2 * np.pi)
        det_L_V = np.trace(L_V) #determinant of triangular matrices is equal to trace
        det_L_W = np.trace(L_W) 
        ln_detcov = m*np.log(det_L_V) + n*np.log(det_L_W)
        dev_reshaped = self._reshape_from_vec( dev, (n,m) )
        Q = scipy.linalg.cho_solve((L_W, True), dev_reshaped)
        R = scipy.linalg.cho_solve((L_V, True), Q.T, overwrite_b=True)
        print("dev = ",dev)
        print("R = ",R)
        y = np.dot(dev.reshape((m*n)), R.T.reshape((m*n)))

        #y = self._devT_covINV_dev(dev,cov)
        return -0.5 * (self._mean_dim*ln_2pi + ln_detcov + y)
    
    def _reshape_from_vec(self, A, shape):
        '''Returns matrix of shape retrieved of the vectorization of A. Works only for 2D-arrays!'''
        return A.T.reshape( (shape[1],shape[0]) ).T

    def _devT_covINV_dev(self, dev, cov):
        '''
        Calculates dev.T*cov^-1+dev by solving linear equation systems assuming that cov is a kronecker-product represented by two lower left triangular matrices as a result of cholesky
        '''
        L_V = cov.A.todense()
        L_W = cov.B.todense()
        T = np.linalg.solve_triangular(L_V, dev, trans=0, lower=True, overwrite_b=False)
        S = np.linalg.solve_triangular(L_V, T, trans=1, lower=False, overwrite_b=True)
        R = np.linalg.solve_triangular(L_W, S, trans=0, lower=True, overwrite_b=True)
        Q = np.linalg.solve_triangular(L_W, R, trans=1, lower=False, overwrite_b=True)
        return np.dot(Q,dev)

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError
    
    def sample(self, val=((),1)):
        (size, sel) = val
        cov_factors = self._cov_cholesky(zeros=True)
        print("Faktoren = ", cov_factors)
        (m,n) = self.mean().shape

        if isinstance(size, (int, np.integer)):
            shape = [size]
        else:
            shape = list(size)

        shape.extend([n,m])
        print("randoms_shape = ",shape)
        randoms = np.random.standard_normal(shape)
        if sel == 1 and isinstance(cov_factors[0], np.ndarray):
            stacked_product = self._stacked_matvec_ndarray(cov_factors, randoms)
        else:
            stacked_product = self._stacked_matvec_linop(cov_factors, randoms)
        return self.mean() + self._stacked_transpose(stacked_product)

    def _stacked_matvec_ndarray(self, cov_factors, vec_stack):
        """ Using (A (x) B)vec(X) = vec(AXB^T). vec_stack and output are actually stacks of matrices."""
        print("You are using method for ndarrays.")
        (L_V,L_W) = cov_factors
        return L_W @ vec_stack @ L_V.T 

    def _stacked_matvec_linop(self, cov_factors, vec_stack):
        """ Using (A (x) B)vec(X) = vec(AXB^T). vec_stack and output are actually stacks of matrices."""
        print("You are using method for linear operators.")
        (L_V,L_W) = cov_factors
        return np.array([L_W @ X @ L_V.T for X in vec_stack])
    
    def _stacked_transpose(self, A):
        n = len(A.shape)
        order = np.append(np.arange(n-2),[n-1,n-2])
        return np.transpose(A, axes=order)
    
    def sample_test(self, size=()):
        print("You are using the old method.")
        mean, cov = self._params_todense()
        samples_ravelled = scipy.stats.multivariate_normal.rvs(
            mean=mean.ravel(), cov=cov, size=size, random_state=self.random_state
        )
        return samples_ravelled.reshape(samples_ravelled.shape[:-1] + self.mean().shape)

    def reshape(self, newshape):
        raise NotImplementedError

    # Arithmetic Operations

    # TODO: implement special rules for matrix-variate RVs and Kronecker structured covariances
    #  (see e.g. p.64 Thm. 2.3.10 of Gupta: Matrix-variate Distributions)
    def __matmul__(self, other):
        if isinstance(other, Dirac):
            othermean = other.mean()
            delta = linops.Kronecker(linops.Identity(othermean.shape[0]), othermean)
            return Normal(
                mean=self.mean() @ othermean,
                cov=delta.T @ (self.cov() @ delta),
                random_state=self.random_state,
            )
        return NotImplemented


def _check_shapes_if_kronecker(mean, cov):
    """
    If mean has dimension (m x n) then kernels factors must be
    (m x m) and (n x n)
    """
    m, n = mean.shape
    if (
        m != cov.A.shape[0]
        or m != cov.A.shape[1]
        or n != cov.B.shape[0]
        or n != cov.B.shape[1]
    ):
        raise ValueError(
            "Kronecker structured kernels must have "
            "factors with the same shape as the mean."
        )


class _SymmetricKroneckerIdenticalFactorsNormal(_OperatorvariateNormal):
    """
    Normal distribution with symmetric Kronecker structured
    kernels with identical factors V (x)_s V.
    """

    def __init__(self, mean, cov, random_state=None):
        _check_shapes_if_symmetric_kronecker(mean, cov)
        self._n = mean.shape[1]
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def sample(self, size=()):

        # Draw standard normal samples
        if np.isscalar(size):
            size = [size]
        size_sample = [self._n * self._n] + list(size)
        stdnormal_samples = scipy.stats.norm.rvs(
            size=size_sample, random_state=self.random_state
        )

        # Cholesky decomposition
        eps = 10 ** -12  # damping needed to avoid negative definite covariances
        cholA = scipy.linalg.cholesky(
            self.cov().A.todense() + eps * np.eye(self._n), lower=True
        )

        # Scale and shift
        # TODO: can we avoid todense here and just return operator samples?
        if isinstance(self.mean(), scipy.sparse.linalg.LinearOperator):
            mean = self.mean().todense()
        else:
            mean = self.mean()

        # Appendix E: Bartels, S., Probabilistic Linear Algebra, PhD Thesis 2019
        samples_scaled = linops.Symmetrize(dim=self._n) @ (
            linops.Kronecker(A=cholA, B=cholA) @ stdnormal_samples
        )

        return mean[None, :, :] + samples_scaled.T.reshape(-1, self._n, self._n)


def _check_shapes_if_symmetric_kronecker(mean, cov):
    """
    Mean has to be square. If mean has dimension (n x n) then kernels
    factors must be (n x n).
    """
    m, n = mean.shape
    if m != n or n != cov.A.shape[0] or n != cov.B.shape[1]:
        raise ValueError(
            "Normal distributions with symmetric "
            "Kronecker structured kernels must "
            "have square mean and square "
            "kernels factors with matching "
            "dimensions."
        )
