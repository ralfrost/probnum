""" This module implements normally distributed or Gaussian random variables. """

from typing import Callable, Optional, Union

import numpy as np
import scipy.linalg
from scipy.linalg import interpolative
import scipy.stats

from probnum import utils as _utils
from probnum.linalg import linops
from probnum.type import (
    ShapeType,
    # Argument Types
    ArrayLikeGetitemArgType,
    FloatArgType,
    RandomStateArgType,
    ShapeArgType,
)

from . import _random_variable

from scipy.spatial.distance import squareform

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


COV_CHOLESKY_DAMPING = 10 ** -12


_ValueType = Union[np.floating, np.ndarray, linops.LinearOperator]


class Normal(_random_variable.ContinuousRandomVariable[_ValueType]):
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
    >>> from probnum import random_variables as rvs
    >>> N = rvs.Normal(mean=0.5, cov=1.0)
    >>> N.parameters
    {'mean': 0.5, 'cov': 1.0}
    """

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def __init__(
        self,
        mean: Union[float, np.floating, np.ndarray, linops.LinearOperator],
        cov: Union[float, np.floating, np.ndarray, linops.LinearOperator],
        cov_cholesky: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
        random_state: RandomStateArgType = None,
    ):
        # Type normalization
        if np.isscalar(mean):
            mean = _utils.as_numpy_scalar(mean)

        if np.isscalar(cov):
            cov = _utils.as_numpy_scalar(cov)

        # Data type normalization
        is_mean_floating = mean.dtype is not None and np.issubdtype(
            mean.dtype, np.floating
        )
        is_cov_floating = cov.dtype is not None and np.issubdtype(
            cov.dtype, np.floating
        )

        if is_mean_floating and is_cov_floating:
            dtype = np.promote_types(mean.dtype, cov.dtype)
        elif is_mean_floating:
            dtype = mean.dtype
        elif is_cov_floating:
            dtype = cov.dtype
        else:
            dtype = np.dtype(np.float_)

        if not isinstance(mean, linops.LinearOperator):
            mean = mean.astype(dtype, order="C", casting="safe", subok=True, copy=False)
        else:
            # TODO: Implement casting for linear operators

            if mean.dtype != dtype:
                raise ValueError(
                    f"The mean must have type `{dtype.name}` not `{mean.dtype.name}`, "
                    f"but a linear operator does not implement type casting."
                )

        if not isinstance(cov, linops.LinearOperator):
            cov = cov.astype(dtype, order="C", casting="safe", subok=True, copy=False)
        else:
            # TODO: Implement casting for linear operators

            if cov.dtype != dtype:
                raise ValueError(
                    f"The covariance must have type `{dtype.name}` not "
                    f"`{cov.dtype.name}`, but a linear operator does not implement "
                    f"type casting."
                )

        # Shape checking
        if len(mean.shape) not in [0, 1, 2]:
            raise ValueError(
                f"Gaussian random variables must either be scalars, vectors, or "
                f"matrices (or linear operators), but the given mean is a {mean.ndim}-"
                f"dimensional tensor."
            )

        expected_cov_shape = (np.prod(mean.shape),) * 2 if len(mean.shape) > 0 else ()

        if len(cov.shape) != len(expected_cov_shape) or cov.shape != expected_cov_shape:
            raise ValueError(
                f"The covariance matrix must be of shape {expected_cov_shape}, but "
                f"shape {cov.shape} was given."
            )

        self._mean = mean
        self._cov = cov

        self._compute_cov_cholesky: Callable[[], _ValueType] = None

        # Method selection
        univariate = len(mean.shape) == 0
        dense = isinstance(mean, np.ndarray) and isinstance(cov, np.ndarray)
        cov_operator = isinstance(cov, linops.LinearOperator)

        if univariate:
            # Univariate Gaussian
            sample = self._univariate_sample
            in_support = Normal._univariate_in_support
            pdf = self._univariate_pdf
            logpdf = self._univariate_logpdf
            cdf = self._univariate_cdf
            logcdf = self._univariate_logcdf
            quantile = self._univariate_quantile

            median = lambda: self._mean
            var = lambda: self._cov
            entropy = self._univariate_entropy

            self._compute_cov_cholesky = self._univariate_cov_cholesky
        elif dense or cov_operator:
            # Multi- and matrixvariate Gaussians
            sample = self._dense_sample
            in_support = Normal._dense_in_support
            pdf = self._dense_pdf
            logpdf = self._dense_logpdf
            cdf = self._dense_cdf
            logcdf = self._dense_logcdf
            quantile = None

            median = None
            var = self._dense_var
            entropy = self._dense_entropy

            if cov_cholesky is None:
                self._compute_cov_cholesky = self.dense_cov_cholesky
            else:
                if not isinstance(cov_cholesky, type(self._cov)):
                    raise ValueError(
                        f"The covariance matrix is of type `{type(self._cov)}`, so its "
                        f"Cholesky decomposition must be of the same type, but an "
                        f"object of type `{type(cov_cholesky)}` was given."
                    )

                if cov_cholesky.shape != self._cov.shape:
                    raise ValueError(
                        f"The cholesky decomposition of the covariance matrix must "
                        f"have the same shape as the covariance matrix, i.e. "
                        f"{self._cov.shape}, but shape {cov_cholesky.shape} was given"
                    )

                if cov_cholesky.dtype != self._cov.dtype:
                    # TODO: Implement casting for linear operators
                    if not isinstance(cov_cholesky, linops.LinearOperator):
                        cov_cholesky = cov_cholesky.astype(self._cov.dtype)

                self._compute_cov_cholesky = lambda: cov_cholesky

            if isinstance(cov, linops.SymmetricKronecker):
                m, n = mean.shape

                if m != n or n != cov.A.shape[0] or n != cov.B.shape[1]:
                    raise ValueError(
                        "Normal distributions with symmetric Kronecker structured "
                        "kernels must have square mean and square kernels factors with "
                        "matching dimensions."
                    )

                if cov._ABequal:
                    sample = self._symmetric_kronecker_identical_factors_sample
                    pdf = self._symmetric_kronecker_identical_factors_pdf
                    logpdf = self._symmetric_kronecker_identical_factors_logpdf

                    # pylint: disable=redefined-variable-type
                    self._compute_cov_cholesky = (
                        self._symmetric_kronecker_identical_factors_cov_cholesky
                    )
            elif isinstance(cov, linops.Kronecker):
                m, n = mean.shape

                if (
                    m != cov.A.shape[0]
                    or m != cov.A.shape[1]
                    or n != cov.B.shape[0]
                    or n != cov.B.shape[1]
                ):
                    raise ValueError(
                        "Kronecker structured kernels must have factors with the same "
                        "shape as the mean."
                    )

                sample = self._kronecker_sample
                pdf = self._kronecker_pdf
                logpdf = self._kronecker_logpdf
                self._compute_cov_cholesky = self._kronecker_cov_cholesky
        else:
            raise ValueError(
                f"Cannot instantiate normal distribution with mean of type "
                f"{mean.__class__.__name__} and kernels of type "
                f"{cov.__class__.__name__}."
            )

        super().__init__(
            shape=mean.shape,
            dtype=mean.dtype,
            random_state=random_state,
            parameters={"mean": self._mean, "cov": self._cov},
            sample=sample,
            in_support=in_support,
            pdf=pdf,
            logpdf=logpdf,
            cdf=cdf,
            logcdf=logcdf,
            quantile=quantile,
            mode=lambda: self._mean,
            median=median,
            mean=lambda: self._mean,
            cov=lambda: self._cov,
            var=var,
            entropy=entropy,
        )

    @cached_property
    def cov_cholesky(self) -> _ValueType:
        if self._compute_cov_cholesky is None:
            raise NotImplementedError

        return self._compute_cov_cholesky()

    @cached_property
    def dense_mean(self) -> Union[np.floating, np.ndarray]:
        if isinstance(self._mean, linops.LinearOperator):
            return self._mean.todense()
        else:
            return self._mean

    @cached_property
    def dense_cov(self) -> Union[np.floating, np.ndarray]:
        if isinstance(self._cov, linops.LinearOperator):
            return self._cov.todense()
        else:
            return self._cov

    def __getitem__(self, key: ArrayLikeGetitemArgType) -> "Normal":
        """
        Marginalization in multi- and matrixvariate normal distributions, expressed by
        means of (advanced) indexing, masking and slicing.

        We support all modes of array indexing presented in

        https://numpy.org/doc/1.19/reference/arrays.indexing.html.

        Note that, currently, this method does not work for normal distributions other
        than the multi- and matrixvariate versions.

        Parameters
        ----------
        key : int or slice or ndarray or tuple of None, int, slice, or ndarray
            Indices, slice objects and/or boolean masks specifying which entries to keep
            while marginalizing over all other entries.
        """

        if not isinstance(key, tuple):
            key = (key,)

        # Select entries from mean
        mean = self.dense_mean[key]

        # Select submatrix from covariance matrix
        cov = self.dense_cov.reshape(self.shape + self.shape)
        cov = cov[key][tuple([slice(None)] * mean.ndim) + key]

        if mean.ndim > 0:
            cov = cov.reshape(mean.size, mean.size)

        return Normal(
            mean=mean,
            cov=cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def reshape(self, newshape: ShapeArgType) -> "Normal":
        try:
            reshaped_mean = self.dense_mean.reshape(newshape)
        except ValueError as exc:
            raise ValueError(
                f"Cannot reshape this normal random variable to the given shape: "
                f"{newshape}"
            ) from exc

        reshaped_cov = self.dense_cov

        if reshaped_mean.ndim > 0 and reshaped_cov.ndim == 0:
            reshaped_cov = reshaped_cov.reshape(1, 1)

        return Normal(
            mean=reshaped_mean,
            cov=reshaped_cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def transpose(self, *axes: int) -> "Normal":
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        elif (len(axes) == 1 and axes[0] is None) or len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))

        mean_t = self.dense_mean.transpose(*axes).copy()

        # Transpose covariance
        cov_axes = axes + tuple(mean_t.ndim + axis for axis in axes)
        cov_t = self.dense_cov.reshape(self.shape + self.shape)
        cov_t = cov_t.transpose(*cov_axes).copy()

        if mean_t.ndim > 0:
            cov_t = cov_t.reshape(mean_t.size, mean_t.size)

        return Normal(
            mean=mean_t,
            cov=cov_t,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    # Unary arithmetic operations

    def __neg__(self) -> "Normal":
        return Normal(
            mean=-self._mean,
            cov=self._cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def __pos__(self) -> "Normal":
        return Normal(
            mean=+self._mean,
            cov=self._cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    # TODO: Overwrite __abs__ and add absolute moments of normal
    # TODO: (https://arxiv.org/pdf/1209.4340.pdf)

    # Binary arithmetic operations

    def _add_normal(self, other: "Normal") -> "Normal":
        if other.shape != self.shape:
            raise ValueError(
                "Addition of two normally distributed random variables is only "
                "possible if both operands have the same shape."
            )

        return Normal(
            mean=self._mean + other._mean,
            cov=self._cov + other._cov,
            random_state=_utils.derive_random_seed(
                self.random_state, other.random_state
            ),
        )

    def _sub_normal(self, other: "Normal") -> "Normal":
        if other.shape != self.shape:
            raise ValueError(
                "Subtraction of two normally distributed random variables is only "
                "possible if both operands have the same shape."
            )

        return Normal(
            mean=self._mean - other._mean,
            cov=self._cov + other._cov,
            random_state=_utils.derive_random_seed(
                self.random_state, other.random_state
            ),
        )

    # Univariate Gaussians
    def _univariate_cov_cholesky(self) -> np.floating:
        return np.sqrt(self._cov)

    def _univariate_sample(
        self, size: ShapeType = ()
    ) -> Union[np.floating, np.ndarray]:
        sample = scipy.stats.norm.rvs(
            loc=self._mean, scale=self.std, size=size, random_state=self.random_state
        )

        if np.isscalar(sample):
            sample = _utils.as_numpy_scalar(sample, dtype=self.dtype)
        else:
            sample = sample.astype(self.dtype)

        assert sample.shape == size

        return sample

    @staticmethod
    def _univariate_in_support(x: _ValueType) -> bool:
        return np.isfinite(x)

    def _univariate_pdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.norm.pdf(x, loc=self._mean, scale=self.std)

    def _univariate_logpdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.norm.logpdf(x, loc=self._mean, scale=self.std)

    def _univariate_cdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.norm.cdf(x, loc=self._mean, scale=self.std)

    def _univariate_logcdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.norm.logcdf(x, loc=self._mean, scale=self.std)

    def _univariate_quantile(self, p: FloatArgType) -> np.floating:
        return scipy.stats.norm.ppf(p, loc=self._mean, scale=self.std)

    def _univariate_entropy(self: _ValueType) -> np.float_:
        return _utils.as_numpy_scalar(
            scipy.stats.norm.entropy(loc=self._mean, scale=self.std),
            dtype=np.float_,
        )

    # Multi- and matrixvariate Gaussians
    def dense_cov_cholesky(self) -> np.ndarray:
        dense_cov = self.dense_cov

        return scipy.linalg.cholesky(
            dense_cov + COV_CHOLESKY_DAMPING * np.eye(self.size, dtype=self.dtype),
            lower=True,
        )

    def _dense_sample(self, size: ShapeType = ()) -> np.ndarray:
        sample = scipy.stats.multivariate_normal.rvs(
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
            size=size,
            random_state=self.random_state,
        )

        return sample.reshape(sample.shape[:-1] + self.shape)

    @staticmethod
    def _arg_todense(x: Union[np.ndarray, linops.LinearOperator]) -> np.ndarray:
        if isinstance(x, linops.LinearOperator):
            return x.todense()
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise ValueError(f"Unsupported argument type {type(x)}")

    @staticmethod
    def _dense_in_support(x: _ValueType) -> bool:
        return np.all(np.isfinite(Normal._arg_todense(x)))

    def _dense_pdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.multivariate_normal.pdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _dense_logpdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.multivariate_normal.logpdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _dense_cdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.multivariate_normal.cdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _dense_logcdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.multivariate_normal.logcdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _dense_var(self) -> np.ndarray:
        return np.diag(self.dense_cov).reshape(self.shape)

    def _dense_entropy(self) -> np.float_:
        return _utils.as_numpy_scalar(
            scipy.stats.multivariate_normal.entropy(
                mean=self.dense_mean.ravel(),
                cov=self.dense_cov,
            ),
            dtype=np.float_,
        )

    # Matrixvariate Gaussian with Kronecker covariance
    def _kronecker_cov_cholesky(self) -> linops.Kronecker:
        assert isinstance(self._cov, linops.Kronecker)

        A = self._cov.A.todense()
        B = self._cov.B.todense()

        return linops.Kronecker(
            A=scipy.linalg.cholesky(
                A + COV_CHOLESKY_DAMPING * np.eye(A.shape[0], dtype=self.dtype),
                lower=True,
            ),
            B=scipy.linalg.cholesky(
                B + COV_CHOLESKY_DAMPING * np.eye(B.shape[0], dtype=self.dtype),
                lower=True,
            ),
            dtype=self.dtype,
        )

    def _kronecker_cov_decompose(self):
        """Returns depending on type of covariance and if decomposition are already cached a decomposition for covariance. chol=True if cholesky False if svd."""
        if self.__dict__.get('cov_cholesky')!= None: 
            return ((self.cov_cholesky.A, self.cov_cholesky.B), 'chol')
        #TODO: SVD decomposition should be implemented as cached property. Then after checking for cholesky factors in cache check for svd factors in cache
        if isinstance(self._cov.A, linops.LinearOperator) or isinstance(self._cov.B, linops.LinearOperator):
            #TODO: It would be nicer if svd algorithm wasn't randomized 
            return (
                (
                    scipy.linalg.interpolative.svd(A=self._cov.A, eps_or_k=10**-7),
                    scipy.linalg.interpolative.svd(A=self._cov.B, eps_or_k=10**-7),
                ),                
                'svd'
            )
        else:
            return ((self.cov_cholesky.A, self.cov_cholesky.B), 'chol')

    def _svd_cov_decompose(self, covA_svdfactors, covB_svdfactors):
        """ Returns factors to decompose cov into a product U*U.T using svd decomposition """
        covA_factor = covA_svdfactors[0] * np.sqrt(covA_svdfactors[1]) @ covA_svdfactors[0].T
        covB_factor = covB_svdfactors[0] * np.sqrt(covB_svdfactors[1]) @ covB_svdfactors[0].T
        return (covA_factor, covB_factor)

    def _kronecker_sample(self, size: ShapeType = ()) -> np.ndarray:
        (cov_factors, decomp) = self._kronecker_cov_decompose()
        if decomp == 'svd':
            cov_factors = self._svd_cov_decompose(cov_factors[0], cov_factors[1])
        elif decomp != 'chol':
            raise NotImplementedError("Decomposition must be 'chol' or 'svd'")

        # Draw standard normal samples
        if isinstance(size, (int, np.integer)):
            flat_shape = [size]+list(self._mean.shape)
            final_shape = flat_shape
        else:
            flat_shape = [np.prod(size)]+list(self._mean.shape)
            final_shape = list(size)+list(self._mean.shape)

        stdnormal_samples = self._random_state.normal(size=flat_shape)

        #TODO: Can be simplified once addition of linear operators and ndarrays is implemented
        if isinstance(self._mean, np.ndarray):
            return (self._mean + cov_factors[0] @ stdnormal_samples @ cov_factors[1].T).reshape(final_shape)
        else:
            samples_flat = [self._mean + linops.aslinop(cov_factors[0] @ sample @ cov_factors[1]) for sample in stdnormal_samples]
            return np.array(samples_flat).reshape(size)

    def _kronecker_pdf(self, x: _ValueType) -> np.float_:
        return np.exp( self._kronecker_logpdf(x) )

    def _kronecker_logpdf(self, x: _ValueType) -> np.float_:
        dev = linops.aslinop(x-self._mean).todense()
        (cov_factors, decomp) = self._kronecker_cov_decompose()
        if decomp == 'svd':
            (logdet_cov, maha) = self._logpdf_calc_svd(dev, cov_factors[0], cov_factors[1])
        elif decomp == 'chol':
            (logdet_cov, maha) = self._logpdf_calc_chol(dev, cov_factors)
        else:
            raise NotImplementedError("Decomposition must be 'chol' or 'svd'")

        # normalizing constant
        log2pi = np.log(2 * np.pi)
        normconst = np.prod(self._mean.shape)*log2pi + logdet_cov
        return -0.5*(normconst + maha)

    def _logpdf_calc_chol(self, dev, cov_cholfactors):
        logdet_cov = self._logabsdet_chol(cov_cholfactors)
        maha = self._chol_mahaldist(dev, cov_cholfactors)
        return (logdet_cov, maha)     
        
    def _logpdf_calc_svd(self, dev, covA_svdfactors, covB_svdfactors):
        logdet_cov = self._logabsdet_svd(covA_svdfactors, covB_svdfactors)
        maha = self._svd_mahaldist(dev, covA_svdfactors, covB_svdfactors)
        return (logdet_cov, maha)     

    def _logabsdet_chol(self, cov_cholfactors):
        """ Efficiently calculates determinant of Kronecker(V,W) using cholesky factors """
        # TODO: move to kronecker.py
        logdet_L_A = np.sum(np.log(np.diag(cov_cholfactors[0]))) #determinant of triangular matrices is equal to product of entries of diagonal and det of matrix is equal to product of det of its cholesky factor
        logdet_L_B = np.sum(np.log(np.diag(cov_cholfactors[1])))
        return (
            2*self._cov.B.shape[0]*logdet_L_A 
            + 2*self._cov.A.shape[0]*logdet_L_B
        )

    def _logabsdet_svd(self, covA_svdfactors, covB_svdfactors):
        """ Efficiently calculates determinant of Kronecker(V,W) using cholesky factors """
        # TODO: move to kronecker.py, test_kron_detforpdf refers directly to this method
        logdet_S_A = np.sum(np.log(covA_svdfactors[1])) #determinant of matrix is equal to product of singular values
        logdet_S_B = np.sum(np.log(covB_svdfactors[1]))
        return(
            self._cov.B.shape[0]*logdet_S_A 
            + self._cov.A.shape[0]*logdet_S_B
        )

    def _chol_mahaldist(self, dev, cov_cholfactors):
        """ Calculates mahalanobis distance vec(dev.T) @ Kronecker(A,B)^-1 @ vec(dev) using cholesky factors instead of inverting A and B """
        Q = scipy.linalg.cho_solve((cov_cholfactors[0], True), dev) #A*Q=dev => A^-1*dev=Q
        R_T = scipy.linalg.cho_solve((cov_cholfactors[1], True), Q.T, overwrite_b=True) #B*R.T=Q.T => Q*B.T^-1=R
        return dev.ravel() @ R_T.T.ravel()

    def _svd_mahaldist(self, dev,  covA_svdfactors, covB_svdfactors):
        """ Calculates mahalanobis distance vec(dev.T) @ Kronecker(A,B)^-1 @ vec(dev) using svd factors to calculate A^-1 and B^-1 """
        covA_inv = covA_svdfactors[0] * (1/covA_svdfactors[1]) @ covA_svdfactors[0].T
        covB_inv = covB_svdfactors[0] * (1/covB_svdfactors[1]) @ covB_svdfactors[0].T
        return dev.ravel() @ (covA_inv @ dev @ covB_inv.T).ravel()

    # Matrixvariate Gaussian with symmetric Kronecker covariance from identical
    # factors
    def _symmetric_kronecker_identical_factors_cov_cholesky(
        self,
    ) -> linops.SymmetricKronecker:
        assert isinstance(self._cov, linops.SymmetricKronecker) and self._cov._ABequal

        A = self._cov.A.todense()

        return linops.SymmetricKronecker(
            A=scipy.linalg.cholesky(
                A + COV_CHOLESKY_DAMPING * np.eye(A.shape[0], dtype=self.dtype),
                lower=True,
            ),
            dtype=self.dtype,
        )
    
    def _symmetric_kronecker_cov_decompose(self):
        """Returns depending on type of covariance and if decomposition are already cached a decomposition for covariance. chol=True if cholesky False if svd."""
        if self._cov._ABequal:
            if self.__dict__.get('cov_cholesky')!= None: 
                return (self.cov_cholesky.A, 'chol')
            #TODO: SVD decomposition should be implemented as cached property. Then after checking for cholesky factors in cache check for svd factors in cache
            if isinstance(self._cov.A, linops.LinearOperator):
                #TODO: It would be nicer if svd algorithm wasn't randomized 
                return (scipy.linalg.interpolative.svd(A=self._cov.A, eps_or_k=10**-7), 'svd')
            else:
                return (self.cov_cholesky.A, 'chol')
        else:
            raise NotImplementedError

    def _symmetric_svd_cov_decompose(self, covA_svdfactor):
        """ Returns factors to decompose cov into a product U*U.T using svd decomposition """
        covA_factor = covA_svdfactor[0] * np.sqrt(covA_svdfactor[1]) @ covA_svdfactor[0].T
        return covA_factor

    def _symmetric_kronecker_identical_factors_sample(
        self, size: ShapeType = ()
    ) -> np.ndarray:

        (cov_factor, decomp) = self._symmetric_kronecker_cov_decompose()
        if decomp == 'svd':
            cov_factor = self._symmetric_svd_cov_decompose(cov_factor)
        elif decomp != 'chol':
            raise NotImplementedError("Decomposition must be 'chol' or 'svd'")

        n = self._cov.A.shape[0]

        flat_shape = (np.prod(size),) + (n*n,)
        final_shape = size + (n,n)

        stdnormal_samples = self._random_state.normal(size=final_shape)

        transformed_samples = cov_factor @ stdnormal_samples @ cov_factor.T

        #TODO: Can be simplified once addition of linear operators and ndarrays is implemented
        if isinstance(self._mean, np.ndarray):
            return self._mean + 0.5*(
                transformed_samples
                + self._stacked_transpose(transformed_samples)
            )
        else:
            samples_flat = [
                self._mean + linops.aslinop(
                    0.5*(row.reshape(n,n) + row.reshape(n,n).T)
                ) for row in transformed_samples.reshape(flat_shape)
            ]
            return np.array(samples_flat).reshape(size)

    def _stacked_transpose(self, matrix_stack):
        """transposes matrices in an arbitrary shaped stack"""
        perm = np.arange(len(matrix_stack.shape))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return np.transpose(matrix_stack, axes=perm)
        
    def _symmetric_kronecker_identical_factors_pdf(self, x: _ValueType) -> np.float_:
        return np.exp(self._symmetric_kronecker_identical_factors_logpdf(x))

    def _symmetric_kronecker_identical_factors_logpdf(self, x: _ValueType) -> np.float_:
        dev = linops.aslinop(x-self._mean).todense()
        (cov_factor, decomp) = self._symmetric_kronecker_cov_decompose()
        if decomp == 'svd':
            (logdet_cov, maha) = self._symmlogpdf_calc_svd(dev, cov_factor)
        elif decomp == 'chol':
            (logdet_cov, maha) = self._symmlogpdf_calc_chol(dev, cov_factor)
        else:
            raise NotImplementedError("Decomposition must be 'chol' or 'svd'")

        # normalizing constant
        log2pi = np.log(2 * np.pi)
        normconst = np.prod(self._mean.shape)*log2pi + logdet_cov
        return -0.5*(normconst + maha)

    def _symm_logpdf_calc_chol(self, dev, cov_cholesky):
        logdet_cov = self._symm_logabsdet_chol(cov_cholesky)
        maha = self._symm_mahaldist_chol(dev, cov_cholesky)
        return (logdet_cov, maha)     
        
    def _symm_logpdf_calc_svd(self, dev,covA_svdfactor):
        logdet_cov = self._symm_logabsdet_svd(covA_svdfactor)
        maha = self._symm_mahaldist_svd(dev, covA_svdfactor)
        return (logdet_cov, maha)     

    def _symm_logabsdet_chol(self, cov_cholfactor):
        """ Efficiently calculates determinant of Kronecker(V,W) using cholesky factors """
        # TODO: move to kronecker.py, test_symm_kron_detforpdf refers directly to this method
        if self._cov._ABequal:
            logdet_L_A = np.sum(np.log(np.diag(cov_cholfactor))) #determinant of triangular matrices is equal to product of entries of diagonal and det of matrix is equal to product of det of its cholesky factor
            return 4*self._cov.A.shape[0]*logdet_L_A
        else:
            raise NotImplementedError

    def _symm_logabsdet_svd(self, covA_svdfactors):
        """ Efficiently calculates determinant of Kronecker(V,W) using cholesky factors """
        # TODO: move to kronecker.py
        if self._cov._ABequal:
            logdet_S_A = np.sum(np.log(covA_svdfactors[1])) #determinant of matrix is equal to product of singular values
            return 2*self._cov.A.shape[0]*logdet_S_A
        else:
            raise NotImplementedError

    def _symm_mahaldist_chol(self, dev, cov_cholfactor):
        """ Calculates mahalanobis distance vec(dev.T) @ Kronecker(A,B)^-1 @ vec(dev) using cholesky factors instead of inverting A and B """
        Q = scipy.linalg.cho_solve((cov_cholfactor, True), dev) #A*Q=dev => A^-1*dev=Q
        R_T = scipy.linalg.cho_solve((cov_cholfactor, True), Q.T, overwrite_b=True) #B*R.T=Q.T => Q*B.T^-1=R
        return dev.ravel() @ R_T.T.ravel()

    def _symm_mahaldist_svd(self, dev,  covA_svdfactors):
        """ Calculates mahalanobis distance vec(dev.T) @ Kronecker(A,B)^-1 @ vec(dev) using svd factors to calculate A^-1 and B^-1 """
        covA_inv = covA_svdfactors[0] * (1/covA_svdfactors[1]) @ covA_svdfactors[0].T
        return dev.ravel() @ (covA_inv @ dev @ covA_inv.T).ravel()

