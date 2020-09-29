"""Tests for the normal distribution."""
import itertools
import unittest

import numpy as np
import scipy.sparse
import scipy.stats

import probnum
from probnum import random_variables as rvs
from probnum.linalg import linops
from tests.testing import NumpyAssertions


def _random_spd_matrix(D=10):
    """ Generates a random symmetric positive definite matrix. """

    # Sample a rotation matrix Q in SO(D) (the special orthogonal group SO(D), or
    # orthogonal matrices with unit determinant, drawn uniformly from the Haar measure.
    #
    # The algorithm used is the subgroup algorithm as originally proposed by
    #
    # P. Diaconis & M. Shahshahani, "The subgroup algorithm for generating uniform
    # random variables". Probability in the Engineering and Informational Sciences 1:
    # 15?32 (1987)
    t = 2 * np.pi * np.random.rand()
    A = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])

    for d in range(2, D):
        v = np.random.randn(d + 1, 1)
        v /= np.sqrt(v.T.dot(v))
        e = np.zeros((d + 1, 1))
        e[0] += 1

        x = e - v
        x /= np.sqrt(x.T.dot(x))

        D_ = np.zeros((d + 1, d + 1))
        D_[0, 0] += 1
        D_[1:, 1:] += A
        A = D_ - 2 * x * (x.T.dot(D_))

    Q = -A

    # Generate random positive spectrum
    lam = np.random.gamma(shape=6, scale=1.0, size=D)

    return Q @ np.diag(lam) @ Q.T


class NormalTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for the normal distribution."""

    def setUp(self):
        """Resources for tests."""
        # Seed
        np.random.seed(seed=42)

        # Parameters
        m = 7
        n = 3
        self.constants = [-1, -2.4, 0, 200, np.pi]
        sparsemat = scipy.sparse.rand(m=m, n=n, density=0.1, random_state=1)
        self.normal_params = [
            # Univariate
            (-1.0, 3.0),
            (1, 3),
            # Multivariate
            (np.random.uniform(size=10), np.eye(10)),
            (np.random.uniform(size=10), _random_spd_matrix(10)),
            # Matrixvariate
            (
                np.random.uniform(size=(2, 2)),
                linops.SymmetricKronecker(
                    A=np.array([[2.0, 1.0], [1.0, 2.0]]),
                    B=np.array([[5.0, -1.0], [-1.0, 10.0]]),
                ).todense(),
            ),
            # Operatorvariate
            (
                np.array([1.0, -5.0]),
                linops.MatrixMult(A=np.array([[1.0, 2.0], [-0.1, 1.0]])),
            ),
            (
                linops.MatrixMult(A=np.array([[0.0, -5.0]])),
                linops.Identity(shape=(2, 2)),
            ),
            (
                np.array([[1.0, 2.0], [-3.0, -0.4], [4.0, 1.0]]),
                linops.Kronecker(A=np.eye(3), B=5 * np.eye(2)),
            ),
            (
                linops.MatrixMult(A=sparsemat.todense()),
                linops.Kronecker(linops.Identity(m), linops.Identity(n)),
            ),
            (
                linops.MatrixMult(A=np.random.uniform(size=(2, 2))),
                linops.SymmetricKronecker(
                    A=np.array([[2.0, 1.0], [1.0, 2.0]]),
                    B=np.array([[5.0, -1.0], [-1.0, 10.0]]),
                ),
            ),
            # Symmetric Kronecker Identical Factors
            (
                linops.Identity(shape=25),
                linops.SymmetricKronecker(A=linops.Identity(25)),
            ),
        ]

    def test_correct_instantiation(self):
        """Test whether different variants of the normal distribution are instances of Normal."""
        for mean, cov in self.normal_params:
            with self.subTest():
                dist = rvs.Normal(mean=mean, cov=cov)
                self.assertIsInstance(dist, rvs.Normal)

    def test_scalarmult(self):
        """Multiply a rv with a normal distribution with a scalar."""
        for (mean, cov), const in list(
            itertools.product(self.normal_params, self.constants)
        ):
            with self.subTest():
                normrv = const * rvs.Normal(mean=mean, cov=cov)

                self.assertIsInstance(normrv, probnum.RandomVariable)

                if const != 0:
                    self.assertIsInstance(normrv, rvs.Normal)
                else:
                    self.assertIsInstance(normrv, rvs.Dirac)

    def test_addition_normal(self):
        """Add two random variables with a normal distribution"""
        for (mean0, cov0), (mean1, cov1) in list(
            itertools.product(self.normal_params, self.normal_params)
        ):
            with self.subTest():
                normrv0 = rvs.Normal(mean=mean0, cov=cov0)
                normrv1 = rvs.Normal(mean=mean1, cov=cov1)

                if normrv0.shape == normrv1.shape:
                    try:
                        newmean = mean0 + mean1
                        newcov = cov0 + cov1
                    except TypeError:
                        continue

                    self.assertIsInstance(normrv0 + normrv1, rvs.Normal)
                else:
                    with self.assertRaises(ValueError):
                        normrv_added = normrv0 + normrv1

    def test_rv_linop_kroneckercov(self):
        """Create a rv with a normal distribution with linear operator mean and Kronecker product kernels."""

        def mv(v):
            return np.array([2 * v[0], 3 * v[1]])

        A = linops.LinearOperator(shape=(2, 2), matvec=mv)
        V = linops.Kronecker(A, A)
        rvs.Normal(mean=A, cov=V)

    def test_normal_dimension_mismatch(self):
        """Instantiating a normal distribution with mismatched mean and kernels should result in a ValueError."""
        for mean, cov in [
            (0, np.array([1, 2])),
            (np.array([1, 2]), np.array([1, 0])),
            (np.array([[-1, 0], [2, 1]]), np.eye(3)),
        ]:
            with self.subTest():
                err_msg = "Mean and kernels mismatch in normal distribution did not raise a ValueError."
                with self.assertRaises(ValueError, msg=err_msg):
                    assert rvs.Normal(mean=mean, cov=cov)

    def test_normal_instantiation(self):
        """Instantiation of a normal distribution with mixed mean and cov type."""
        for mean, cov in self.normal_params:
            with self.subTest():
                rvs.Normal(mean=mean, cov=cov)

    def test_normal_pdf(self):
        """Evaluate pdf at random input."""
        for mean, cov in self.normal_params:
            print("mean = ", mean, type(mean), "Cov = ", cov, type(cov))
            with self.subTest():
                rv = rvs.Normal(mean=mean, cov=cov)
                density = rv.pdf(mean)

    def test_normal_cdf(self):
        """Evaluate cdf at random input."""
        pass

    def test_sample_zero_cov(self):
        """Draw sample from distribution with zero kernels and check whether it equals the mean."""
        for mean, cov in self.normal_params:
            with self.subTest():
                rv = rvs.Normal(mean=mean, cov=0 * cov, random_state=1)
                rv_sample = rv.sample(size=1)
                assert_str = "Draw with kernels zero does not match mean."
                if isinstance(rv.mean, linops.LinearOperator):
                    self.assertAllClose(rv_sample, rv.mean.todense(), msg=assert_str)
                else:
                    self.assertAllClose(rv_sample, rv.mean, msg=assert_str)

    def test_kron_pdf_correct(self):
        """Compare result of pdf with linearoperators as input to multivariate numpy version"""
        for (dimA, dimB) in [(1, 1), (2, 3), (11, 6)]:
            with self.subTest():
                print("DimA = ", dimA, ", DimB = ", dimB)
                np.random.seed(496)
                # generate mean and covariance
                A = np.random.uniform(size=(dimA, dimA))
                A = A @ A.T + dimA * np.eye(dimA)
                B = np.random.uniform(size=(dimB, dimB))
                B = B @ B.T + dimB * np.eye(dimB)
                mean = np.random.uniform(size=(dimA, dimB))
                mean_vec = mean.ravel()
                cov = linops.Kronecker(A, B)
                cov_dense = np.kron(A, B)

                # create distributions
                dist_op = rvs.Normal(mean=mean, cov=cov)
                dist_mult = scipy.stats.multivariate_normal(
                    mean=mean_vec, cov=cov_dense
                )

                # compute densities
                density_kron = dist_op.pdf(mean)
                density_mult = dist_mult.pdf(mean_vec)
                self.assertAlmostEqual(density_kron, density_mult)

    def test_symm_kron_pdf_correct(self):
        """Compare result of pdf with linearoperators as input to multivariate numpy version"""
        for dimA in [1, 3, 6]:
            print("Dim = ", dimA)
            np.random.seed(496)
            # generate mean and covariance
            A = np.random.uniform(size=(dimA, dimA))
            A = A @ A.T + dimA * np.eye(dimA)
            mean = np.random.uniform(size=(dimA, dimA))
            mean_vec = mean.ravel()
            cov = linops.SymmetricKronecker(A=A)
            cov_dense = cov.todense()

            # create distributions
            rv_symmkron = rvs.Normal(mean=mean, cov=cov)
            rv_mult = scipy.stats.multivariate_normal(mean=mean_vec, cov=cov_dense)

            # compute densities
            density_symmkron = rv_symmkron.pdf(mean)
            density_mult = rv_mult.pdf(mean_vec)
            self.assertAlmostEqual(density_symmkron, density_mult)

    def test_kron_detforpdf(self):
        for (dimA, dimB) in [(1, 1), (2, 3), (11, 6)]:
            print("DimA = ", dimA, ", DimB = ", dimB)
            np.random.seed(496)
            # generate mean and covariance
            A = np.random.uniform(size=(dimA, dimA))
            A = A @ A.T + dimA * np.eye(dimA)
            B = np.random.uniform(size=(dimB, dimB))
            B = B @ B.T + dimB * np.eye(dimB)
            mean = np.random.uniform(size=(dimA, dimB))
            mean_vec = mean.ravel()
            cov = linops.Kronecker(A, B)
            cov_dense = np.kron(A, B)
            rv = rvs.Normal(mean=mean, cov=cov)
            # compute decompositions
            cov_chol_factors = (np.linalg.cholesky(A), np.linalg.cholesky(B))
            covA_svd_factors = np.linalg.svd(A)
            covB_svd_factors = np.linalg.svd(B)
            cov_dense = cov.todense()

            self.assertAlmostEqual(
                np.log(np.linalg.det(cov_dense)), rv._logabsdet_chol(cov_chol_factors)
            )
            self.assertAlmostEqual(
                np.log(np.linalg.det(cov_dense)),
                rv._logabsdet_svd(covA_svd_factors, covB_svd_factors),
            )

    def test_symm_kron_detforpdf(self):
        for dimA in [1, 3, 6]:
            with self.subTest():
                np.random.seed(496)
                # generate mean and covariance
                A = np.random.uniform(size=(dimA, dimA))
                A = A @ A.T + dimA * np.eye(dimA)
                mean = np.random.rand(dimA, dimA)
                mean_vec = mean.ravel()
                cov = linops.SymmetricKronecker(A=A)
                rv = rvs.Normal(mean=mean, cov=cov)
                # compute decompositions
                cov_chol_factor = np.linalg.cholesky(A)
                cov_svd_factors = np.linalg.svd(A)

                cov_dense = cov.todense()
                self.assertAlmostEqual(
                    np.log(np.linalg.det(cov_dense)),
                    rv._symm_logabsdet_chol(cov_chol_factor),
                )
                self.assertAlmostEqual(
                    np.log(np.linalg.det(cov_dense)),
                    rv._symm_logabsdet_svd(cov_svd_factors),
                )

    def test_normal_cdf(self):
        """Evaluate cdf at random input."""
        pass

    def test_sample_outputdim(self):
        """Draw samples and check all sample dimensions."""
        for mean, cov in self.normal_params:
            print("mean = ", mean, type(mean), "Cov = ", cov, type(cov))
            for size in [[3], (2, 2), (3, 2, 3)]:
                print("~~~> size = ", size)
                with self.subTest():
                    # TODO: check dimension of each realization in rv_sample
                    rv = rvs.Normal(mean=mean, cov=cov, random_state=1)
                    rv_sample = rv.sample(size=size)
                    print(rv.mean)
                    if np.isscalar(rv.mean):
                        print("scalar")
                        self.assertEqual(
                            list(rv_sample.shape),
                            list(size),
                            msg="Realization shape does not match size x mean shape.",
                        )
                    elif isinstance(rv_sample.ravel()[0], linops.LinearOperator):
                        print("linop")
                        self.assertEqual(
                            list(rv_sample.shape),
                            list(size),
                            msg="Realization shape does not match size.",
                        )
                        self.assertEqual(
                            list(rv_sample.ravel()[0].shape),
                            list(rv.mean.shape),
                            msg="Realization shape of elements does not match mean shape.",
                        )
                    else:
                        print("other: ", type(rv.mean))
                        self.assertEqual(
                            list(rv_sample.shape),
                            list(size) + list(rv.mean.shape),
                            msg="Realization shape does not match size x mean shape.",
                        )

    def test_kron_sample_outputdim(self):
        np.random.seed(496)
        for (DIM_V, DIM_W) in [
            (1, 1),
            (1, 3),
            (5, 4),
            (
                10,
                10,
            ),
        ]:
            with self.subTest():
                V = np.random.rand(DIM_V, DIM_V)
                V = np.dot(V, V.T)
                W = np.random.rand(DIM_W, DIM_W)
                W = np.dot(W, W.transpose())
                cov = linops.Kronecker(V, W)
                rv = rvs.Normal(
                    mean=np.eye(DIM_V, DIM_W),
                    cov=cov,
                    cov_cholesky=linops.Kronecker(V, W),
                )
                rv_sample = rv.sample(size=(15, 20))
                self.assertEqual(rv_sample.shape, (15, 20, DIM_V, DIM_W))

    def test_symm_kron_sample_outputdim(self):
        np.random.seed(496)
        for n in [1, 3, 10]:
            with self.subTest():
                A = np.random.uniform(size=(n, n))
                A = 0.5 * (A + A.T) + n * np.eye(n)
                rv = rvs.Normal(
                    mean=np.eye(A.shape[0]),
                    cov=linops.SymmetricKronecker(A=A),
                    random_state=1,
                )
                rv_sample = rv.sample(size=(15, 20))
                self.assertEqual(rv_sample.shape, (15, 20, n, n))

    def test_symm_kron_sample_symmetric(self):
        """Samples from a normal distribution with symmetric Kronecker kernels of two symmetric matrices are
        symmetric."""
        np.random.seed(42)
        n = 3
        A = np.random.uniform(size=(n, n))
        A = 0.5 * (A + A.T) + n * np.eye(n)
        rv = rvs.Normal(
            mean=np.eye(A.shape[0]), cov=linops.SymmetricKronecker(A=A), random_state=1
        )
        rv = rv.sample(size=10)
        for i, B in enumerate(rv):
            self.assertAllClose(
                B,
                B.T,
                atol=1e-5,
                rtol=1e-5,
                msg="Sample {} from symmetric Kronecker distribution is not symmetric.".format(
                    i
                ),
            )

    def _mean_tol(self, mean, cov, size):
        """Returns atol for testing real vs empiric mean with 99,7% limitting accuracy"""
        n = np.prod(size)
        max_var = np.max(np.diag(cov))
        atol = 3 * np.sqrt(max_var / n)
        return atol

    def _cov_tol(self, cov, size):
        """Returns atol for testing real vs empiric cov with 99,7% limitting accuracy"""
        n = np.prod(size)
        empiric_cov_var = np.outer(np.diag(cov), np.diag(cov)) + cov ** 2
        max_empiric_cov_var = np.max(empiric_cov_var)
        atol = 3 * np.sqrt(max_empiric_cov_var / n)
        return atol

    def test_kron_sample_mean(self):
        """Verifies if empiric covariance based on samples of symmetric Kronecker distribution coincides with mathematical covariance."""
        np.random.seed(496)
        SIZE = int(1e04)  # if SIZE is tuple you have to change axis at np.mean call
        # create distribution
        (DIM_A, DIM_B) = (3, 5)
        A = np.random.rand(DIM_A, DIM_A)
        A = np.dot(A, A.T)
        B = np.random.rand(DIM_B, DIM_B)
        B = np.dot(B, B.transpose())
        mean = np.outer(np.linspace(0, 1, DIM_A), np.linspace(0.1, 2, DIM_B))
        cov = linops.Kronecker(A=A, B=B)
        rv = rvs.Normal(mean=mean, cov=cov, random_state=1)
        # take random samples
        samples = rv.sample(size=SIZE)
        # compute empiric mean
        sampled_mean = np.mean(samples, axis=(0))
        atol = self._mean_tol(mean=mean, cov=cov.todense(), size=SIZE)
        self.assertAllClose(sampled_mean, mean, atol=atol, rtol=0)

    def test_symm_kron_sample_mean(self):
        """Verifies if empiric covariance based on samples of symmetric Kronecker distribution coincides with mathematical covariance."""
        np.random.seed(496)
        SIZE = int(1e04)  # if SIZE is tuple you have to change axis at np.mean call
        # create distribution
        n = 4
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T) + n * np.eye(n)
        mean = np.outer(np.linspace(0.1, 1, n), np.linspace(0.1, 2, n))
        cov = linops.SymmetricKronecker(A=A)
        rv = rvs.Normal(mean=mean, cov=cov, random_state=1)
        # take random samples
        samples = rv.sample(size=SIZE)
        # compute empiric mean
        sampled_mean = np.mean(samples, axis=(0))

        atol = self._mean_tol(mean=mean, cov=cov.todense(), size=SIZE)
        self.assertAllClose(sampled_mean, mean, atol=atol, rtol=0)

    def test_kron_sample_cov(self):
        """Verifies if empiric covariance based on samples of symmetric Kronecker distribution coincides with mathematical covariance."""
        np.random.seed(496)
        SIZE = int(1e04)  # only use onedimensional size for this test
        # create distribution
        (DIM_A, DIM_B) = (3, 5)
        A = np.random.rand(DIM_A, DIM_A)
        A = np.dot(A, A.T)
        B = np.random.rand(DIM_B, DIM_B)
        B = np.dot(B, B.transpose())
        mean = np.random.rand(DIM_A, DIM_B)
        cov = linops.Kronecker(A=A, B=B)
        rv = rvs.Normal(mean=mean, cov=cov, random_state=1)
        # take random samples
        samples = rv.sample(size=SIZE)
        # compute empiric and dense covariance
        samples_multivar = np.array([mat.ravel() for mat in samples]).T
        sampled_cov = np.cov(samples_multivar)
        dense_cov = linops.Kronecker(A, B).todense()

        atol = self._cov_tol(cov=dense_cov, size=SIZE)
        self.assertAllClose(sampled_cov, dense_cov, atol=atol, rtol=0)

    def test_symm_kron_sample_cov(self):
        """Verifies if empiric covariance based on samples of symmetric Kronecker distribution coincides with mathematical covariance."""
        np.random.seed(496)
        SIZE = int(1e04)  # only use onedimensional size for this test
        # create distribution
        n = 4
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T) + n * np.eye(n)
        mean = np.eye(n)
        cov = linops.SymmetricKronecker(A=A)
        rv = rvs.Normal(mean=mean, cov=cov, random_state=1)
        # take random samples
        samples = rv.sample(size=SIZE)
        # compute empiric and dense covariance
        samples_multivar = np.array([mat.ravel() for mat in samples]).T
        sampled_cov = np.cov(samples_multivar)
        dense_cov = linops.SymmetricKronecker(A).todense()

        atol = self._cov_tol(cov=dense_cov, size=SIZE)
        self.assertAllClose(sampled_cov, dense_cov, atol=atol, rtol=0)

    def test_indexing(self):
        """ Indexing with Python integers yields a univariate normal distribution. """
        for mean, cov in self.normal_params:
            rv = rvs.Normal(mean=mean, cov=cov)

            with self.subTest():
                # Sample random index
                idx = tuple(np.random.randint(dim_size) for dim_size in rv.shape)

                # Index into distribution
                indexed_rv = rv[idx]

                self.assertIsInstance(indexed_rv, rvs.Normal)

                # Compare with expected parameter values
                if rv.ndim == 0:
                    flat_idx = ()
                elif rv.ndim == 1:
                    flat_idx = (idx[0],)
                else:
                    assert rv.ndim == 2

                    flat_idx = (idx[0] * rv.shape[1] + idx[1],)

                self.assertEqual(indexed_rv.shape, ())
                self.assertEqual(indexed_rv.mean, rv.dense_mean[idx])
                self.assertEqual(indexed_rv.var, rv.var[idx])
                self.assertEqual(indexed_rv.cov, rv.dense_cov[flat_idx + flat_idx])

    def test_slicing(self):
        """ Slicing into a normal distribution yields a normal distribution of the same type """
        for mean, cov in self.normal_params:
            rv = rvs.Normal(mean=mean, cov=cov)

            def _random_slice(dim_size):
                start = np.random.randint(0, dim_size)
                stop = np.random.randint(start + 1, dim_size + 1)

                return slice(start, stop)

            with self.subTest():
                # Sample random slice objects for each dimension
                slices = tuple(_random_slice(dim_size) for dim_size in rv.shape)

                # Get slice from distribution
                sliced_rv = rv[slices]

                # Compare with expected parameter values
                slice_mask = np.zeros_like(rv.dense_mean, dtype=np.bool)
                slice_mask[slices] = True
                slice_mask = slice_mask.ravel()

                self.assertArrayEqual(sliced_rv.mean, rv.dense_mean[slices])
                self.assertArrayEqual(sliced_rv.var, rv.var[slices])

                if rv.ndim > 0:
                    self.assertArrayEqual(
                        sliced_rv.cov, rv.dense_cov[np.ix_(slice_mask, slice_mask)]
                    )
                else:
                    self.assertArrayEqual(sliced_rv.cov, rv.cov)

    def test_array_indexing(self):
        """ Indexing with 1-dim integer arrays yields a multivariate normal. """
        for mean, cov in self.normal_params:
            rv = rvs.Normal(mean=mean, cov=cov)

            if rv.ndim == 0:
                continue

            with self.subTest():
                # Sample random indices
                idcs = tuple(
                    np.random.randint(dim_shape, size=10) for dim_shape in mean.shape
                )

                # Index into distribution
                indexed_rv = rv[idcs]

                self.assertIsInstance(indexed_rv, rvs.Normal)

                # Compare with expected parameter values
                if len(rv.shape) == 1:
                    flat_idcs = idcs[0]
                else:
                    assert len(rv.shape) == 2

                    flat_idcs = idcs[0] * rv.shape[1] + idcs[1]

                self.assertEqual(indexed_rv.shape, (10,))

                self.assertArrayEqual(indexed_rv.mean, rv.dense_mean[idcs])
                self.assertArrayEqual(indexed_rv.var, rv.var[idcs])
                self.assertArrayEqual(
                    indexed_rv.cov, rv.dense_cov[np.ix_(flat_idcs, flat_idcs)]
                )

    def test_array_indexing_broadcast(self):
        """ Indexing with broadcasted integer arrays yields a matrixvariate normal. """
        for mean, cov in self.normal_params:
            rv = rvs.Normal(mean=mean, cov=cov)

            if rv.ndim != 2:
                continue

            with self.subTest():
                # Sample random indices
                idcs = np.ix_(
                    *tuple(
                        np.random.randint(dim_shape, size=10) for dim_shape in rv.shape
                    )
                )

                # Index into distribution
                indexed_rv = rv[idcs]

                self.assertIsInstance(indexed_rv, rvs.Normal)
                self.assertEqual(indexed_rv.shape, (10, 10))

                # Compare with expected parameter values
                flat_idcs = np.broadcast_arrays(*idcs)
                flat_idcs = flat_idcs[0] * rv.shape[1] + flat_idcs[1]
                flat_idcs = flat_idcs.ravel()

                self.assertArrayEqual(indexed_rv.mean, rv.dense_mean[idcs])
                self.assertArrayEqual(indexed_rv.var, rv.var[idcs])
                self.assertArrayEqual(
                    indexed_rv.cov, rv.dense_cov[np.ix_(flat_idcs, flat_idcs)]
                )

    def test_masking(self):
        """ Masking a multivariate or matrixvariate normal yields a multivariate normal. """
        for mean, cov in self.normal_params:
            rv = rvs.Normal(mean=mean, cov=cov)

            with self.subTest():
                # Sample random indices
                idcs = tuple(
                    np.random.randint(dim_shape, size=10) for dim_shape in rv.shape
                )

                mask = np.zeros_like(rv.dense_mean, dtype=np.bool)
                mask[idcs] = True

                # Mask distribution
                masked_rv = rv[mask]

                self.assertIsInstance(masked_rv, rvs.Normal)

                # Compare with expected parameter values
                flat_mask = mask.flatten()

                self.assertArrayEqual(masked_rv.mean, rv.dense_mean[mask])
                self.assertArrayEqual(masked_rv.var, rv.var[mask])

                if rv.ndim == 0:
                    self.assertArrayEqual(masked_rv.cov, rv.cov)
                else:
                    self.assertArrayEqual(
                        masked_rv.cov, rv.dense_cov[np.ix_(flat_mask, flat_mask)]
                    )


class UnivariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.params = (np.random.uniform(), np.random.gamma(shape=6, scale=1.0))

    def test_reshape_newaxis(self):
        dist = rvs.Normal(*self.params)

        for ndim in range(1, 3):
            for method in ["newaxis", "reshape"]:
                with self.subTest():
                    newshape = tuple([1] * ndim)

                    if method == "newaxis":
                        if ndim == 1:
                            newdist = dist[np.newaxis]
                        elif ndim == 2:
                            newdist = dist[np.newaxis, np.newaxis]
                    elif method == "reshape":
                        newdist = dist.reshape(newshape)

                    self.assertEqual(newdist.shape, newshape)
                    self.assertEqual(np.squeeze(newdist.mean), dist.mean)
                    self.assertEqual(np.squeeze(newdist.cov), dist.cov)

    def test_transpose(self):
        dist = rvs.Normal(*self.params)
        dist_t = dist.transpose()

        self.assertArrayEqual(dist_t.mean, dist.mean)
        self.assertArrayEqual(dist_t.cov, dist.cov)

        # Test sampling
        dist.random_state = 42
        dist_sample = dist.sample(size=5)

        dist_t.random_state = 42
        dist_t_sample = dist_t.sample(size=5)

        self.assertArrayEqual(dist_t_sample, dist_sample)


class MultivariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.params = (np.random.uniform(size=10), _random_spd_matrix(10))

    def test_newaxis(self):
        vector_rv = rvs.Normal(*self.params)

        matrix_rv = vector_rv[:, np.newaxis]

        self.assertEqual(matrix_rv.shape, (10, 1))
        self.assertArrayEqual(np.squeeze(matrix_rv.mean), vector_rv.mean)
        self.assertArrayEqual(matrix_rv.cov, vector_rv.cov)

    def test_reshape(self):
        rv = rvs.Normal(*self.params)

        newshape = (5, 2)
        reshaped_rv = rv.reshape(newshape)

        self.assertArrayEqual(reshaped_rv.mean, rv.mean.reshape(newshape))
        self.assertArrayEqual(reshaped_rv.cov, rv.cov)

        # Test sampling
        rv.random_state = 42
        dist_sample = rv.sample(size=5)

        reshaped_rv.random_state = 42
        dist_reshape_sample = reshaped_rv.sample(size=5)

        self.assertArrayEqual(
            dist_reshape_sample, dist_sample.reshape((-1,) + newshape)
        )

    def test_transpose(self):
        rv = rvs.Normal(*self.params)
        transposed_rv = rv.transpose()

        self.assertArrayEqual(transposed_rv.mean, rv.mean)
        self.assertArrayEqual(transposed_rv.cov, rv.cov)

        # Test sampling
        rv.random_state = 42
        dist_sample = rv.sample(size=5)

        transposed_rv.random_state = 42
        dist_t_sample = transposed_rv.sample(size=5)

        self.assertArrayEqual(dist_t_sample, dist_sample)


class MatrixvariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def test_reshape(self):
        rv = rvs.Normal(
            mean=np.random.uniform(size=(4, 3)),
            cov=linops.Kronecker(
                A=_random_spd_matrix(4), B=_random_spd_matrix(3)
            ).todense(),
        )

        newshape = (2, 6)
        reshaped_rv = rv.reshape(newshape)

        self.assertArrayEqual(reshaped_rv.mean, rv.mean.reshape(newshape))
        self.assertArrayEqual(reshaped_rv.cov, rv.cov)

        # Test sampling
        rv.random_state = 42
        dist_sample = rv.sample(size=5)

        reshaped_rv.random_state = 42
        dist_reshape_sample = reshaped_rv.sample(size=5)

        self.assertArrayEqual(
            dist_reshape_sample, dist_sample.reshape((-1,) + newshape)
        )

    def test_transpose(self):
        rv = rvs.Normal(mean=np.random.uniform(size=(2, 2)), cov=_random_spd_matrix(4))
        transposed_rv = rv.transpose()

        self.assertArrayEqual(transposed_rv.mean, rv.mean.T)

        # Test covariance
        for ii, ij in itertools.product(range(2), range(2)):
            for ji, jj in itertools.product(range(2), range(2)):
                idx = (2 * ii + ij, 2 * ji + jj)
                idx_t = (2 * ij + ii, 2 * jj + ji)

                self.assertEqual(transposed_rv.cov[idx_t], rv.cov[idx])

        # Sadly, sampling is not stable w.r.t. permutations of variables


if __name__ == "__main__":
    unittest.main()
