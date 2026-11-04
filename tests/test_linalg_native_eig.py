# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Test the native (no vendor LAPACK) nonsymmetric real eigensolver
(balancing, Hessenberg reduction, Francis double-shift QR, back
substitution), sc.EigenDecompositionFloat64, sc.eig, and sc.eigvals.
"""

import time
import unittest

import numpy as np

import solvcon as sc


EPS = np.finfo(np.float64).eps

# The residual |A v - lambda v| / (|A| |v|) is the backward error of an
# eigenpair; a backward-stable QR algorithm keeps it at a small multiple of
# eps independent of the conditioning, so the bound is fixed.
RESIDUAL_TOL = 1e-12


def eigenvalue_tolerance(a):
    """
    Per-eigenvalue bound on |lambda_i - lambda_i(numpy)| for the eigenvalues
    sorted as numpy returns them. To first order an eigenvalue moves by
    kappa_i |E| for a backward error E (Wilkinson), where
    kappa_i = 1 / |y_i^H x_i| with unit right and left eigenvectors x_i and
    y_i. Both solvers have |E| below 10 * n * eps * |A|_F, so they differ by
    at most twice that times kappa_i.
    """
    n = a.shape[0]
    w, x = np.linalg.eig(a)
    y = np.linalg.inv(x).conj().T
    x = x / np.linalg.norm(x, axis=0)
    y = y / np.linalg.norm(y, axis=0)
    kappa = 1.0 / np.abs(np.sum(y.conj() * x, axis=0))
    return w, 2.0 * 10.0 * n * EPS * np.linalg.norm(a) * kappa


def match_to(ref, values):
    """Reorder values to pair each with the nearest unmatched ref entry."""
    pool = list(values)
    out = []
    for target in ref:
        dist = [abs(target - value) for value in pool]
        out.append(pool.pop(int(np.argmin(dist))))
    return np.array(out)


def to_sa(a):
    return sc.SimpleArrayFloat64(array=np.ascontiguousarray(a, dtype=float))


def companion(roots):
    """Companion matrix of the monic polynomial with the given roots."""
    coeffs = np.real(np.poly(roots))
    n = len(roots)
    mat = np.zeros((n, n))
    mat[0, :] = -coeffs[1:]
    mat[1:, :-1] = np.eye(n - 1)
    return mat


class EigenDecompositionTC(unittest.TestCase):

    def check_pairs(self, a, eig):
        n = a.shape[0]
        w = np.array(eig.eigenvalues)
        v = np.array(eig.eigenvectors)
        self.assertEqual(w.dtype, np.complex128)
        self.assertEqual(v.shape, (n, n))
        anorm = max(np.linalg.norm(a), np.finfo(float).tiny)
        for it in range(n):
            vec = v[:, it]
            self.assertLessEqual(abs(np.linalg.norm(vec) - 1.0), 1e-14)
            res = np.linalg.norm(a @ vec - w[it] * vec) / anorm
            self.assertLessEqual(res, RESIDUAL_TOL)
            # Largest component real, the convention of LAPACK geev. Two
            # components may tie in magnitude up to rounding, so any of the
            # largest within 1e-12 relative may be the real one.
            mags = np.abs(vec)
            largest = vec[mags >= mags.max() * (1.0 - 1e-12)]
            self.assertIn(0.0, largest.imag)
        return w, v

    def check_vs_numpy(self, a, w):
        w_ref, tol = eigenvalue_tolerance(a)
        matched = match_to(w_ref, w)
        np.testing.assert_array_less(np.abs(matched - w_ref), tol + 1e-300)

    def test_diagonal(self):
        d = np.array([3.0, -1.0, 0.5, 7.0, 0.0])
        w, _ = self.check_pairs(np.diag(d), sc.EigenDecompositionFloat64(
            to_sa(np.diag(d))))
        np.testing.assert_array_equal(np.sort(w.real), np.sort(d))
        np.testing.assert_array_equal(w.imag, 0.0)

    def test_upper_triangular(self):
        rng = np.random.default_rng(1)
        a = np.triu(rng.standard_normal((6, 6)))
        np.fill_diagonal(a, [5.0, -2.0, 1.0, 3.0, -4.0, 0.25])
        w, _ = self.check_pairs(a, sc.EigenDecompositionFloat64(to_sa(a)))
        # A triangular matrix is already in Schur form: no QR step runs, so
        # the eigenvalues are the diagonal entries exactly.
        np.testing.assert_array_equal(np.sort(w.real), np.sort(np.diag(a)))

    def test_lower_triangular(self):
        rng = np.random.default_rng(2)
        a = np.tril(rng.standard_normal((6, 6)))
        np.fill_diagonal(a, [5.0, -2.0, 1.0, 3.0, -4.0, 0.25])
        w, _ = self.check_pairs(a, sc.EigenDecompositionFloat64(to_sa(a)))
        self.check_vs_numpy(a, w)

    def test_companion_real_roots(self):
        roots = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = companion(roots)
        w, _ = self.check_pairs(a, sc.EigenDecompositionFloat64(to_sa(a)))
        _, tol = eigenvalue_tolerance(a)
        matched = match_to(roots, w)
        np.testing.assert_array_less(np.abs(matched - roots), tol)

    def test_companion_complex_roots(self):
        roots = np.array([1.0 + 2.0j, 1.0 - 2.0j, 3.0, -0.5 + 0.5j,
                          -0.5 - 0.5j, -2.0])
        a = companion(roots)
        w, _ = self.check_pairs(a, sc.EigenDecompositionFloat64(to_sa(a)))
        _, tol = eigenvalue_tolerance(a)
        matched = match_to(roots, w)
        np.testing.assert_array_less(np.abs(matched - roots), tol)
        # Complex eigenvalues of a real matrix come in conjugate pairs,
        # positive imaginary part first.
        for it in range(len(w)):
            if w[it].imag > 0.0:
                self.assertEqual(w[it + 1], np.conj(w[it]))

    def test_rotation(self):
        theta = 0.7
        axis = np.array([1.0, 2.0, 2.0]) / 3.0
        kmat = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        rot = np.eye(3) + np.sin(theta) * kmat \
            + (1.0 - np.cos(theta)) * kmat @ kmat
        w, v = self.check_pairs(rot, sc.EigenDecompositionFloat64(
            to_sa(rot)))
        expected = np.array([np.exp(1j * theta), np.exp(-1j * theta), 1.0])
        # A rotation is normal, so kappa_i = 1 and the bound is the backward
        # error alone, 10 * n * eps * |A|_F.
        tol = 10.0 * 3 * EPS * np.linalg.norm(rot)
        np.testing.assert_array_less(
            np.abs(match_to(expected, w) - expected), tol)
        # The eigenvector of the real eigenvalue 1 is the rotation axis.
        ireal = int(np.argmin(np.abs(w - 1.0)))
        self.assertLessEqual(
            abs(abs(np.real(v[:, ireal]) @ axis) - 1.0), tol)

    def test_jordan_block_eigenvalues(self):
        # A defective Jordan block J_k(2) under an orthogonal similarity:
        # the rounding of the similarity (|E| ~ eps |A|) moves a k-fold
        # defective eigenvalue by ~ |E|^(1/k) (Lidskii), so the bound is
        # 10 * (eps * |A|_F)^(1/k).
        k = 4
        jordan = 2.0 * np.eye(k) + np.diag(np.ones(k - 1), 1)
        q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((k, k)))
        a = q @ jordan @ q.T
        w = np.array(sc.eigvals(to_sa(a)))
        tol = 10.0 * (EPS * np.linalg.norm(a)) ** (1.0 / k)
        np.testing.assert_array_less(np.abs(w - 2.0), tol)
        # The mean of the cluster is well conditioned (trace / k).
        self.assertLessEqual(abs(w.mean() - 2.0),
                             10.0 * k * EPS * np.linalg.norm(a))

    def test_random_vs_numpy(self):
        for n in (1, 2, 3, 5, 8, 20, 50):
            with self.subTest(n=n):
                a = np.random.default_rng(n).standard_normal((n, n))
                w, _ = self.check_pairs(
                    a, sc.EigenDecompositionFloat64(to_sa(a)))
                self.check_vs_numpy(a, w)

    def test_badly_scaled_balancing(self):
        # Rows and columns scaled over 12 orders of magnitude: balancing
        # (powers of 2) undoes the scaling exactly, so the eigenvalues keep
        # the accuracy of the unscaled matrix.
        rng = np.random.default_rng(4)
        base = rng.standard_normal((6, 6))
        scale = 10.0 ** np.linspace(-6.0, 6.0, 6)
        a = (base * scale[None, :]) / scale[:, None]
        eig = sc.EigenDecompositionFloat64(to_sa(a), balance=True)
        w = np.array(eig.eigenvalues)
        w_ref, tol = eigenvalue_tolerance(base)
        np.testing.assert_array_less(np.abs(match_to(w_ref, w) - w_ref), tol)
        # Without balancing the same matrix still converges.
        w_nb = np.array(sc.eigvals(to_sa(a), balance=False))
        self.assertEqual(w_nb.shape, (6,))

    def test_eigenvalues_only(self):
        a = np.random.default_rng(5).standard_normal((10, 10))
        eig = sc.EigenDecompositionFloat64(to_sa(a), compute_vectors=False)
        self.assertFalse(eig.has_vectors)
        with self.assertRaisesRegex(RuntimeError, "eigenvectors"):
            eig.eigenvectors
        full = sc.EigenDecompositionFloat64(to_sa(a))
        self.assertTrue(full.has_vectors)
        np.testing.assert_array_equal(np.array(eig.eigenvalues),
                                      np.array(full.eigenvalues))

    def test_free_eig(self):
        a = np.random.default_rng(6).standard_normal((7, 7))
        w, v = sc.eig(to_sa(a))
        w, v = np.array(w), np.array(v)
        for it in range(7):
            res = np.linalg.norm(a @ v[:, it] - w[it] * v[:, it])
            self.assertLessEqual(res / np.linalg.norm(a), RESIDUAL_TOL)

    def test_zero_and_identity(self):
        for a in (np.zeros((4, 4)), np.eye(4)):
            with self.subTest(a=a[0, 0]):
                w, v = self.check_pairs(
                    a, sc.EigenDecompositionFloat64(to_sa(a)))
                np.testing.assert_array_equal(w, a[0, 0])
                np.testing.assert_array_equal(np.abs(v), np.eye(4))

    def test_large_random_convergence(self):
        n = 200
        a = np.random.default_rng(200).standard_normal((n, n))
        start = time.perf_counter()
        eig = sc.EigenDecompositionFloat64(to_sa(a))
        elapsed = time.perf_counter() - start
        w, _ = self.check_pairs(a, eig)
        self.check_vs_numpy(a, w)
        # LAPACK dlahqr allows 30 * max(10, n) QR sweeps per deflation; the
        # Francis iteration needs about 2 to 4 per eigenvalue.
        self.assertLessEqual(eig.iterations, 10 * n)
        # A loose guard against an accidental O(n^4) path; the measured time
        # is about 0.1 s.
        self.assertLess(elapsed, 30.0)

    def test_reject_bad_input(self):
        with self.assertRaisesRegex(ValueError, "square"):
            sc.EigenDecompositionFloat64(to_sa(np.ones((3, 2))))
        a = np.ones((3, 3))
        a[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sc.EigenDecompositionFloat64(to_sa(a))

    def test_float32(self):
        a = np.random.default_rng(7).standard_normal((6, 6))
        eig = sc.EigenDecompositionFloat32(
            sc.SimpleArrayFloat32(array=a.astype(np.float32)))
        w = np.array(eig.eigenvalues)
        v = np.array(eig.eigenvectors)
        self.assertEqual(w.dtype, np.complex64)
        # Single precision: the backward error bound with eps(float32).
        tol = 1000.0 * np.finfo(np.float32).eps
        a32 = a.astype(np.float32).astype(np.float64)
        for it in range(6):
            res = np.linalg.norm(a32 @ v[:, it] - w[it] * v[:, it])
            self.assertLessEqual(res / np.linalg.norm(a32), tol)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
