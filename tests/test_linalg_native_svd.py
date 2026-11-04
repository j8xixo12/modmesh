# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Test the native (no vendor LAPACK) one-sided Jacobi SVD,
sc.SvdFactorizationFloat64, and the free functions sc.svd, sc.pinv,
sc.null_space, and sc.lstsq built on it.
"""

import unittest

import numpy as np

import solvcon as sc


EPS = np.finfo(np.float64).eps

# The reconstruction A = U diag(s) V^T and the orthogonality of U and V are
# backward-stable quantities (errors of a small multiple of eps, independent
# of the condition number), so the bounds are fixed at 1e-13.
RECON_TOL = 1e-13
ORTHO_TOL = 1e-13


def sv_tolerance(m, n, smax):
    """
    Bound on |s_i - s_i(numpy)|. Both SVDs are backward stable, and a
    singular value moves by at most the 2-norm of the backward error (Weyl),
    which is a small multiple of max(m, n) * eps * |A|_2; 10 is the small
    multiple.
    """
    return 10.0 * max(m, n) * EPS * smax


def solution_tolerance(m, n, cond):
    """
    Relative bound on the difference of pinv(A) or of a least-squares
    solution from numpy: the perturbation theory of the least-squares problem
    (Golub and Van Loan, Thm. 5.3.1) bounds the relative change by cond(A)^2
    times the backward error, here 10 * max(m, n) * eps.
    """
    return 10.0 * max(m, n) * EPS * cond ** 2


def make_matrix(m, n, cond, seed):
    """Random m x n matrix with singular values from 1 down to 1/cond."""
    rng = np.random.default_rng(seed)
    k = min(m, n)
    u, _ = np.linalg.qr(rng.standard_normal((m, k)))
    v, _ = np.linalg.qr(rng.standard_normal((n, k)))
    s = np.logspace(0.0, -np.log10(cond), k)
    return (u * s) @ v.T


def make_rank(m, n, rank, seed, cond=10.0):
    """Random m x n matrix of exact rank with nonzero singular values in
    [1/cond, 1]."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((m, rank)))
    v, _ = np.linalg.qr(rng.standard_normal((n, rank)))
    s = np.logspace(0.0, -np.log10(cond), rank)
    return (u * s) @ v.T


def skew(r):
    return np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ])


def four_bar_jacobian():
    """
    Velocity constraint Jacobian (20 x 18, rank 17) of a planar four-bar
    modeled with 3D revolute joints; see test_linalg_native_qr.py.
    """
    pa = np.array([0.0, 0.0, 0.0])
    pb = np.array([0.5, np.sqrt(3.0) / 2.0, 0.0])
    pc = np.array([3.5, 2.0, 0.0])
    pd = np.array([4.0, 0.0, 0.0])
    centers = [(pa + pb) / 2.0, (pb + pc) / 2.0, (pc + pd) / 2.0]
    joints = [(-1, 0, pa), (0, 1, pb), (1, 2, pc), (2, -1, pd)]
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    jac = np.zeros((20, 18))
    for ij, (ibody, jbody, point) in enumerate(joints):
        rows = slice(5 * ij, 5 * ij + 3)
        for body, sign in ((ibody, 1.0), (jbody, -1.0)):
            if body < 0:
                continue
            col = 6 * body
            arm = point - centers[body]
            jac[rows, col:col + 3] = sign * np.eye(3)
            jac[rows, col + 3:col + 6] = -sign * skew(arm)
            jac[5 * ij + 3, col + 3:col + 6] = sign * ex
            jac[5 * ij + 4, col + 3:col + 6] = sign * ey
    return jac


def to_sa(a):
    return sc.SimpleArrayFloat64(array=np.ascontiguousarray(a))


def factorize(a):
    return sc.SvdFactorizationFloat64(to_sa(a))


class SvdFactorizationTC(unittest.TestCase):

    SHAPES = [(6, 6), (12, 5), (5, 12), (1, 1), (1, 7), (7, 1), (30, 30),
              (40, 17), (17, 40)]

    def check_svd(self, a, svd):
        m, n = a.shape
        k = min(m, n)
        u, s, v = np.array(svd.u), np.array(svd.s), np.array(svd.v)
        self.assertEqual(u.shape, (m, k))
        self.assertEqual(s.shape, (k,))
        self.assertEqual(v.shape, (n, k))
        self.assertTrue(np.all(s >= 0.0))
        self.assertTrue(np.all(np.diff(s) <= 0.0))

        anorm = max(np.linalg.norm(a), np.finfo(float).tiny)
        recon = np.linalg.norm(a - (u * s) @ v.T) / anorm
        self.assertLessEqual(recon, RECON_TOL)
        self.assertLessEqual(np.abs(u.T @ u - np.eye(k)).max(), ORTHO_TOL)
        self.assertLessEqual(np.abs(v.T @ v - np.eye(k)).max(), ORTHO_TOL)
        return u, s, v

    def test_random_shapes_vs_numpy(self):
        for m, n in self.SHAPES:
            with self.subTest(shape=(m, n)):
                a = make_matrix(m, n, cond=1e4, seed=m * 100 + n)
                svd = factorize(a)
                self.assertEqual((svd.m, svd.n, svd.k), (m, n, min(m, n)))
                _, s, _ = self.check_svd(a, svd)
                s_ref = np.linalg.svd(a, compute_uv=False)
                np.testing.assert_allclose(
                    s, s_ref, rtol=0.0, atol=sv_tolerance(m, n, s_ref[0]))

    def test_singular_vectors_vs_numpy(self):
        # Well separated singular values (ratio 10 apart): a singular vector
        # moves by the backward error over the gap (Wedin), so with a
        # relative gap of at least 0.5 the bound is twice sv_tolerance.
        m, n = 9, 5
        rng = np.random.default_rng(8)
        u0, _ = np.linalg.qr(rng.standard_normal((m, n)))
        v0, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s0 = 10.0 ** -np.arange(n, dtype=float)
        a = (u0 * s0) @ v0.T
        u, s, v = self.check_svd(a, factorize(a))
        u_ref, s_ref, vt_ref = np.linalg.svd(a, full_matrices=False)
        for it in range(n):
            tol = 2.0 * sv_tolerance(m, n, s_ref[0]) / (s_ref[it] * 0.5)
            sign = np.sign(u[:, it] @ u_ref[:, it])
            np.testing.assert_allclose(sign * u[:, it], u_ref[:, it],
                                       rtol=0.0, atol=tol)
            np.testing.assert_allclose(sign * v[:, it], vt_ref[it],
                                       rtol=0.0, atol=tol)

    def test_free_svd(self):
        a = make_matrix(8, 6, cond=10.0, seed=2)
        u, s, v = sc.svd(to_sa(a))
        u, s, v = np.array(u), np.array(s), np.array(v)
        self.assertLessEqual(
            np.linalg.norm(a - (u * s) @ v.T) / np.linalg.norm(a), RECON_TOL)

    def test_hilbert_8x8(self):
        n = 8
        idx = np.arange(n)
        hilbert = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
        _, s, _ = self.check_svd(hilbert, factorize(hilbert))
        s_ref = np.linalg.svd(hilbert, compute_uv=False)
        np.testing.assert_allclose(
            s, s_ref, rtol=0.0, atol=sv_tolerance(n, n, s_ref[0]))
        # Full numerical rank: s_min ~ 1.1e-10 is far above max(m, n) * eps.
        self.assertEqual(factorize(hilbert).rank(), n)

    def test_rank_known(self):
        cases = [(8, 8, 3), (10, 6, 1), (6, 10, 4), (20, 15, 9), (15, 20, 9)]
        for m, n, rank in cases:
            with self.subTest(shape=(m, n), rank=rank):
                a = make_rank(m, n, rank, seed=m + 2 * n + rank)
                svd = factorize(a)
                self.assertEqual(svd.rank(), rank)
                _, s, _ = self.check_svd(a, svd)
                # The zero singular values come out at rounding level.
                self.assertLessEqual(s[rank:].max(initial=0.0),
                                     sv_tolerance(m, n, s[0]))

    def test_rank_duplicated_rows_columns(self):
        rng = np.random.default_rng(4)
        base = rng.standard_normal((5, 4))
        cols = np.hstack([base, base[:, :1], base[:, 1:2] - base[:, 2:3]])
        self.assertEqual(factorize(cols).rank(), 4)
        rows = np.vstack([base.T, 2.0 * base.T[:2]])
        self.assertEqual(factorize(rows).rank(), 4)

    def test_rank_rtol(self):
        svd = factorize(np.diag([1.0, 1e-3, 1e-9]))
        self.assertEqual(svd.rank(), 3)
        self.assertEqual(svd.rank(1e-6), 2)
        self.assertEqual(svd.rank(1e-2), 1)

    def test_four_bar_jacobian(self):
        jac = four_bar_jacobian()
        svd = factorize(jac)
        self.assertEqual(svd.rank(), 17)
        self.check_svd(jac, svd)

        null = np.array(svd.null_space())
        self.assertEqual(null.shape, (18, 1))
        self.assertLessEqual(np.abs(jac @ null).max(),
                             RECON_TOL * np.linalg.norm(jac))
        self.assertLessEqual(abs(np.linalg.norm(null) - 1.0), ORTHO_TOL)
        # The null space is the single mechanism mode; compare the projector
        # with the one from numpy (subspace angle bound: backward error over
        # the gap s_16, Wedin).
        _, s_ref, vt_ref = np.linalg.svd(jac)
        ref = vt_ref[17:].T
        tol = 2.0 * sv_tolerance(20, 18, s_ref[0]) / s_ref[16]
        np.testing.assert_allclose(null @ null.T, ref @ ref.T,
                                   rtol=0.0, atol=tol)

        # Minimum-norm least squares of the redundant velocity equations:
        # the consistent right-hand side is solved exactly, and the solution
        # is orthogonal to the mechanism mode.
        qdot = np.random.default_rng(9).standard_normal(18)
        rhs = jac @ qdot
        x = np.array(sc.lstsq(to_sa(jac), to_sa(rhs)))
        self.assertLessEqual(np.abs(jac @ x - rhs).max(),
                             RECON_TOL * np.linalg.norm(jac)
                             * np.linalg.norm(qdot) * 10.0)
        self.assertLessEqual(abs(null[:, 0] @ x), tol * np.linalg.norm(x))

    def test_zero_matrix(self):
        for m, n in [(4, 3), (3, 4), (3, 3)]:
            with self.subTest(shape=(m, n)):
                a = np.zeros((m, n))
                svd = factorize(a)
                self.assertEqual(svd.rank(), 0)
                u, s, v = self.check_svd(a, svd)
                np.testing.assert_array_equal(s, 0.0)
                null = np.array(svd.null_space())
                self.assertEqual(null.shape, (n, n))
                self.assertLessEqual(
                    np.abs(null.T @ null - np.eye(n)).max(), ORTHO_TOL)
                np.testing.assert_array_equal(
                    np.array(sc.pinv(to_sa(a))), np.zeros((n, m)))

    def test_identity(self):
        svd = factorize(np.eye(5))
        u, s, v = self.check_svd(np.eye(5), svd)
        np.testing.assert_array_equal(s, 1.0)
        self.assertEqual(svd.rank(), 5)
        self.assertEqual(np.array(svd.null_space()).shape, (5, 0))

    def test_null_space_wide(self):
        # m < n: the null space needs the right singular vectors beyond the
        # thin V, i.e. the full n x n V.
        for m, n, rank in [(3, 7, 3), (4, 9, 2), (1, 5, 1)]:
            with self.subTest(shape=(m, n), rank=rank):
                a = make_rank(m, n, rank, seed=m * n + rank)
                null = np.array(sc.null_space(to_sa(a)))
                self.assertEqual(null.shape, (n, n - rank))
                self.assertLessEqual(np.abs(a @ null).max(),
                                     RECON_TOL * np.linalg.norm(a))
                self.assertLessEqual(
                    np.abs(null.T @ null - np.eye(n - rank)).max(),
                    ORTHO_TOL)

    def test_pinv_vs_numpy(self):
        for m, n in [(6, 6), (12, 5), (5, 12), (1, 1), (1, 7), (7, 1)]:
            with self.subTest(shape=(m, n)):
                cond = 100.0
                a = make_matrix(m, n, cond=cond, seed=m + 7 * n)
                pinv = np.array(sc.pinv(to_sa(a)))
                ref = np.linalg.pinv(a)
                self.assertEqual(pinv.shape, (n, m))
                err = np.linalg.norm(pinv - ref) / np.linalg.norm(ref)
                self.assertLessEqual(err, solution_tolerance(m, n, cond))

    def test_pinv_rank_deficient(self):
        # Rank 4 with nonzero singular values in [0.1, 1]: the pseudo-inverse
        # is well conditioned (cond of the nonzero part is 10), and the four
        # Penrose conditions hold to the same bound.
        m, n, cond = 9, 7, 10.0
        a = make_rank(m, n, 4, seed=13, cond=cond)
        x = np.array(sc.pinv(to_sa(a)))
        ref = np.linalg.pinv(a)
        tol = solution_tolerance(m, n, cond)
        self.assertLessEqual(
            np.linalg.norm(x - ref) / np.linalg.norm(ref), tol)
        self.assertLessEqual(np.linalg.norm(a @ x @ a - a), tol)
        self.assertLessEqual(np.linalg.norm(x @ a @ x - x), tol * cond)
        self.assertLessEqual(np.abs(a @ x - (a @ x).T).max(), tol)
        self.assertLessEqual(np.abs(x @ a - (x @ a).T).max(), tol)

    def test_lstsq_vs_numpy(self):
        rng = np.random.default_rng(17)
        cases = [(12, 5, 5), (5, 12, 5), (9, 9, 9), (10, 8, 3), (6, 11, 2)]
        for m, n, rank in cases:
            for nrhs in (None, 3):
                with self.subTest(shape=(m, n), rank=rank, nrhs=nrhs):
                    cond = 10.0
                    a = make_rank(m, n, rank, seed=m + n + rank, cond=cond)
                    shape = (m,) if nrhs is None else (m, nrhs)
                    b = rng.standard_normal(shape)
                    x = np.array(sc.lstsq(to_sa(a), to_sa(b)))
                    ref = np.linalg.lstsq(a, b, rcond=None)[0]
                    self.assertEqual(x.shape, ref.shape)
                    err = np.linalg.norm(x - ref) / np.linalg.norm(ref)
                    self.assertLessEqual(err, solution_tolerance(m, n, cond))

    def test_solve_method(self):
        a = make_matrix(7, 4, cond=10.0, seed=19)
        b = np.random.default_rng(20).standard_normal((7, 2))
        x = np.array(factorize(a).solve(to_sa(b)))
        ref = np.linalg.lstsq(a, b, rcond=None)[0]
        self.assertLessEqual(np.linalg.norm(x - ref) / np.linalg.norm(ref),
                             solution_tolerance(7, 4, 10.0))

    def test_reject_bad_input(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            sc.SvdFactorizationFloat64(sc.SimpleArrayFloat64(4))
        a = np.ones((3, 3))
        a[0, 2] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            factorize(a)
        svd = factorize(np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "rows"):
            svd.solve(sc.SimpleArrayFloat64(array=np.ones(4)))

    def test_float32(self):
        a = make_matrix(7, 5, cond=10.0, seed=23).astype(np.float32)
        u, s, v = sc.svd(sc.SimpleArrayFloat32(array=a))
        u, s, v = np.array(u), np.array(s), np.array(v)
        self.assertEqual(s.dtype, np.float32)
        # Single precision: the same backward-stable bound with eps(float32).
        tol = 100.0 * np.finfo(np.float32).eps
        self.assertLessEqual(
            np.linalg.norm(a - (u * s) @ v.T) / np.linalg.norm(a), tol)
        s_ref = np.linalg.svd(a.astype(np.float64), compute_uv=False)
        np.testing.assert_allclose(s, s_ref, rtol=0.0, atol=tol)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
