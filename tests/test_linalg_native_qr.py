# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Test the native (no vendor LAPACK) Householder QR with column pivoting,
sc.QrPivotFactorizationFloat64 and sc.qr_pivot.
"""

import unittest

import numpy as np

import solvcon as sc

try:
    from scipy import linalg as sp_linalg
except ImportError:  # pragma: no cover
    sp_linalg = None


EPS = np.finfo(np.float64).eps

# The reconstruction A P = Q R and the orthogonality of Q are backward-stable
# quantities: Householder QR gives errors of a small multiple of eps,
# independent of the condition number, so the bound is fixed at 1e-13.
RECON_TOL = 1e-13
ORTHO_TOL = 1e-13

# The column norms are downdated as in LAPACK dlaqp2 and recomputed only when
# the downdate loses more than sqrt(eps) of relative accuracy (tol3z), so the
# pivot order, hence |R_kk| non-increasing, is guaranteed to that level.
ORDER_SLACK = np.sqrt(EPS)


def make_matrix(m, n, cond, seed):
    """Random m x n matrix with singular values from 1 down to 1/cond."""
    rng = np.random.default_rng(seed)
    k = min(m, n)
    u, _ = np.linalg.qr(rng.standard_normal((m, k)))
    v, _ = np.linalg.qr(rng.standard_normal((n, k)))
    s = np.logspace(0.0, -np.log10(cond), k)
    return (u * s) @ v.T


def make_rank(m, n, rank, seed):
    """Random m x n matrix of exact rank (outer products of random vectors)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, rank)) @ rng.standard_normal((rank, n))


def skew(r):
    return np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ])


def four_bar_jacobian():
    """
    Velocity constraint Jacobian of a planar four-bar modeled with 3D
    revolute joints.

    Three moving bodies (crank, coupler, rocker) each have 6 velocity
    coordinates (v, w), 18 in all. Four revolute joints, all with axes along
    z, each give 5 rows: 3 for the coincident joint point and 2 for the axis
    alignment, 20 in all. The mechanism has 1 degree of freedom, so the rank
    is 17 and 3 constraint rows are redundant.
    """
    pa = np.array([0.0, 0.0, 0.0])
    pb = np.array([0.5, np.sqrt(3.0) / 2.0, 0.0])
    pc = np.array([3.5, 2.0, 0.0])
    pd = np.array([4.0, 0.0, 0.0])
    centers = [(pa + pb) / 2.0, (pb + pc) / 2.0, (pc + pd) / 2.0]
    # (body i, body j, joint point); -1 is the ground.
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
            # v + w x arm = v - skew(arm) w
            jac[rows, col:col + 3] = sign * np.eye(3)
            jac[rows, col + 3:col + 6] = -sign * skew(arm)
            jac[5 * ij + 3, col + 3:col + 6] = sign * ex
            jac[5 * ij + 4, col + 3:col + 6] = sign * ey
    return jac


def factorize(a):
    return sc.QrPivotFactorizationFloat64(sc.SimpleArrayFloat64(array=a))


class QrPivotTC(unittest.TestCase):

    SHAPES = [(6, 6), (12, 5), (5, 12), (1, 1), (1, 7), (7, 1), (30, 30),
              (40, 17)]

    def check_factorization(self, a, qrp):
        m, n = a.shape
        k = min(m, n)
        q = np.array(qrp.q())
        r = np.array(qrp.r())
        perm = np.array(qrp.perm)
        self.assertEqual(q.shape, (m, k))
        self.assertEqual(r.shape, (k, n))
        self.assertEqual(sorted(perm.tolist()), list(range(n)))
        np.testing.assert_array_equal(np.tril(r, -1), 0.0)

        anorm = max(np.linalg.norm(a), np.finfo(float).tiny)
        recon = np.linalg.norm(a[:, perm] - q @ r) / anorm
        self.assertLessEqual(recon, RECON_TOL)
        ortho = np.abs(q.T @ q - np.eye(k)).max()
        self.assertLessEqual(ortho, ORTHO_TOL)

        diag = np.abs(np.diag(r))
        for it in range(len(diag) - 1):
            self.assertLessEqual(
                diag[it + 1], diag[it] * (1.0 + ORDER_SLACK) + EPS * diag[0])
        return q, r, perm

    def test_random_shapes(self):
        for m, n in self.SHAPES:
            with self.subTest(shape=(m, n)):
                a = make_matrix(m, n, cond=1e3, seed=m * 100 + n)
                qrp = factorize(a)
                self.assertEqual((qrp.m, qrp.n, qrp.k), (m, n, min(m, n)))
                self.check_factorization(a, qrp)

    def test_free_function(self):
        a = make_matrix(9, 6, cond=10.0, seed=7)
        q, r, perm = sc.qr_pivot(sc.SimpleArrayFloat64(array=a))
        q, r, perm = np.array(q), np.array(r), np.array(perm)
        self.assertEqual(q.shape, (9, 6))
        self.assertEqual(r.shape, (6, 6))
        self.assertLessEqual(
            np.linalg.norm(a[:, perm] - q @ r) / np.linalg.norm(a),
            RECON_TOL)

    @unittest.skipIf(sp_linalg is None, "scipy is not available")
    def test_matches_lapack_geqp3(self):
        # scipy.linalg.qr(pivoting=True) calls LAPACK geqp3, the algorithm
        # implemented natively, so the pivot order must be the same. R agrees
        # row by row up to the sign of the Householder vector; the error
        # bound of a QR factor is cond(A) * eps times a small multiple of
        # the size (Higham, Accuracy and Stability, Thm. 19.4).
        for m, n in self.SHAPES:
            with self.subTest(shape=(m, n)):
                cond = 1e3
                a = make_matrix(m, n, cond=cond, seed=m * 31 + n)
                qrp = factorize(a)
                r = np.array(qrp.r())
                perm = np.array(qrp.perm)
                _, r_ref, perm_ref = sp_linalg.qr(
                    a, mode='economic', pivoting=True)
                np.testing.assert_array_equal(perm, perm_ref)
                signs = np.sign(np.diag(r)) * np.sign(np.diag(r_ref))
                tol = 10.0 * max(m, n) * cond * EPS * np.abs(r_ref).max()
                np.testing.assert_allclose(
                    signs[:, None] * r, r_ref, rtol=0.0, atol=tol)

    def test_apply_q(self):
        m, n = 11, 7
        a = make_matrix(m, n, cond=100.0, seed=3)
        qrp = factorize(a)
        perm = np.array(qrp.perm)
        r = np.array(qrp.r())

        # Q^T (A P) is R on top and zero below (full m x m Q).
        qta = np.array(qrp.apply_qt(sc.SimpleArrayFloat64(
            array=np.ascontiguousarray(a[:, perm]))))
        self.assertEqual(qta.shape, (m, n))
        scale = np.linalg.norm(a)
        self.assertLessEqual(np.abs(qta[:n] - r).max(), RECON_TOL * scale)
        self.assertLessEqual(np.abs(qta[n:]).max(), RECON_TOL * scale)

        # The full Q is orthogonal: Q (Q^T b) = b, and Q e_j gives the thin
        # Q for j < k.
        b = np.random.default_rng(5).standard_normal(m)
        qtb = qrp.apply_qt(sc.SimpleArrayFloat64(array=b))
        back = np.array(qrp.apply_q(qtb))
        self.assertEqual(back.shape, (m,))
        self.assertLessEqual(np.abs(back - b).max(),
                             RECON_TOL * np.abs(b).max())
        qfull = np.array(qrp.apply_q(sc.SimpleArrayFloat64(array=np.eye(m))))
        self.assertLessEqual(np.abs(qfull.T @ qfull - np.eye(m)).max(),
                             ORTHO_TOL)
        np.testing.assert_allclose(qfull[:, :n], np.array(qrp.q()),
                                   rtol=0.0, atol=ORTHO_TOL)

    def test_rank_outer_products(self):
        for m, n, rank in [(8, 8, 3), (10, 6, 1), (6, 10, 4), (20, 15, 9)]:
            with self.subTest(shape=(m, n), rank=rank):
                a = make_rank(m, n, rank, seed=m + n + rank)
                qrp = factorize(a)
                self.assertEqual(qrp.rank(), rank)
                self.assertEqual(np.linalg.matrix_rank(a), rank)
                self.check_factorization(a, qrp)

    def test_rank_duplicated_columns(self):
        rng = np.random.default_rng(11)
        base = rng.standard_normal((9, 4))
        a = np.hstack([base, base[:, :2], 3.0 * base[:, 3:4]])
        qrp = factorize(a)
        self.assertEqual(qrp.rank(), 4)
        q, r, perm = self.check_factorization(a, qrp)
        # The four independent leading pivots come first; the duplicates
        # leave |R_kk| at rounding level.
        self.assertLessEqual(np.abs(np.diag(r))[4:].max(),
                             100.0 * EPS * abs(r[0, 0]))

    def test_rank_duplicated_rows(self):
        rng = np.random.default_rng(12)
        base = rng.standard_normal((3, 7))
        a = np.vstack([base, base, base[:1] - base[1:2]])
        self.assertEqual(factorize(a).rank(), 3)

    def test_rank_four_bar_jacobian(self):
        jac = four_bar_jacobian()
        self.assertEqual(np.linalg.matrix_rank(jac), 17)
        qrp = factorize(jac)
        self.assertEqual(qrp.rank(), 17)
        self.check_factorization(jac, qrp)
        # The same rule on the transposed Jacobian (redundant rows become
        # dependent columns).
        self.assertEqual(factorize(np.ascontiguousarray(jac.T)).rank(), 17)

    def test_rank_rtol(self):
        a = np.diag([1.0, 1e-3, 1e-9])
        qrp = factorize(a)
        self.assertEqual(qrp.rank(), 3)
        self.assertEqual(qrp.rank(1e-6), 2)
        self.assertEqual(qrp.rank(1e-2), 1)

    def test_zero_matrix(self):
        a = np.zeros((4, 3))
        qrp = factorize(a)
        self.assertEqual(qrp.rank(), 0)
        np.testing.assert_array_equal(np.array(qrp.r()), 0.0)
        np.testing.assert_array_equal(np.array(qrp.perm), [0, 1, 2])
        q = np.array(qrp.q())
        np.testing.assert_array_equal(q, np.eye(4)[:, :3])

    def test_identity(self):
        qrp = factorize(np.eye(5))
        self.assertEqual(qrp.rank(), 5)
        np.testing.assert_array_equal(np.array(qrp.perm), np.arange(5))
        np.testing.assert_allclose(np.abs(np.array(qrp.r())), np.eye(5),
                                   rtol=0.0, atol=EPS)

    def test_reject_bad_input(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            sc.QrPivotFactorizationFloat64(sc.SimpleArrayFloat64(4))
        a = np.ones((3, 3))
        a[1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            factorize(a)
        qrp = factorize(np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "rows"):
            qrp.apply_q(sc.SimpleArrayFloat64(array=np.ones(4)))

    def test_float32(self):
        a = make_matrix(7, 5, cond=10.0, seed=21).astype(np.float32)
        q, r, perm = sc.qr_pivot(sc.SimpleArrayFloat32(array=a))
        q, r, perm = np.array(q), np.array(r), np.array(perm)
        self.assertEqual(q.dtype, np.float32)
        # Single precision: the same backward-stable bound with eps(float32).
        tol = 100.0 * np.finfo(np.float32).eps
        self.assertLessEqual(
            np.linalg.norm(a[:, perm] - q @ r) / np.linalg.norm(a), tol)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
