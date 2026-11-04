# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of solvcon.multibody.tmm.riccati: derivation utilities (Ch. 10.2-10.8)
and the Riccati TMM (Ch. 10.9).

Book: Rui, Wang, Zhang, Transfer Matrix Method for Multibody Systems (Wiley
2018). Tolerances: closed-form identities 1e-12 relative (rounding of a few
matrix products); eigenfrequencies 1e-10 relative unless stated.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest


from solvcon.multibody.tmm import library as lib
from solvcon.multibody.tmm import riccati as ric
from solvcon.multibody.tmm.linear import Chain

DATA = Path(__file__).parent / "data" / "multibody"
with open(DATA / "tmm_ch10_examples.json") as _fh:
    CH10 = json.load(_fh)["examples"]


def _example_10_11(force=1.0):
    d = CH10["10.11"]["data"]
    k, m = d["K"], d["m"]
    els = [lib.Spring(k), lib.LumpedMass(m, force=force), lib.Spring(2 * k),
           lib.LumpedMass(2 * m), lib.Spring(6 * k)]
    return Chain(els, [0], [0]), k, m


# ----------------------------------------------------------------------
# Example 10.11 (BK)
# ----------------------------------------------------------------------
def test_example_10_11_riccati_matrices_and_response():
    """BK Example 10.11 Eq. (h)-(l), p. 421-422: S_i, e_i, state vectors at
    Omega^2 = 3K/m."""
    ch, k, m = _example_10_11()
    ref = CH10["10.11"]
    res = ric.riccati_steady_state(ch, 1j * math.sqrt(3 * k / m))
    s_vals = [s_mat[0, 0].real for s_mat in res["S"]]
    e_vals = [e[0].real for e in res["e"]]
    np.testing.assert_allclose(s_vals, ref["S_at_Omega_over_unit"], atol=1e-12)
    np.testing.assert_allclose(e_vals, ref["e_at_Omega_over_F_K"], atol=1e-12)
    states = ref["states_over_F"]
    expected = [states["Z1_0"], states["Z2_1"], states["Z2_3"], states["Z4_3"],
                states["Z4_5"]]
    np.testing.assert_allclose(np.real(res["inputs"]), expected, atol=1e-12)
    np.testing.assert_allclose(np.real(res["outputs"][-1]), states["Z5_0"],
                               atol=1e-12)
    # XM: plain extended TMM (Eq. 2.119-2.120) gives the same response
    z = ch.steady_state(math.sqrt(3 * k / m))
    np.testing.assert_allclose(z[:-1], res["inputs"], atol=1e-12)


def test_example_10_11_s6_formula_and_eigenpairs():
    """BK Example 10.11 Eq. (c), (e)-(g): S_6(omega), eigenfrequencies,
    eigenvector ratios."""
    ch, k, m = _example_10_11(force=0.0)
    for w in (0.7, 1.9, 2.6):
        w2 = w * w
        mats = [el.transfer(1j * w) for el in ch.elements]
        sweep = ric.riccati_sweep(mats, [np.zeros(2)] * 5, [0], [1])
        s6 = sweep["S"][-1][0, 0].real
        book = -(m**2 * w2**2 - 7 * k * m * w2 + 10 * k**2) / (
            6 * k * (m**2 * w2**2 - 4 * k * m * w2 + k**2))
        assert s6 == pytest.approx(book, rel=1e-12)
    w_r = ch.natural_frequencies_riccati(3.0)
    np.testing.assert_allclose(
        w_r**2, np.array(CH10["10.11"]["omega2_over_K_m"]) * k / m,
        rtol=1e-12)
    ratios = [md.states[3][0] / md.states[1][0] for md in ch.modes(w_r)]
    np.testing.assert_allclose(ratios,
                               CH10["10.11"]["eigenvector_ratio_X43_over_X21"],
                               rtol=1e-10)


# ----------------------------------------------------------------------
# Riccati vs plain TMM (XM) and precision on a long chain (Sec. 10.9-10.10)
# ----------------------------------------------------------------------
def _ex_2_6():
    els = []
    for _ in range(4):
        els += [lib.PlanarElasticHinge(1e3, 5e4, 4e3),
                lib.PlanarRigidBody(1.0, 5.0 / 12.0 * 0.01, (0.1, 0.0),
                                    (0.05, 0.0))]
    return Chain(els, [0, 1, 2], [3, 4, 5])


def test_riccati_characteristic_equals_plain_determinant():
    """XM Eq. (10.134): det(W_c) prod det(G_i) = det(U[c][:, b]) where both are
    accurate."""
    ch = _ex_2_6()
    for w in (5.0, 40.0, 150.0, 700.0):
        sign_r, log_r = ric.riccati_characteristic(ch, 1j * w, 0.0)
        mat = ch.characteristic_matrix(1j * w, 0.0).real
        sign_p, log_p = np.linalg.slogdet(mat)
        assert np.real(sign_r) == sign_p
        assert log_r == pytest.approx(log_p, abs=1e-9)


def _ex_2_6_global_eigenfrequencies():
    """sqrt of the eigenvalues of L^-1 K L^-T, M = L L^T, for the assembled
    Fig. 2.22 chain."""
    m, ln, c1 = 1.0, 0.1, 0.05
    mb = np.array([[m, 0, 0], [0, m, m * c1],
                   [0, m * c1, m * ln**2 / 6 + m * c1**2]])
    mg = np.kron(np.eye(4), mb)
    kd = np.diag([1e3, 5e4, 4e3])
    kg = np.zeros((12, 12))
    for i in range(4):
        b = np.zeros((3, 12))
        b[:, 3 * i:3 * i + 3] = np.eye(3)
        if i > 0:
            b[:,
              3 * i - 3:3 * i] = -np.array([[1, 0, 0], [0, 1, ln], [0, 0, 1]])
        kg += b.T @ kd @ b
    li = np.linalg.inv(np.linalg.cholesky(mg))
    return np.sqrt(np.sort(np.linalg.eigvalsh(li @ kg @ li.T)))


def test_riccati_frequencies_vs_plain_and_global_eigenproblem():
    """XM/IV Sec. 10.9.3: Riccati and plain TMM vs the symmetric eigenproblem
    of the same chain.

    Riccati keeps 1e-13 for all 12 modes of the stiff chain of Table 2.3;
    the plain TMM loses digits in the highest modes (about 5e-11 at mode 12).
    """
    ch = _ex_2_6()
    exact = _ex_2_6_global_eigenfrequencies()
    np.testing.assert_allclose(ch.natural_frequencies_riccati(3500.0), exact,
                               rtol=1e-13)
    plain = ch.natural_frequencies(3500.0)
    np.testing.assert_allclose(plain, exact, rtol=1e-10)
    assert abs(plain[-1] - exact[-1]) / exact[-1] > 1e-12


def _cantilever_roots(n):
    out = []
    for k in range(1, n + 1):
        a, b = max(0.5, (k - 0.5) * np.pi - 1.0), (k - 0.5) * np.pi + 1.0
        for _ in range(200):
            c = 0.5 * (a + b)
            if (np.sign(np.cos(c) + 1 / np.cosh(c))
                    == np.sign(np.cos(a) + 1 / np.cosh(a))):
                a = c
            else:
                b = c
        out.append(0.5 * (a + b))
    return np.array(out)


def test_riccati_keeps_precision_where_plain_tmm_fails():
    """IV/AN Sec. 10.9-10.10: cantilever of 20 beam elements, 20 modes.

    Plain TMM: the characteristic determinant is a difference of products of
    size cosh(lambda L)^2, so above lambda L ~ 36 (mode 11) rounding noise
    dominates and the scan reports spurious roots. Riccati TMM: the error
    stays at the root tolerance level for all 20 modes. The Riccati
    tolerance 1e-8 is 10 x the largest error seen (1e-9, near cancelling
    zero-pole pairs of the factors det G_i).
    """
    n_el, n_modes = 20, 20
    ch = Chain([lib.EulerBernoulliBeam(1.0 / n_el, 1.0,
                                       1.0) for _ in range(n_el)], [0, 1],
               [2, 3])
    exact = _cantilever_roots(n_modes) ** 2
    w_max = exact[-1] * 1.01
    w_r = ch.natural_frequencies_riccati(w_max, n_grid=1500, max_halvings=1)
    assert len(w_r) == n_modes
    np.testing.assert_allclose(w_r, exact, rtol=1e-8)
    w_p = ch.natural_frequencies(w_max, n_grid=1500, max_halvings=1)
    # the first modes agree, the precision loss grows with the mode number
    np.testing.assert_allclose(w_p[:6], exact[:6], rtol=1e-9)
    assert len(w_p) > n_modes  # spurious sign changes from rounding noise
    err_10 = np.min(np.abs(w_p - exact[9])) / exact[9]
    assert err_10 > 1e-7


def test_riccati_steady_state_damped_torsion_equals_plain():
    """XM Eq. (10.135): Riccati steady state of Example 2.14 = complex extended
    TMM."""
    els = [lib.Disk(2.0), lib.RotarySpring(8.0 / 3.0),
           lib.Disk(1.0, c_ground=2.2), lib.RotarySpring(0.0, c=3.2),
           lib.Disk(1.1, torque=-0.75 - 0.70j), lib.RotarySpring(2.5, g=0.2),
           lib.Disk(1.5, torque=0.20 - 0.60j)]
    ch = Chain(els, [1], [1])
    big_omega = math.sqrt(0.8)
    plain = ch.steady_state(big_omega)
    res = ric.riccati_steady_state(ch, 1j * big_omega)
    np.testing.assert_allclose(np.array(res["inputs"]), plain[:-1], rtol=1e-12,
                               atol=1e-14)


def test_riccati_static_beam_equals_plain():
    """XM: Riccati static response (Sec. 10.9.1, static load) of Example 2.13 =
    plain TMM."""
    ln, ei, p = 1.0, 200.0, 10.0
    ch = Chain([lib.EulerBernoulliBeam(ln, ei, p=-p),
                lib.BeamPointMass(k=2 * ei / ln**3, moment=-p * ln**2),
                lib.EulerBernoulliBeam(2 * ln, ei)], [0, 2], [0, 1])
    res = ric.riccati_steady_state(ch, 0.0)
    np.testing.assert_allclose(np.real(res["inputs"]),
                               ch.static_response()[:-1], rtol=1e-12,
                               atol=1e-14)


# ----------------------------------------------------------------------
# Derivation utilities (Sec. 10.3-10.8)
# ----------------------------------------------------------------------
def test_expm_matches_series_eq_10_70():
    """AN Eq. (10.70): e^{As} = sum (As)^k / k! (series converges fast for a
    small norm)."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 5))
    for scale in (0.1, 0.5, 1.0):
        x = a * scale
        series = np.eye(5)
        term = np.eye(5)
        for k in range(1, 200):
            term = term @ x / k
            series = series + term
        np.testing.assert_allclose(ric.expm(x), series, rtol=1e-12,
                                   atol=1e-13 * np.abs(series).max())


def test_cayley_hamilton_euler_bernoulli_eq_10_75_10_77():
    """BK Eq. (10.75)-(10.77): Cayley-Hamilton e^{Ax} of the EB system matrix =
    Krylov form."""
    ei, m, w, x = 3.0, 2.0, 7.0, 0.8
    a = np.array([[0, 1, 0, 0], [0, 0, 1 / ei, 0], [0, 0, 0, 1],
                  [m * w**2, 0, 0, 0]])
    u_ch = ric.cayley_hamilton_transfer(a, x)
    u_kr = lib.EulerBernoulliBeam(x, ei, m).transfer(1j * w)
    np.testing.assert_allclose(u_ch, u_kr, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(ric.transfer_from_first_order(a, x), u_kr,
                               rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError):
        ric.cayley_hamilton_transfer(np.zeros((2, 2)), 1.0)


def test_fundamental_matrix_rod_example_10_2():
    """BK Example 10.2 Eq. (f)-(g), (10.66): U = B(x) B(0)^{-1} for the rod."""
    ea, m, w, x = 5.0, 1.5, 3.0, 0.7
    beta = math.sqrt(m * w**2 / ea)

    def b_of(xx):
        return np.array([[math.cos(beta * xx), math.sin(beta * xx)],
                         [beta * ea * math.sin(beta * xx),
                          -beta * ea * math.cos(beta * xx)]])

    u = ric.transfer_from_fundamental(b_of, x)
    np.testing.assert_allclose(u, lib.Rod(x, ea, m).transfer(1j * w),
                               rtol=1e-12, atol=1e-14)


def test_stiffness_to_transfer_example_10_5():
    """BK Example 10.5: transfer matrix of a massless beam from its stiffness
    matrix (Eq. 10.84).

    Stiffness relation with d = [y, theta], f = [m_z, q_y] in the sign
    convention of the state vector (internal forces at I and O), built from
    the standard finite element beam stiffness.
    """
    ei, ln = 4.0, 1.3
    # standard beam stiffness: [F1, M1, F2, M2] = K_fe [v1, theta1, v2, theta2]
    k_fe = ei / ln**3 * np.array([[12, 6 * ln, -12, 6 * ln],
                                 [6 * ln, 4 * ln**2, -6 * ln, 2 * ln**2],
                                 [-12, -6 * ln, 12, -6 * ln],
                                 [6 * ln, 2 * ln**2, -6 * ln, 4 * ln**2]])
    # internal forces of the state vector: m_I = -M1, q_I = F1, m_O = M2, q_O =
    # -F2
    sign = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    k = sign @ k_fe
    u = ric.transfer_from_stiffness(k, 2)
    book = lib.EulerBernoulliBeam(ln, ei).transfer(0.0).real
    np.testing.assert_allclose(u, book, rtol=1e-12, atol=1e-13)


def test_extended_matrix_uniform_loads_example_10_3_10_4():
    """BK Example 10.4 Eq. (e) and Table 10.2: load columns = int U(x - s) F ds
    (Eq. 10.80)."""
    ei, m, w, x, p, q_m = 3.0, 2.0, 5.0, 0.9, 1.7, -0.4
    a = np.array([[0, 1, 0, 0], [0, 0, 1 / ei, 0], [0, 0, 0, 1],
                  [m * w**2, 0, 0, 0]])
    ext = ric.extended_from_first_order(a, np.array([0, 0, -q_m, p]), x)
    beam = lib.EulerBernoulliBeam(x, ei, m, p=p, q_m=q_m)
    np.testing.assert_allclose(beam.forcing(1j * w), ext[:4, 4], rtol=1e-12,
                               atol=1e-14)
    ea = 6.0
    a_rod = np.array([[0, -1 / ea], [-m * w**2 * 0 + m * w**2, 0]])
    ext_r = ric.extended_from_first_order(a_rod, np.array([0, p]), x)
    rod = lib.Rod(x, ea, m, p=p)
    np.testing.assert_allclose(rod.forcing(1j * w), ext_r[:2, 2], rtol=1e-12)
    beta = math.sqrt(m * w**2 / ea)
    table = [-p / (m * w**2) * (1 - math.cos(beta * x)),
             p * math.sin(beta * x) / beta]
    np.testing.assert_allclose(rod.forcing(1j * w).real, table, rtol=1e-12)


def test_chessboard_inverse_and_symmetry_criterion():
    """BK Eq. (10.105)-(10.116), Examples 10.7 and 10.10.

    The Euler-Bernoulli inverse follows from the chessboard rule, det U = 1,
    and with the symmetric state definition of Fig. 10.20 (Theta and Q
    reversed at the input) U P is an involution (Eq. 10.114). The Timoshenko
    system matrix of Eq. (10.54) is also a black chessboard matrix, so the
    chessboard inverse holds for it as well (checked numerically).
    """
    u = lib.EulerBernoulliBeam(1.1, 2.0, 3.0).transfer(1j * 9.0).real
    np.testing.assert_allclose(ric.chessboard_inverse(u) @ u, np.eye(4),
                               atol=1e-12)
    assert np.linalg.det(u) == pytest.approx(1.0, rel=1e-12)
    flip = np.diag([1.0, -1.0, 1.0, -1.0])
    assert ric.is_symmetric_transfer(u @ flip)
    assert not ric.is_symmetric_transfer(u)
    ut = lib.TimoshenkoBeam(1.1, 2.0, 3.0, 40.0, 0.1).transfer(1j * 9.0).real
    np.testing.assert_allclose(ric.chessboard_inverse(ut) @ ut, np.eye(4),
                               atol=1e-11)


def test_real_form_product_eq_2_99_10_90():
    """BK Eq. (2.99)/(10.90): products of real forms are real forms of the
    complex product."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    b = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    np.testing.assert_allclose(ric.real_form(a) @ ric.real_form(b),
                               ric.real_form(a @ b), atol=1e-13)
    # damped spring of Eq. (2.96)-(2.97)
    k, c, lam = 3.0, 0.4, -0.2 + 1.7j
    u = lib.Spring(k, c=c).transfer(lam)
    den = (k - c * 0.2) ** 2 + (c * 1.7) ** 2
    assert u[0, 1].real == pytest.approx(-(k - c * 0.2) / den, rel=1e-12)
    assert u[0, 1].imag == pytest.approx(c * 1.7 / den, rel=1e-12)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
