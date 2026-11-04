# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of the linear transfer matrix method (solvcon.multibody.tmm.linear,
solvcon.multibody.tmm.library).

Book: Rui, Wang, Zhang, Transfer Matrix Method for Multibody Systems (Wiley
2018), Ch. 2, 3 and 15.2-15.10. Check types (docs/roadmap.md Sec. 4.2): AN
analytic, BK book example, XM cross-method, IV invariant.

Tolerance rules:

* BK: one unit in the last digit printed by the book (values are rounded).
* AN for eigenfrequencies: 1e-10 relative. The roots are refined to 1e-14;
  the characteristic determinant of a single element loses about
  eps * cosh(lambda l) (Sec. 10.9), which is below 1e-12 for the modes used.
* Orthogonality: 10 times the relative change of the mode shapes (mass and
  energy norms) caused by a perturbation of the eigenfrequency equal to the
  root tolerance 1e-14. This is the accuracy limit of the mode shapes; the
  high modes of stiff chains are very sensitive to the eigenfrequency.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest


from solvcon import multibody
from solvcon.multibody.tmm import library as lib
from solvcon.multibody.tmm import riccati as ric
from solvcon.multibody.tmm.linear import Chain, ModalModel, Mode

DATA = Path(__file__).parent / "data" / "multibody"
EPS = np.finfo(float).eps


def _load(name):
    with open(DATA / name) as fh:
        return json.load(fh)


CH2 = _load("tmm_ch2_examples.json")["examples"]
CH3 = _load("tmm_ch3_examples.json")["examples"]


def _assert_book(values, book_strings, label):
    """Compare with book values: one unit in the last printed digit."""
    assert len(values) >= len(book_strings), (f"{label}: too few values "
                                              f"{values}")
    for k, (v, txt) in enumerate(zip(values, book_strings)):
        dec = len(txt.split(".")[1]) if "." in txt else 0
        tol = 10.0 ** (-dec)
        assert abs(v
                   - float(txt)) <= tol, (f"{label}[{k}]: {v} vs book {txt} "
                                          f"(tol {tol})")


def _cantilever_roots(n):
    """Roots of cos(x) cosh(x) = -1 by bisection of cos(x) + 1/cosh(x)."""
    out = []
    for k in range(1, n + 1):
        a, b = max(0.5, (k - 0.5) * np.pi - 1.0), (k - 0.5) * np.pi + 1.0
        g = lambda x: np.cos(x) + 1.0 / np.cosh(x)  # noqa: E731
        for _ in range(200):
            c = 0.5 * (a + b)
            if np.sign(g(c)) == np.sign(g(a)):
                a = c
            else:
                b = c
        out.append(0.5 * (a + b))
    return np.array(out)


def _free_free_roots(n):
    """Nonzero roots of cos(x) cosh(x) = 1 by bisection of cos(x) -
    1/cosh(x)."""
    out = []
    for k in range(1, n + 1):
        a, b = (k + 0.5) * np.pi - 1.0, (k + 0.5) * np.pi + 1.0
        g = lambda x: np.cos(x) - 1.0 / np.cosh(x)  # noqa: E731
        for _ in range(200):
            c = 0.5 * (a + b)
            if np.sign(g(c)) == np.sign(g(a)):
                a = c
            else:
                b = c
        out.append(0.5 * (a + b))
    return np.array(out)


# ----------------------------------------------------------------------
# Book examples of Ch. 2 (BK)
# ----------------------------------------------------------------------
def test_example_2_1_mass_on_simply_supported_beam():
    """BK/AN Example 2.1: omega = sqrt(6 EI / (m l^3)) (Eq. j)."""
    ei, m, ln = 3.0, 2.0, 0.7
    ch = Chain([lib.EulerBernoulliBeam(ln, ei), lib.BeamPointMass(m),
                lib.EulerBernoulliBeam(ln, ei)],
               [0, 2], [0, 2])
    w = ch.natural_frequencies(10.0)
    assert len(w) == 1
    assert w[0] == pytest.approx(math.sqrt(6 * ei / (m * ln**3)), rel=1e-12)


def test_example_2_2_two_dof_chain_and_eigenvectors():
    """BK/AN Example 2.2 Eq. (f) and Fig. 2.19 eigenvector ratios."""
    k, m = 1.0, 1.0
    ch = Chain([lib.Spring(k), lib.LumpedMass(m), lib.Spring(2 * k),
                lib.LumpedMass(2 * m)],
               [0], [1])
    w = ch.natural_frequencies(3.0)
    exact = np.sqrt([(2 - math.sqrt(3)) * k / m, (2 + math.sqrt(3)) * k / m])
    np.testing.assert_allclose(w, exact, rtol=1e-12)
    ratios = [mode.states[3][0] / mode.states[1][0] for mode in ch.modes(w)]
    np.testing.assert_allclose(ratios,
                               [(1 + math.sqrt(3)) / 2,
                                -(math.sqrt(3) - 1) / 2], rtol=1e-10)


def _example_2_3_chain(zero_in, zero_out):
    d = CH2["2.3"]["data"]
    ei, m, ln = d["EI"], d["m"], d["l"]
    els = [lib.EulerBernoulliBeam(ln, ei), lib.BeamPointMass(m),
           lib.EulerBernoulliBeam(ln, 2 * ei), lib.BeamPointMass(3 * m),
           lib.EulerBernoulliBeam(ln, 2 * ei), lib.BeamPointMass(3 * m),
           lib.EulerBernoulliBeam(ln, 2 * ei), lib.BeamPointMass(m),
           lib.EulerBernoulliBeam(ln, ei)]
    return Chain(els, zero_in, zero_out)


@pytest.mark.parametrize("key, zin, zout", [
    ("free_free", [2, 3], [2, 3]),
    ("fixed_free", [0, 1], [2, 3]),
    ("simply_supported", [0, 2], [0, 2]),
])
def test_example_2_3_table_2_2(key, zin, zout):
    """BK Example 2.3, Table 2.2 (Eq. 2.43-2.45), p. 37."""
    w = _example_2_3_chain(zin, zout).natural_frequencies(7.0, n_grid=1000)
    _assert_book(w, CH2["2.3"][key], "Table 2.2 " + key)


def _example_2_6_chain():
    d = CH2["2.6"]["data"]
    m, ln = d["m"], d["l"]
    j = 5.0 / 12.0 * m * ln**2
    els = []
    for _ in range(4):
        els += [lib.PlanarElasticHinge(d["Kx"], d["Ky"], d["Kz"]),
                lib.PlanarRigidBody(m, j, (ln, 0.0), (ln / 2, 0.0))]
    return Chain(els, [0, 1, 2], [3, 4, 5])


def test_example_2_6_table_2_3():
    """BK Example 2.6, Table 2.3, p. 43."""
    w = _example_2_6_chain().natural_frequencies(3500.0)
    _assert_book(w, CH2["2.6"]["omega"], "Table 2.3")


def _example_2_7_beam():
    d = CH2["2.7"]["data"]
    ei = d["E"] * d["b"] * d["h"] ** 3 / 12.0
    mb = d["rho"] * d["b"] * d["h"]
    return ei, mb, d["l"]


def test_example_2_7_table_2_4():
    """BK Example 2.7, Table 2.4 (omega and lambda l), p. 45."""
    ei, mb, ln = _example_2_7_beam()
    w = Chain([lib.EulerBernoulliBeam(ln, ei, mb)], [0, 1],
              [2, 3]).natural_frequencies(8500.0)
    _assert_book(w, CH2["2.7"]["omega"], "Table 2.4 omega")
    _assert_book((mb * w**2 / ei) ** 0.25 * ln, CH2["2.7"]["lambda_l"],
                 "Table 2.4 lambda l")


def test_example_2_8_table_2_5_timoshenko():
    """BK Example 2.8, Table 2.5, p. 46 (rho_z = h/sqrt(12); printed 0.029 is a
    typo)."""
    ei, mb, ln = _example_2_7_beam()
    h = CH2["2.7"]["data"]["h"]
    beam = lib.TimoshenkoBeam(ln, ei, mb, CH2["2.8"]["data"]["GAs"],
                              h / math.sqrt(12.0))
    w = Chain([beam], [0, 1], [2, 3]).natural_frequencies(8200.0)
    _assert_book(w, CH2["2.8"]["omega"], "Table 2.5")


@pytest.mark.parametrize("n", [10, 100])
def test_example_2_9_table_2_9_wedge_beam(n):
    """BK Example 2.9, Table 2.9, p. 50: stepped wedge cantilever, EB and
    Timoshenko."""
    d = CH2["2.9"]["data"]
    els_eb, els_t = [], []
    for i in range(n):
        hh = d["root_height"] * (1.0 - (i + 0.5) / n)
        area = d["b"] * hh
        inertia = d["b"] * hh**3 / 12.0
        seg = d["l"] / n
        els_eb.append(lib.EulerBernoulliBeam(seg, d["E"] * inertia,
                                             d["rho"] * area))
        els_t.append(lib.TimoshenkoBeam(seg, d["E"] * inertia, d["rho"] * area,
                                        0.833 * d["G"] * area,
                                        math.sqrt(inertia / area)))
    w_eb = Chain(els_eb, [0, 1], [2, 3]).natural_frequencies(4800.0,
                                                             n_grid=600)
    w_t = Chain(els_t, [0, 1], [2, 3]).natural_frequencies(4800.0, n_grid=600)
    _assert_book(w_eb, CH2["2.9"][f"euler_n{n}"], f"Table 2.9 EB n={n}")
    _assert_book(w_t, CH2["2.9"][f"timoshenko_n{n}"],
                 f"Table 2.9 Timoshenko n={n}")


def _example_2_10_chain(m2=None, k1r=None):
    d = CH2["2.10"]["data"]
    m2 = d["m2"] if m2 is None else m2
    k1r = d["K1r"] if k1r is None else k1r
    b, h = d["b"], d["h"]
    j2 = m2 * (b**2 + 4 * h**2) / 12.0
    els = [lib.PlanarElasticHinge(np.inf, d["K1"], k1r),
           lib.PlanarRigidBody(m2, j2, (b / 2, h / 2), (0.0, h / 2)),
           lib.PlanarBeam(d["l3"], d["EI3"], d["m3"])]
    return Chain(els, [0, 1, 2], [3, 4, 5])


def test_example_2_10_tables_2_10_to_2_12():
    """BK Example 2.10, Tables 2.10-2.12, p. 52-53."""
    _assert_book(_example_2_10_chain().natural_frequencies(8500.0),
                 CH2["2.10"]["omega"], "Table 2.10")
    _assert_book(_example_2_10_chain(m2=7800.0).natural_frequencies(8500.0),
                 CH2["2.10"]["omega_m2_7800"], "Table 2.11")
    _assert_book(_example_2_10_chain(k1r=np.inf).natural_frequencies(8500.0),
                 CH2["2.10"]["omega_K1r_inf"], "Table 2.12")


def test_example_2_11_tables_2_13_2_14_torsion():
    """BK Example 2.11, Tables 2.13 and 2.14, p. 54-55."""
    d = CH2["2.11"]["data"]

    def chain(massless):
        els = []
        for i in range(4):
            jp = math.pi * d["r"][i] ** 4 / 2.0
            els.append(lib.Shaft(d["l"][i], d["G"] * jp,
                                 0.0 if massless else d["rho"] * jp))
            if i < 3:
                els.append(lib.Disk(d["J"][i]))
        return Chain(els, [1], [1])

    _assert_book(chain(False).natural_frequencies(17000.0, n_grid=4000),
                 CH2["2.11"]["omega"], "Table 2.13")
    _assert_book(chain(True).natural_frequencies(17000.0),
                 CH2["2.11"]["omega_massless"], "Table 2.14")


def test_example_2_12_forced_two_dof_closed_form():
    """BK/AN Example 2.12 Eq. (e)-(f): steady-state amplitudes of the forced
    2-DOF chain."""
    k, m, f = 3.0, 1.5, 0.7
    for big_omega in (0.4, 1.3, 2.9):
        ch = Chain([lib.Spring(k), lib.LumpedMass(m), lib.Spring(2 * k),
                    lib.LumpedMass(2 * m, force=f)], [0], [1])
        z = ch.steady_state(big_omega).real
        r = m * big_omega**2 / k
        den = r**2 - 4 * r + 1
        assert z[0][1] == pytest.approx(-f / den, rel=1e-12)
        assert z[1][0] == pytest.approx(f / k / den, rel=1e-12)
        assert z[3][0] == pytest.approx(f / (2 * k) * (3 - r) / den, rel=1e-12)
        assert z[4][0] == pytest.approx(f / (2 * k) * (3 - r) / den, rel=1e-12)


def test_example_2_13_static_beam_against_finite_elements():
    """BK/XM Example 2.13: static beam with load, spring and moment vs exact
    Hermite FE.

    Cubic Hermite elements with consistent loads give exact nodal displacements
    of a massless Euler-Bernoulli beam under uniform load.
    """
    ln, ei, p = 1.0, 200.0, 10.0
    k = 2 * ei / ln**3
    moment = -p * ln**2  # M_out = M_in + P l^2 in U_2 means M_ext = -P l^2
    ch = Chain([lib.EulerBernoulliBeam(ln, ei, p=-p),
                lib.BeamPointMass(k=k, moment=moment),
                lib.EulerBernoulliBeam(2 * ln, ei)], [0, 2], [0, 1])
    z = ch.static_response()

    nodes = [0.0, ln, 3 * ln]
    kg = np.zeros((6, 6))
    fg = np.zeros(6)
    for e, (xa, xb) in enumerate(zip(nodes[:-1], nodes[1:])):
        le = xb - xa
        ke = ei / le**3 * np.array([[12, 6 * le, -12, 6 * le],
                                    [6 * le, 4 * le**2, -6 * le, 2 * le**2],
                                    [-12, -6 * le, 12, -6 * le],
                                    [6 * le, 2 * le**2, -6 * le, 4 * le**2]])
        load = -p if e == 0 else 0.0
        fe = load * np.array([le / 2, le**2 / 12, le / 2, -le**2 / 12])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        kg[np.ix_(idx, idx)] += ke
        fg[idx] += fe
    kg[2, 2] += k
    fg[3] += moment
    free = [1, 2, 3]  # y0 = 0 (support), y2 = theta2 = 0 (clamped)
    u = np.zeros(6)
    u[free] = np.linalg.solve(kg[np.ix_(free, free)], fg[free])
    # TMM states: z[0] input of beam 1, z[1] at the spring (x = l), z[3] output
    # end
    assert z[0][1] == pytest.approx(u[1], rel=1e-12)
    assert z[1][0] == pytest.approx(u[2], rel=1e-12)
    assert z[1][1] == pytest.approx(u[3], rel=1e-12)
    assert abs(z[3][0]) < 1e-14 and abs(z[3][1]) < 1e-14
    # global equilibrium: support reactions balance the applied load
    reaction_left = z[0][3]
    reaction_right = -z[3][3]
    spring_force = -k * z[1][0]
    assert reaction_left + reaction_right + spring_force - p * ln == (
        pytest.approx(0.0, abs=1e-11))


def test_example_2_14_table_2_15_damped_torsion():
    """BK Example 2.14, Table 2.15, p. 77 (complex extended TMM, Eq.
    2.141-2.151)."""
    d = CH2["2.14"]["data"]
    m5 = complex(d["M5"][0], -d["M5"][1])  # M^r = M^c, M^i = -M^s (Eq. b)
    m7 = complex(d["M7"][0], -d["M7"][1])
    els = [lib.Disk(2.0), lib.RotarySpring(8.0 / 3.0),
           lib.Disk(1.0, c_ground=d["r3"]), lib.RotarySpring(0.0, c=d["C4"]),
           lib.Disk(1.1, torque=m5), lib.RotarySpring(d["K6"], g=d["g6"]),
           lib.Disk(1.5, torque=m7)]
    ch = Chain(els, [1], [1])
    z = ch.steady_state(math.sqrt(d["Omega2"]))
    theta = [z[i][0] for i in (0, 2, 4, 6)]
    _assert_book([t.real for t in theta], CH2["2.14"]["theta_real"],
                 "Table 2.15 real")
    _assert_book([t.imag for t in theta], CH2["2.14"]["theta_imag"],
                 "Table 2.15 imag")
    # XM: the real form of Eq. (2.145) gives the same response (Eq. 2.95, 2.99)
    s = 1j * math.sqrt(d["Omega2"])
    big = np.eye(5)
    for el in els:
        big = ric.real_form(el.transfer(s), el.forcing(s)) @ big
    a = big[np.ix_([1, 3], [0, 2])]
    sol = np.linalg.solve(a, -big[[1, 3], 4])
    assert sol[0] == pytest.approx(theta[0].real, rel=1e-12)
    assert sol[1] == pytest.approx(theta[0].imag, rel=1e-12)


# ----------------------------------------------------------------------
# Analytic checks (AN)
# ----------------------------------------------------------------------
def test_uniform_spring_mass_chains_closed_form():
    """AN: n masses, n+1 springs fixed-fixed and n masses, n springs
    fixed-free."""
    k, m, n = 40.0, 0.5, 12
    els = []
    for _ in range(n):
        els += [lib.Spring(k), lib.LumpedMass(m)]
    w_ff = Chain(els + [lib.Spring(k)], [0],
                 [0]).natural_frequencies(2 * math.sqrt(k / m) * 1.01)
    kk = np.arange(1, n + 1)
    exact_ff = 2 * math.sqrt(k / m) * np.sin(kk * np.pi / (2 * (n + 1)))
    np.testing.assert_allclose(w_ff, exact_ff, rtol=1e-10)
    w_fr = Chain(els, [0],
                 [1]).natural_frequencies(2 * math.sqrt(k / m) * 1.01)
    exact_fr = 2 * math.sqrt(k / m) * np.sin((2 * kk - 1) * np.pi
                                             / (2 * (2 * n + 1)))
    np.testing.assert_allclose(w_fr, exact_fr, rtol=1e-10)


@pytest.mark.parametrize("key, zin, zout", [
    ("fixed_free", [0, 1], [2, 3]),
    ("fixed_fixed", [0, 1], [0, 1]),
    ("fixed_pinned", [0, 1], [0, 2]),
    ("pinned_pinned", [0, 2], [0, 2]),
    ("pinned_free", [0, 2], [2, 3]),
    ("free_free", [2, 3], [2, 3]),
])
def test_table_2_6_euler_bernoulli_boundary_conditions(key, zin, zout):
    """BK Table 2.6, p. 47: lambda_k l of a uniform beam for six boundary
    conditions."""
    ei, mb, ln = 2.0, 3.0, 1.5
    w = Chain([lib.EulerBernoulliBeam(ln, ei, mb)], zin,
              zout).natural_frequencies(16.0**2 / ln**2 * math.sqrt(ei / mb))
    lam_l = (mb * w**2 / ei) ** 0.25 * ln
    _assert_book(lam_l, CH2["table_2.6"][key], "Table 2.6 " + key)


def test_cantilever_and_free_free_beam_high_precision():
    """AN: beta_n L = 1.8751..., 4.6941... (cantilever) and 4.7300...
    (free-free) to 1e-10."""
    ei, mb, ln = 5.0, 0.4, 2.0
    scale = math.sqrt(ei / mb) / ln**2
    w_c = Chain([lib.EulerBernoulliBeam(ln, ei, mb)], [0, 1],
                [2, 3]).natural_frequencies(12.0**2 * scale)
    np.testing.assert_allclose(np.sqrt(w_c / scale), _cantilever_roots(4),
                               rtol=1e-10)
    w_f = Chain([lib.EulerBernoulliBeam(ln, ei, mb)], [2, 3],
                [2, 3]).natural_frequencies(12.0**2 * scale)
    assert list(w_f[:2]) == [0.0, 0.0]
    np.testing.assert_allclose(np.sqrt(w_f[2:] / scale), _free_free_roots(3),
                               rtol=1e-10)


def test_beam_split_into_segments_equals_single_element():
    """IV: U(l1 + l2) = U(l2) U(l1) (Eq. 2.34) for the beam, rod and shaft."""
    s = 1j * 37.0
    for make in (lambda x: lib.EulerBernoulliBeam(x, 3.0, 1.2),
                 lambda x: lib.TimoshenkoBeam(x, 3.0, 1.2, 50.0, 0.05),
                 lambda x: lib.Rod(x, 7.0, 0.3),
                 lambda x: lib.Shaft(x, 2.0, 0.1)):
        u_full = make(1.0).transfer(s)
        u_split = make(0.6).transfer(s) @ make(0.4).transfer(s)
        np.testing.assert_allclose(u_split, u_full, rtol=1e-12,
                                   atol=1e-12 * abs(u_full).max())


@pytest.mark.parametrize("zin, zout, k_values", [
    # fixed-free (Table 2.7 / 2.8 row 2)
    # fixed-fixed (row 1)
    # free-free (row 3)
    ([0], [1], lambda k: (k - 0.5) * np.pi), ([0], [0], lambda k: k * np.pi),
    ([1], [1], lambda k: (k - 1) * np.pi),
])
def test_tables_2_7_2_8_rod_and_shaft(zin, zout, k_values):
    """AN Tables 2.7 and 2.8, p. 48: rod beta_k l and shaft gamma_k l.

    State [X, Q_x] of the rod: X = 0 is component 0. For the shaft [Theta, M].
    """
    ln, ea, m = 1.3, 9.0, 0.8
    rod = Chain([lib.Rod(ln, ea, m)], zin,
                zout).natural_frequencies(14.0 / ln * math.sqrt(ea / m))
    beta_l = rod * math.sqrt(m / ea) * ln
    kk = np.arange(1, len(beta_l) + 1)
    np.testing.assert_allclose(beta_l, k_values(kk), rtol=1e-10, atol=1e-12)
    gj, rho_j = 4.0, 0.3
    shaft = Chain([lib.Shaft(ln, gj, rho_j)], zin, zout).natural_frequencies(
        14.0 / ln * math.sqrt(gj / rho_j))
    np.testing.assert_allclose(shaft * math.sqrt(rho_j / gj) * ln,
                               k_values(kk), rtol=1e-10, atol=1e-12)


def test_static_uniform_load_cantilever_and_rod():
    """AN: cantilever tip deflection p L^4/(8 EI) and rod tip displacement p
    L^2/(2 EA)."""
    ln, ei, p = 2.0, 30.0, 4.0
    z = Chain([lib.EulerBernoulliBeam(ln, ei, m=1.0, p=p)], [0, 1],
              [2, 3]).static_response()
    assert z[-1][0] == pytest.approx(p * ln**4 / (8 * ei), rel=1e-12)
    assert z[-1][1] == pytest.approx(p * ln**3 / (6 * ei), rel=1e-12)
    ea = 50.0
    zr = Chain([lib.Rod(ln, ea, m=1.0, p=p)], [0], [1]).static_response()
    assert zr[-1][0] == pytest.approx(p * ln**2 / (2 * ea), rel=1e-12)


def test_euler_buckling_of_axially_loaded_column():
    """AN: pinned-pinned column with compressive force P has det = 0 at P =
    pi^2 EI/L^2."""
    ei, ln = 7.0, 2.5
    p_cr = math.pi**2 * ei / ln**2

    def det(p):
        ch = Chain([lib.AxialLoadBeam(ln, ei, p)], [0, 2], [0, 2])
        return ch.characteristic(0.0)

    a, b = 0.9 * p_cr, 1.1 * p_cr
    assert np.sign(det(a)) != np.sign(det(b))
    for _ in range(80):
        c = 0.5 * (a + b)
        if np.sign(det(c)) == np.sign(det(a)):
            a = c
        else:
            b = c
    assert 0.5 * (a + b) == pytest.approx(p_cr, rel=1e-10)
    # P = 0 gives the massless beam of Eq. (15.22)
    np.testing.assert_allclose(lib.AxialLoadBeam(ln, ei, 0.0).transfer(0.0),
                               lib.EulerBernoulliBeam(ln, ei).transfer(0.0),
                               atol=1e-13)


def test_massless_beam_on_foundation_matches_eq_15_25():
    """BK Eq. (15.25): e-functions of the massless beam on a Winkler
    foundation."""
    ei, k, x = 3.0, 48.0, 0.9
    lam = k / ei
    beta = (lam / 4.0) ** 0.25
    bx = beta * x
    e1 = np.cosh(bx) * np.cos(bx)
    e2 = (np.cosh(bx) * np.sin(bx) + np.sinh(bx) * np.cos(bx)) / (2 * beta)
    e3 = np.sinh(bx) * np.sin(bx) / (2 * beta**2)
    e4 = (np.cosh(bx) * np.sin(bx) - np.sinh(bx) * np.cos(bx)) / (4 * beta**3)
    book = np.array([[e1, e2, e3 / ei, e4 / ei],
                     [-lam * e4, e1, e2 / ei, e3 / ei],
                     [-lam * ei * e3, -lam * ei * e4, e1, e2],
                     [-lam * ei * e2, -lam * ei * e3, -lam * e4, e1]])
    u = lib.EulerBernoulliBeam(x, ei, k_found=k).transfer(0.0)
    np.testing.assert_allclose(u, book, rtol=1e-12, atol=1e-14)


def test_massless_limits_eq_2_31_and_15_31_15_37():
    """BK Eq. (2.31)/(15.22), (15.31), (15.37): massless beam, shaft, rod."""
    ln, ei = 0.8, 5.0
    u = lib.EulerBernoulliBeam(ln, ei).transfer(1j * 3.0)
    book = np.array([[1, ln, ln**2 / (2 * ei), ln**3 / (6 * ei)],
                     [0, 1, ln / ei, ln**2 / (2 * ei)], [0, 0, 1, ln],
                     [0, 0, 0, 1]])
    np.testing.assert_allclose(u, book, atol=1e-15)
    np.testing.assert_allclose(lib.Shaft(ln, ei).transfer(2j),
                               [[1, ln / ei], [0, 1]], atol=1e-15)
    np.testing.assert_allclose(lib.Rod(ln, ei).transfer(2j),
                               [[1, -ln / ei], [0, 1]], atol=1e-15)
    shear = lib.MasslessShearBeam(ln, ei, 20.0).transfer(0.0)
    assert shear[0, 3] == pytest.approx(ln**3 / (6 * ei) - ln / 20.0)


def test_timoshenko_simply_supported_dispersion():
    """AN: pinned-pinned Timoshenko: det(A(omega) - i k_n I) = 0 with k_n = n
    pi / L."""
    ln, ei, m, gas, rz = 1.0, 2.0e3, 3.0, 4.0e4, 0.06
    beam = lib.TimoshenkoBeam(ln, ei, m, gas, rz)
    w = Chain([beam], [0, 2], [0, 2]).natural_frequencies(4000.0)
    assert len(w) >= 4
    for n, wn in enumerate(w[:4], start=1):
        a = beam.system_matrix(1j * wn)
        ev = np.linalg.eigvals(a)
        kn = n * math.pi / ln
        assert np.min(np.abs(ev - 1j * kn)) < 1e-9 * kn


def test_timoshenko_three_derivations_agree():
    """XM Sec. 10.3-10.4: Cayley-Hamilton form, B(l) B(0)^{-1} (Eq. 10.60),
    expm (Eq. 10.69)."""
    # thick beam: B(0) well conditioned
    beam = lib.TimoshenkoBeam(0.3, 50.0, 8.0, 400.0, 0.08)
    for w in (40.0, 300.0, 900.0):
        s = 1j * w
        u_ch = beam.transfer_at(0.3, s)
        u_b = beam.transfer_fundamental(0.3, s)
        # dimensionless state (Sec. 10.6.4) before the exponential
        x, ei = 0.3, 50.0
        d = np.diag([1.0, 1.0 / x, ei / x**2, ei / x**3])
        d_inv = np.diag([1.0, x, x**2 / ei, x**3 / ei])
        u_e = (d @ ric.transfer_from_first_order(
            d_inv @ beam.system_matrix(s) @ d, x) @ d_inv)
        scale = np.abs(u_ch).max()
        np.testing.assert_allclose(u_b, u_ch, atol=1e-9 * scale)
        np.testing.assert_allclose(u_e, u_ch, atol=1e-11 * scale)
    # EB limit: GA_s -> inf, rho_z -> 0
    eb = lib.EulerBernoulliBeam(0.7, 3.0, 2.0).transfer(1j * 20.0)
    tb = lib.TimoshenkoBeam(0.7, 3.0, 2.0, 1e14, 1e-9).transfer(1j * 20.0)
    np.testing.assert_allclose(tb, eb, rtol=1e-8, atol=1e-10)


def test_rigid_bodies_spatial_reduces_to_planar():
    """BK Eq. (10.19) -> (10.10): delete rows/columns 3, 4, 5, 7, 8, 12 (text
    after Eq. 10.19)."""
    m, b, cc = 2.0, (0.4, 0.15, 0.0), (0.1, 0.07, 0.0)
    jz = 0.3
    j_i = np.diag([0.2, 0.25, jz])
    s = 1j * 11.0
    u3 = lib.SpatialRigidBody(m, j_i, b, cc).transfer(s)
    keep = [0, 1, 5, 8, 9, 10]
    u2 = lib.PlanarRigidBody(m, jz, b[:2], cc[:2]).transfer(s)
    np.testing.assert_allclose(u3[np.ix_(keep, keep)], u2, atol=1e-14)
    # Eq. (15.13): transverse rigid body = planar with X, Q_x removed
    ut = lib.TransverseRigidBody(m, jz, 0.4, 0.1).transfer(s)
    up = lib.PlanarRigidBody(m, jz, (0.4, 0.0), (0.1, 0.0)).transfer(s)
    np.testing.assert_allclose(ut, up[np.ix_([1, 2, 3, 5], [1, 2, 3, 5])],
                               atol=1e-14)


def test_spatial_beam_blocks_eq_15_39_to_15_41():
    """BK Eq. (15.39)-(15.41): the combined beams are block compositions of
    rod, shaft, beam."""
    ln, s = 0.9, 1j * 25.0
    rod = lib.Rod(ln, 80.0, 1.1).transfer(s)
    beam_y = lib.EulerBernoulliBeam(ln, 4.0, 1.1).transfer(s)
    beam_z = lib.EulerBernoulliBeam(ln, 6.0, 1.1).transfer(s)
    shaft = lib.Shaft(ln, 3.0, 0.02).transfer(s)
    u = lib.SpatialBeam(ln, 80.0, 6.0, 4.0, 3.0, 1.1, 0.02).transfer(s)
    np.testing.assert_allclose(u[np.ix_([0, 9], [0, 9])], rod)
    np.testing.assert_allclose(u[np.ix_([3, 6], [3, 6])], shaft)
    np.testing.assert_allclose(u[np.ix_([1, 5, 8, 10], [1, 5, 8, 10])], beam_y)
    flip = np.diag([1, -1, -1, 1])
    np.testing.assert_allclose(u[np.ix_([2, 4, 7, 11], [2, 4, 7, 11])],
                               flip @ beam_z @ flip)
    # Eq. (15.41) entry u_3,5 = -T(lambda_z x)/lambda_z
    lam = (1.1 * 25.0**2 / 6.0) ** 0.25
    t = lib.krylov(lam * ln)[1]
    assert u[2, 4] == pytest.approx(-t / lam, rel=1e-12)
    up = lib.PlanarBeam(ln, 4.0, 1.1, ea=80.0).transfer(s)
    np.testing.assert_allclose(up[np.ix_([0, 4], [0, 4])], rod)
    np.testing.assert_allclose(up[np.ix_([1, 2, 3, 5], [1, 2, 3, 5])], beam_y)


def test_spatial_chain_contains_planar_frequencies():
    """IV: the spatial form of the Example 2.6 chain has the planar frequencies
    of Table 2.3."""
    d = CH2["2.6"]["data"]
    m, ln = d["m"], d["l"]
    jz = 5.0 / 12.0 * m * ln**2
    j_i = np.diag([m * ln**2 / 6, m * ln**2 / 3, jz])
    els = []
    for _ in range(4):
        els += [lib.SpatialElasticHinge([d["Kx"], d["Ky"], d["Ky"]],
                                        [d["Kz"], d["Kz"], d["Kz"]]),
                lib.SpatialRigidBody(m, j_i, (ln, 0, 0), (ln / 2, 0, 0))]
    w3 = Chain(els, list(range(6)),
               list(range(6, 12))).natural_frequencies(3000.0)
    w2 = _example_2_6_chain().natural_frequencies(3000.0)
    for w in w2:
        assert np.min(np.abs(w3 - w)) < 1e-9 * w


# ----------------------------------------------------------------------
# Root search rule (Sec. 2.5, 10.7)
# ----------------------------------------------------------------------
def test_close_and_double_roots_are_found():
    """IV: weakly coupled identical oscillators (close pair) and a planar mass
    (double roots)."""
    k, m, kc = 100.0, 1.0, 1e-5
    ch = Chain([lib.Spring(k), lib.LumpedMass(m), lib.Spring(kc),
                lib.LumpedMass(m), lib.Spring(k)], [0], [0])
    w = ch.natural_frequencies(20.0, n_grid=50)
    exact = np.sqrt([k / m, (k + 2 * kc) / m])
    np.testing.assert_allclose(w, exact, rtol=1e-12)
    ch2 = Chain([lib.Spring([k, k], dims=2), lib.LumpedMass(m, dims=2)],
                [0, 1], [2, 3])
    w2 = ch2.natural_frequencies(20.0, n_grid=50)
    np.testing.assert_allclose(w2, [10.0, 10.0], rtol=1e-8)
    modes = ch2.modes(w2)
    shapes = np.array([md.states[1][:2] / np.linalg.norm(md.states[1][:2])
                       for md in modes])
    assert abs(np.linalg.det(shapes)) > 0.5


def test_double_roots_of_symmetric_grid_plate_counted_once():
    """IV/XM: square grid plate (Sec. 4.7) has exact double roots (x-y
    symmetry).

    Defect regression: rounding gave a sign change next to a double root, so
    the bracket root and the local-minimum root (multiplicity 2) were both
    counted (14 roots instead of 10 for the 2 x 2 grid). The plain and the
    Riccati searches must return the global eigenvalues with their
    multiplicity (1e-8, root tolerance of a double root found as a minimum).
    """
    from solvcon.multibody.tmm.continua import GridPlate

    with open(DATA / "tmm_ch4_examples.json") as fh:
        d = json.load(fh)["examples"]["4.6"]["data"]
    grid = GridPlate(d["a"], d["b"], d["h"], d["E"], d["G"], d["mu"], d["rho"],
                     2, 2)
    ref = grid.global_frequencies()
    w_max = ref[7] * 1.02
    exact = ref[ref < w_max]
    ch = grid.chain()
    for w in (ch.natural_frequencies(w_max, n_grid=500),
              ch.natural_frequencies_riccati(w_max, n_grid=500)):
        assert len(w) == len(exact)
        np.testing.assert_allclose(w, exact, rtol=1e-8)


def test_improved_secant_and_scaled_determinant():
    """BK Eq. (10.96)-(10.99): A(omega) keeps the sign of D, R =
    (A_b/A_a)^k."""
    rng = np.random.default_rng(3)
    mat = rng.normal(size=(6, 6)) * 1e40
    a10 = ric.scaled_determinant(mat, 10.0)
    sign, logabs = np.linalg.slogdet(mat)
    assert np.sign(a10) == sign
    assert math.log(abs(a10)) * 10 == pytest.approx(logabs, rel=1e-12)

    def f(x):
        return (np.sign(math.cos(x)), math.log(abs(math.cos(x))) + 500.0)

    wc = ric.improved_secant_step(1.0, 2.0, f(1.0), f(2.0))
    assert 1.0 < wc < 2.0


# ----------------------------------------------------------------------
# Damped eigenvalues (Sec. 2.7)
# ----------------------------------------------------------------------
def test_damped_eigenvalues_sdof_eq_2_103():
    """AN Eq. (2.103): lambda = -C/(2m) + i sqrt(K/m - (C/2m)^2)."""
    m, k, c = 2.0, 50.0, 3.0
    lam = Chain([lib.Spring(k, c=c), lib.LumpedMass(m)], [0],
                [1]).damped_eigenvalues(10.0)
    exact = -c / (2 * m) + 1j * math.sqrt(k / m - (c / (2 * m)) ** 2)
    assert abs(lam[0] - exact) < 1e-12 * abs(exact)


def test_damped_eigenvalues_chain_vs_state_space():
    """XM: damped 3-DOF chain of Fig. 3.6 vs eigenvalues of the first-order
    state matrix."""
    m, k, c = 2.0, 50.0, 3.0
    els = [lib.Spring(2 * k, c=c), lib.LumpedMass(m), lib.Spring(k),
           lib.LumpedMass(m), lib.Spring(k), lib.LumpedMass(m),
           lib.Spring(2 * k)]
    lam = Chain(els, [0], [0]).damped_eigenvalues(20.0)
    mm = m * np.eye(3)
    kk = np.array([[3 * k, -k, 0], [-k, 2 * k, -k], [0, -k, 3 * k]])
    cc = np.zeros((3, 3))
    cc[0, 0] = c
    a = np.block([[np.zeros((3, 3)), np.eye(3)],
                  [-np.linalg.solve(mm, kk), -np.linalg.solve(mm, cc)]])
    ev = sorted([e for e in np.linalg.eigvals(a) if e.imag > 0],
                key=lambda z: z.imag)
    np.testing.assert_allclose(lam, ev, rtol=1e-11)


def test_damped_eigenvalues_kelvin_voigt_beam():
    """AN: pinned-pinned beam with internal damping, m s^2 + C' k^4 s + EI k^4
    = 0."""
    ei, mb, ln, ci = 100.0, 2.0, 3.0, 0.8
    lam = Chain([lib.EulerBernoulliBeam(ln, ei, mb, c_internal=ci)], [0, 2],
                [0, 2]).damped_eigenvalues(200.0)
    k4 = (np.arange(1, len(lam) + 1) * np.pi / ln) ** 4
    exact = (-ci * k4
             + np.sqrt((ci * k4) ** 2 - 4 * mb * ei * k4 + 0j)) / (2 * mb)
    assert len(lam) == 5
    np.testing.assert_allclose(lam, exact, rtol=1e-10)


# ----------------------------------------------------------------------
# Augmented eigenvectors and system response (Ch. 3)
# ----------------------------------------------------------------------
def _orthogonality_errors(chain, omegas):
    modes = chain.modes([w for w in omegas if w > 0])
    mm = ModalModel(chain, modes)
    mmat = mm.matrix("mass")
    kmat = mm.matrix("stiffness")
    dm = np.diag(mmat)
    off_m = np.max(np.abs(mmat - np.diag(dm)) / np.sqrt(np.outer(dm, dm)))
    kd = mm.omegas**2 * dm
    off_k = np.max(np.abs(kmat - np.diag(kd)) / np.sqrt(np.outer(kd, kd)))
    diag_k = np.max(np.abs(np.diag(kmat) - kd) / kd)
    return off_m, off_k, diag_k


def _ortho_tol(chain, omegas):
    """10 x mode shape change for a relative frequency perturbation of 1e-14
    (root tolerance)."""
    worst = 0.0
    for w in omegas:
        if w <= 0:
            continue
        a = chain.modes([w])[0]
        b = chain.modes([w * (1.0 + 1e-14)])[0]
        diff = Mode(chain, w, a.states - b.states)
        mm = ModalModel(chain, [a])
        for kind in ("mass", "stiffness"):
            ref = mm.inner(a, a, kind)
            worst = max(worst,
                        math.sqrt(abs(mm.inner(diff, diff, kind)) / ref))
    return 10.0 * worst + 1e-13


def test_orthogonality_discrete_chain_example_3_4():
    """IV Example 3.4: augmented eigenvectors of the Fig. 2.22 chain (Eq. 3.48,
    3.76-3.78)."""
    ch = _example_2_6_chain()
    w = ch.natural_frequencies(3500.0)
    errs = _orthogonality_errors(ch, w)
    assert max(errs) < _ortho_tol(ch, w), errs


def test_orthogonality_hybrid_rigid_flexible_example_3_7():
    """IV Example 3.7: elastic hinge + rigid body + beam (Example 2.10), Eq.
    (3.105)."""
    ch = _example_2_10_chain()
    w = ch.natural_frequencies(3000.0)
    errs = _orthogonality_errors(ch, w)
    assert max(errs) < _ortho_tol(ch, w), errs


def test_orthogonality_timoshenko_beam_example_3_6():
    """IV Example 3.6: Timoshenko cantilever (m Y^2 + m rho^2 Theta^2, M^2/EI +
    Q^2/GA_s)."""
    ei, mb, ln = _example_2_7_beam()
    beam = lib.TimoshenkoBeam(ln, ei, mb, 6.664e6, 0.01 / math.sqrt(12.0))
    ch = Chain([beam], [0, 1], [2, 3])
    w = ch.natural_frequencies(4500.0)
    errs = _orthogonality_errors(ch, w)
    assert max(errs) < _ortho_tol(ch, w), errs


def _example_3_9_chain():
    d = CH3["3.9"]["data"]
    return Chain([lib.BeamHinge(d["K"]), lib.BeamPointMass(d["m"]),
                  lib.EulerBernoulliBeam(d["l"], d["EI"], d["m_bar"])], [0, 1],
                 [2, 3])


def test_example_3_9_table_3_2_and_eq_3_115():
    """BK Example 3.9 Table 3.2 (p. 109) and AN Eq. (3.115) roots."""
    d = CH3["3.9"]["data"]
    ch = _example_3_9_chain()
    w = ch.natural_frequencies(100.0)
    _assert_book(w, CH3["3.9"]["omega_tmm"], "Table 3.2")
    k, m, ln, mb, ei = d["K"], d["m"], d["l"], d["m_bar"], d["EI"]

    def eq_3_115(lam):
        return ((lam**4 - k * mb * ln**4 / (m * ei))
                * (1 + np.cos(lam) * np.cosh(lam))
                + mb * ln * lam**3 / m * (np.cosh(lam) * np.sin(lam)
                                          + np.sinh(lam) * np.cos(lam)))

    lam_tmm = ln * (w * math.sqrt(mb / ei)) ** 0.5
    for lam in lam_tmm[:6]:
        a, b = lam * (1 - 1e-6), lam * (1 + 1e-6)
        assert np.sign(eq_3_115(a)) != np.sign(eq_3_115(b))
        for _ in range(100):
            c = 0.5 * (a + b)
            if np.sign(eq_3_115(c)) == np.sign(eq_3_115(a)):
                a = c
            else:
                b = c
        assert 0.5 * (a + b) == pytest.approx(lam, rel=1e-11)


def test_example_3_9_transient_modal_vs_analytic_series():
    """XM Example 3.9: free end y(l, t) by MSTMM modes (Eq. 3.133-3.134) vs Eq.
    (3.113)-(3.114).

    The analytic series uses the eigenfunctions of Eq. (3.114); alpha_i and
    beta_i are computed from the free-end conditions, because the printed
    expressions do not satisfy Y''(l) = 0 as extracted.
    """
    d = CH3["3.9"]["data"]
    m, ln, mb, ei, y0 = d["m"], d["l"], d["m_bar"], d["EI"], d["y0"]
    ch = _example_3_9_chain()
    w = ch.natural_frequencies(100.0)
    mm = ModalModel(ch, ch.modes(w))
    t = np.linspace(0.0, 10.0, 201)
    v0 = {1: [y0, 0, 0, 0], 2: [y0, 0, 0, 0]}
    q = mm.response(t, v0=v0)
    y_tmm = mm.displacement(q, 2, 0, x=ln)

    # analytic method, Eq. (3.113)-(3.114) with the same lambda_i
    xg, wg = np.polynomial.legendre.leggauss(80)
    xg = 0.5 * ln * (xg + 1)
    wg = 0.5 * ln * wg
    y_an = np.zeros_like(t)
    for wi in w:
        li = ln * (wi * math.sqrt(mb / ei)) ** 0.5
        # alpha_i, beta_i from the free-end conditions Y''(l) = Y'''(l) = 0
        # (Eq. 3.111); the mass condition holds through Eq. (3.115).
        cs, ch_, sn, sh = (math.cos(li), math.cosh(li), math.sin(li),
                           math.sinh(li))
        alpha, beta = np.linalg.solve([[-cs, ch_], [sn, sh]],
                                      [sn + sh, cs + ch_])

        def y_i(x, li=li, alpha=alpha, beta=beta):
            u = li * x / ln
            return (np.sin(u) - np.sinh(u) + alpha * np.cos(u)
                    + beta * np.cosh(u))

        num = y0 * (np.sum(wg * y_i(xg)) + m / mb * y_i(0.0))
        den2 = np.sum(wg * y_i(xg) ** 2) + m / mb * y_i(0.0) ** 2
        y_an += y_i(ln) * num / den2 * np.cos(wi * t)
    # both evaluations cancel terms of size cosh(lambda_max l) in the mode
    # shapes
    lam_max = ln * (w.max() * math.sqrt(mb / ei)) ** 0.5
    tol = 10.0 * EPS * math.cosh(lam_max) * np.max(np.abs(y_an))
    np.testing.assert_allclose(y_tmm, y_an, atol=tol)
    assert abs(y_tmm[0] - y0) < 1e-3  # 8 modes reproduce the initial offset


def _rk4(mm, kk, f_of_t, x0, v0, t_end, dt):
    n = len(x0)
    minv = np.linalg.inv(mm)

    def rhs(t, y):
        return np.concatenate([y[n:], minv @ (f_of_t(t) - kk @ y[:n])])

    steps = int(round(t_end / dt))
    y = np.concatenate([x0, v0]).astype(float)
    out = [y[:n].copy()]
    t = 0.0
    for _ in range(steps):
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt / 2 * k1)
        k3 = rhs(t + dt / 2, y + dt / 2 * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        out.append(y[:n].copy())
    return np.array(out)


def _example_2_6_global_matrices():
    """Global M and K of the Fig. 2.22 chain assembled from the energies
    (independent of TMM)."""
    d = CH2["2.6"]["data"]
    m, ln = d["m"], d["l"]
    c1, c2 = ln / 2, 0.0
    jc = m * ln**2 / 6.0
    mb = np.array([[m, 0, -m * c2], [0, m, m * c1],
                   [-m * c2, m * c1, jc + m * (c1**2 + c2**2)]])
    mg = np.kron(np.eye(4), mb)
    kd = np.diag([d["Kx"], d["Ky"], d["Kz"]])
    kg = np.zeros((12, 12))
    for i in range(4):
        b = np.zeros((3, 12))
        b[:, 3 * i:3 * i + 3] = np.eye(3)
        if i > 0:
            # relative motion of the input of body i w.r.t. the output (l, 0)
            # of body i-1
            b[:,
              3 * i - 3:3 * i] = -np.array([[1, 0, 0], [0, 1, ln], [0, 0, 1]])
        kg += b.T @ kd @ b
    return mg, kg


def test_example_2_6_frequencies_vs_global_eigenproblem():
    """XM: TMM eigenfrequencies of Example 2.6 vs eig(M^-1 K) of the assembled
    chain."""
    mg, kg = _example_2_6_global_matrices()
    ev = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(mg, kg)).real))
    np.testing.assert_allclose(
        _example_2_6_chain().natural_frequencies(3500.0), ev, rtol=1e-10)


def test_transient_modal_superposition_vs_rk4():
    """XM Sec. 3.6: modal superposition (Eq. 3.94-3.100) vs RK4 integration of
    M q'' + K q = f.

    Initial velocity of body 2 and a ramp-and-hold force on body 4 of the
    Example 2.6 chain. The tolerance is the RK4 error estimate from step
    halving (Richardson, factor 16/15) plus rounding.
    """
    mg, kg = _example_2_6_global_matrices()
    ch = _example_2_6_chain()
    mm = ModalModel(ch, ch.modes(ch.natural_frequencies(3500.0)))
    t_end, t_ramp = 0.05, 0.01
    g = lambda t: min(t / t_ramp, 1.0)  # noqa: E731
    load = np.array([3.0, -2.0, 0.05])
    vel = np.array([0.0, -1.0, 0.5])
    t = np.linspace(0.0, t_end, 501)
    q = mm.response(t, vdot0={3: np.r_[vel, 0, 0, 0]},
                    loads=[({7: np.r_[load, 0, 0, 0]}, g)])
    modal = np.array([mm.displacement(q, 2 * i + 1, c)
                      for i in range(4) for c in range(3)]).T

    def force(tt):
        f = np.zeros(12)
        f[9:12] = load * g(tt)
        return f

    v0 = np.zeros(12)
    v0[3:6] = vel
    dt = 1e-5
    r1 = _rk4(mg, kg, force, np.zeros(12), v0, t_end, dt)[::10]
    r2 = _rk4(mg, kg, force, np.zeros(12), v0, t_end, dt / 2)[::20]
    est = np.abs(r1 - r2).max() * 16.0 / 15.0
    assert np.abs(modal - r2).max() <= est + 1e-12 * np.abs(r2).max()
    assert est < 1e-6 * np.abs(r2).max()


def test_example_3_6_modal_damping_matrix():
    """BK Example 3.6 (Eq. 3.138-3.140): C_kp = C for the damper on the first
    mass."""
    k, m, c = 1.0, 1.0, 0.3
    els = [lib.Spring(2 * k, c=c), lib.LumpedMass(m), lib.Spring(k),
           lib.LumpedMass(m), lib.Spring(k), lib.LumpedMass(m),
           lib.Spring(2 * k)]
    ch = Chain(els, [0], [0])
    modes = ch.modes(ch.natural_frequencies(5.0))
    shapes = np.array([[md.states[i][0] for i in (1, 3, 5)] for md in modes])
    shapes /= shapes[:, :1]
    np.testing.assert_allclose(shapes, [[1, 2, 1], [1, 0, -1], [1, -1, 1]],
                               atol=1e-12)
    for md in modes:
        md.states = md.states / md.states[1][0]
    ckp = ModalModel(ch, modes).matrix("damping")
    np.testing.assert_allclose(ckp / c, CH3["3.6_damping"]["ckp_over_c"],
                               rtol=1e-12)


def test_modal_steady_state_equals_extended_tmm_for_proportional_damping():
    """XM Sec. 3.8: modal steady state (Eq. 3.181) = extended TMM (Eq. 2.119)
    for C = beta K."""
    beta = 0.01
    ks = [100.0, 60.0, 80.0]
    ms = [1.0, 2.0, 0.5]
    els = []
    for kk, mmass in zip(ks, ms):
        els += [lib.Spring(kk, c=beta * kk), lib.LumpedMass(mmass)]
    force = 2.0 - 0.5j
    els[3] = lib.LumpedMass(ms[1], force=force)
    ch = Chain(els, [0], [1])
    big_omega = 9.0
    z = ch.steady_state(big_omega)
    mm = ModalModel(ch, ch.modes(ch.natural_frequencies(30.0)))
    zeta = beta * mm.omegas / 2.0  # Eq. (3.149) with alpha = 0
    qa = mm.steady_state(big_omega, {3: [force, 0.0]}, zeta)
    for idx in (1, 3, 5):
        x_modal = sum(qa[p] * mm.modes[p].states[idx][0] for p in range(3))
        assert abs(x_modal - z[idx][0]) < 1e-12 * abs(z[idx][0])


# ----------------------------------------------------------------------
# Cross-check with the Haug engine (roadmap P5.6)
# ----------------------------------------------------------------------
def _solver():
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


_HAUG_M = [1.0, 2.0, 1.5]
_HAUG_K = [100.0, 200.0, 150.0, 300.0]


def _haug_chain(mbs, serial, damping):
    """Fixed-fixed spring-mass chain: bodies on prismatic joints along x,
    springs in series."""
    sys_ = mbs.DynamicsSystem(0.001)
    ground = mbs.create_ground(0)
    sys_.Add(ground)
    z3, x_axis, quat = [0, 0, 0], [1, 0, 0], [1, 0, 0, 0]
    bodies = []
    for i, m in enumerate(_HAUG_M):
        body = mbs.create_mobilized_body_quat(i + 1, [float(i + 1), 0, 0], z3,
                                              z3, quat, z3, z3, m,
                                              [0.1, 0.1, 0.1], z3, z3)
        sys_.Add(body)
        bodies.append(body)
    prev = ground
    for body in bodies:
        sys_.Add(mbs.create_joint(2, z3, z3, z3, z3, x_axis, x_axis,
                                  prev if serial else ground, body))
        prev = body
    ends = [ground] + bodies + [ground]
    for i, k in enumerate(_HAUG_K):
        bi, bj = ends[i], ends[i + 1]
        pi = [float(i), 0, 0] if bi is ground else z3
        pj = [float(i + 1), 0, 0] if bj is ground else z3
        sys_.Add(mbs.create_spring_damper(bi, bj, pi, pj, k, damping[i], 1.0))
    sys_.init()
    sys_.find_equilibrium()
    return mbs.modal_analysis(sys_)


def _tmm_haug_chain(damping):
    els = []
    for i, m in enumerate(_HAUG_M):
        els += [lib.Spring(_HAUG_K[i], c=damping[i]), lib.LumpedMass(m)]
    els.append(lib.Spring(_HAUG_K[-1], c=damping[-1]))
    return Chain(els, [0], [0])


@pytest.mark.parametrize("serial", [False, True])
def test_haug_modal_analysis_undamped_chain(serial):
    """XM P5.6: Haug modal_analysis frequencies (Hz) vs TMM, ground and serial
    prismatic chains."""
    mbs = _solver()
    res = _haug_chain(mbs, serial, [0.0] * 4)
    f_tmm = (_tmm_haug_chain([0.0] * 4).natural_frequencies(100.0)
             / (2 * math.pi))
    np.testing.assert_allclose(sorted(np.asarray(res.frequencies)), f_tmm,
                               rtol=1e-8)


def test_haug_modal_analysis_damped_chain():
    """XM P5.6: Haug |lambda|/2pi and damping ratios vs TMM damped eigenvalues
    (Sec. 2.7)."""
    mbs = _solver()
    damping = [0.5, 0.0, 0.8, 0.3]
    res = _haug_chain(mbs, False, damping)
    lam = _tmm_haug_chain(damping).damped_eigenvalues(100.0)
    order = np.argsort(res.frequencies)
    np.testing.assert_allclose(np.array(res.frequencies)[order],
                               np.abs(lam) / (2 * math.pi), rtol=1e-8)
    np.testing.assert_allclose(np.array(res.damping_ratios)[order],
                               -lam.real / np.abs(lam), rtol=1e-8)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
