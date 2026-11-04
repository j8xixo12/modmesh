# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of the discrete time transfer matrix method (book Ch. 7).

Check types (docs/roadmap.md Sec. 4.2): AN analytic, CV convergence order,
BK book figure, XM cross-method (Ch. 5 TMM and Haug engine).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tmm import dttmm as D
from solvcon.multibody.tmm import rigid as R

DATA = Path(__file__).parent / "data" / "multibody"
G_BOOK = 9.8


def _cube():
    """Body of Example 7.2: unit cube, mass 1, I at a face center."""
    return R.SpatialBody(1.0, [1 / 6, 5 / 12, 5 / 12], [1.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0])


def _order(errors):
    """Observed orders from errors at step sizes halved each time."""
    e = np.asarray(errors)
    return np.log2(e[:-1] / e[1:])


def _in_band(curve, t, y, extra_px):
    v = np.interp(curve["t"], t, y)
    pad = extra_px / curve["px_per_unit"]
    ok = (v >= np.array(curve["low"]) - pad) & (v <= np.array(curve["high"])
                                                + pad)
    return v, ok


# ---------------------------------------------------------------------------
# Linearization (Secs. 7.3.1-7.3.4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["newmark", "euler", "houbolt"])
def test_linearizers_are_exact_for_polynomials(method):
    """AN: Table 7.2 coefficients reproduce polynomial motion, tol 1e-9.

    Newmark (beta 1/4) and forward Euler are exact for quadratics; the
    Houbolt start (linear acceleration) and the four-point formula are exact
    for cubics. The tolerance covers round-off of the 1/dT^2 factors.
    """
    dt = 0.1
    if method == "houbolt":
        x = lambda t: 1.0 + 2.0 * t - 0.5 * t ** 2 + 0.3 * t ** 3  # noqa: E731
        v = lambda t: 2.0 - t + 0.9 * t ** 2  # noqa: E731
        a = lambda t: -1.0 + 1.8 * t  # noqa: E731
    else:
        x = lambda t: 1.0 + 2.0 * t - 0.75 * t ** 2  # noqa: E731
        v = lambda t: 2.0 - 1.5 * t  # noqa: E731
        a = lambda t: -1.5 + 0.0 * t  # noqa: E731
    lin = D.make_linearizer(method, dt)
    var = D.Variable(x(0.0), v(0.0), a(0.0))
    for i in range(1, 6):
        t = i * dt
        a_c, b, c_c, d = lin.coefficients(var)
        assert abs(a_c * x(t) + b - a(t)) < 1e-9
        assert abs(c_c * x(t) + d - v(t)) < 1e-9
        lin.update(var, x(t))


def test_trig_and_rotation_linearization_local_error_is_third_order():
    """AN: Eqs. (7.62)-(7.64) and (7.83) have local error O(dT^3).

    Halving dT must divide the error by 8 within the order band 3 +- 0.2.
    """
    th, w, al = 0.7, 1.3, -2.1
    errs_s, errs_a = [], []
    th3 = np.array([0.4, -0.3, 1.1])
    w3, al3 = np.array([0.8, -1.1, 0.5]), np.array([0.3, 0.9, -1.2])
    for dt in (0.04, 0.02, 0.01):
        exact = th + w * dt + 0.5 * al * dt * dt  # quadratic motion
        s_bar, c_bar, g1, g2 = D.trig_linearization(th, w, al, dt)
        errs_s.append(abs(np.cos(th) * exact + g2 - np.sin(exact)))
        ex3 = th3 + w3 * dt + 0.5 * al3 * dt * dt
        _, at, phi = D.rotation_linearization(th3, w3, dt)
        lin_a = sum(m * x for m, x in zip(at, ex3)) + phi
        errs_a.append(np.max(np.abs(lin_a - R.rotation_xyz(ex3))))
    for errs in (errs_s, errs_a):
        assert np.all(np.abs(_order(errs) - 3.0) < 0.2)


# ---------------------------------------------------------------------------
# Spring and lumped-mass chain (Sec. 7.2, Table 7.1, Fig. 7.3)
# ---------------------------------------------------------------------------

def _table_7_1():
    return D.SpringMassChain([1000.0] * 5, [0.1] * 5, [0.1] * 5,
                             [100.0, 0, 0, 0, 0])


def _exact_chain(ch, x0, t):
    """Modal solution of the linear chain M xdd + K x = f."""
    n = len(ch.mass)
    k = np.zeros((n, n))
    for j in range(n):
        k[j, j] += ch.stiffness[j]
        if j + 1 < n:
            k[j, j] += ch.stiffness[j + 1]
            k[j, j + 1] = k[j + 1, j] = -ch.stiffness[j + 1]
    f = ch.force.copy()
    f += ch.stiffness * ch.length
    f[:-1] -= ch.stiffness[1:] * ch.length[1:]
    xs = np.linalg.solve(k, f)
    w2, phi = np.linalg.eigh(k / ch.mass[0])
    c = phi.T @ (np.asarray(x0) - xs)
    return xs[:, None] + (phi * c) @ np.cos(np.outer(np.sqrt(w2), t))


def test_spring_mass_chain_matches_modal_solution_and_is_second_order():
    """AN + CV: Table 7.1 chain vs the modal solution, Newmark order 2 +-
    0.2."""
    ch = _table_7_1()
    x0 = [0.1, 0.2, 0.3, 0.4, 0.5]
    errs = []
    for dt in (4e-4, 2e-4, 1e-4):
        t, x = ch.simulate(x0, [0.0] * 5, 0.2, dt)
        errs.append(np.max(np.abs(x - _exact_chain(ch, x0, t).T)))
    assert np.all(np.abs(_order(errs) - 2.0) < 0.2)
    assert errs[-1] < 1e-3


def test_spring_mass_chain_euler_is_first_order_and_houbolt_second():
    """CV: forward Euler order 1 +- 0.2, Houbolt order 2 +- 0.2 (Table 7.2)."""
    ch = _table_7_1()
    x0 = [0.1, 0.2, 0.3, 0.4, 0.5]
    for method, order, steps in (("euler", 1.0, (2e-5, 1e-5, 5e-6)),
                                 ("houbolt", 2.0, (2e-4, 1e-4, 5e-5))):
        errs = []
        for dt in steps:
            t, x = ch.simulate(x0, [0.0] * 5, 0.05, dt, method=method)
            errs.append(np.max(np.abs(x - _exact_chain(ch, x0, t).T)))
        assert np.all(np.abs(_order(errs) - order) < 0.2), (method, errs)


def test_overall_matrix_product_loses_precision_for_small_steps():
    """Eq. (7.19) product vs Riccati transfer: equal at dT = 1e-2, product
    fails at 1e-4.

    The entries of the product grow like (m / (beta dT^2 K))^5; at
    dT = 1e-4 that is 1.6e23, more than the double precision range of
    exact cancellation. At dT = 1e-2 the growth is 4^5 = 1e3 and both
    solutions agree to 1e-10 m (round-off times 1e3).
    """
    ch = _table_7_1()
    x0 = [0.1, 0.2, 0.3, 0.4, 0.5]
    _, xp = ch.simulate(x0, [0.0] * 5, 0.2, 1e-2, solver="product")
    _, xr = ch.simulate(x0, [0.0] * 5, 0.2, 1e-2, solver="riccati")
    np.testing.assert_allclose(xp, xr, atol=1e-10)
    with np.errstate(all="ignore"):
        _, xp = ch.simulate(x0, [0.0] * 5, 0.01, 1e-4, solver="product")
    _, xr = ch.simulate(x0, [0.0] * 5, 0.01, 1e-4, solver="riccati")
    assert not np.all(np.abs(xp - xr) < 1e-3)


def test_book_table_7_1_figure_7_3():
    """BK Fig. 7.3 (p. 224): positions of lumped masses 1-5, 2 px beyond the
    band."""
    book = json.loads((DATA / "tmm_ch7_fig7_3.json").read_text())
    t, x = _table_7_1().simulate([0.1, 0.2, 0.3, 0.4, 0.5], [0.0] * 5, 1.0,
                                 1e-4)
    for k in range(5):
        curve = book["curves"]["x%d" % (k + 1)]
        v, ok = _in_band(curve, t, x[:, k], book["read_uncertainty_px"])
        assert ok.all(), (k,
                          list(zip(curve["t"], v, curve["low"],
                                   curve["high"])))


# ---------------------------------------------------------------------------
# Planar and spatial rigid bodies (Secs. 7.4-7.7)
# ---------------------------------------------------------------------------

def test_small_pendulum_period_equals_newmark_discrete_period():
    """AN: small-amplitude pendulum period equals the Newmark discrete period.

    For beta = 1/4 the discrete frequency is 2 arctan(w dT / 2) / dT. The
    DT-TMM linearization (Eqs. 7.61-7.64) adds only O(dT^4): measured
    2.4e-6 at dT = 0.025, tolerance 1e-5 relative (the Newmark period error
    itself is 1.0e-3).
    """
    g = 9.81
    bar = R.uniform_bar(2.0, 1.5)
    w = np.sqrt(bar.mass * g * 0.75 / bar.j_i)
    dt = 0.025
    out = D.PlanarChainDT([bar], (0.0, -g)).simulate([-np.pi / 2 + 1e-3],
                                                     [0.0], 40.0, dt)
    x = out["theta"][:, 0] + np.pi / 2
    t = out["t"]
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    tc = t[idx] - x[idx] * (t[idx + 1] - t[idx]) / (x[idx + 1] - x[idx])
    period = np.mean(np.diff(tc))
    period_newmark = 2 * np.pi * dt / (2 * np.arctan(w * dt / 2))
    assert abs(period - period_newmark) / period_newmark < 1e-5


def test_rotary_spring_damper_hinge_converges_to_damped_oscillator():
    """AN + CV: body on a rotary spring-damper hinge (Eqs. 7.208, 7.215), order
    2 +- 0.2."""
    body = R.PlanarBody(1.0, 0.5, [1.0, 0.0], [0.0, 0.0])
    k, c = 20.0, 0.4
    wn = np.sqrt(k / 0.5)
    zeta = c / (2 * np.sqrt(k * 0.5))
    wd = wn * np.sqrt(1 - zeta ** 2)
    errs = []
    for dt in (0.02, 0.01, 0.005):
        ch = D.PlanarChainDT([body], (0.0, 0.0),
                             hinges=[([1e3, 1e3, k], [0.0, 0.0, c])])
        out = ch.simulate([0.1], [0.0], 3.0, dt)
        t = out["t"]
        exact = 0.1 * np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t))
        errs.append(np.max(np.abs(out["theta"][:, 0] - exact)))
    assert np.all(np.abs(_order(errs) - 2.0) < 0.2)


@pytest.mark.parametrize("nb", [1, 2])
def test_planar_chain_newmark_is_second_order(nb):
    """CV: single and double pendulum, DT-TMM vs RK4 reference, order 2 +-
    0.2."""
    g = 9.81
    bars = [R.uniform_bar(1.0, 1.0)] * nb
    th0, w0 = [-np.pi / 3] * nb, [0.0] * nb
    t_ref, th_ref, _ = R.PlanarChain(bars, (0.0, -g)).simulate(th0, w0, 2.0,
                                                               2.5e-4)
    errs = []
    for dt in (0.01, 0.005, 0.0025):
        out = D.PlanarChainDT(bars, (0.0, -g)).simulate(th0, w0, 2.0, dt)
        k = int(round(dt / 2.5e-4))
        errs.append(np.max(np.abs(out["theta"] - th_ref[::k])))
    assert np.all(np.abs(_order(errs) - 2.0) < 0.2), errs


def test_houbolt_is_stable_where_newmark_is_not():
    """CV: triple pendulum released horizontally, Houbolt order 2 +- 0.2.

    The Newmark average acceleration method (no numerical damping) diverges
    for this release at dT <= 5e-3 (it diverges even at 1e-3); the
    Houbolt method converges.
    """
    b = R.PlanarBody(1.0, 5 / 12, [1.0, 0.0], [0.5, 0.0])
    _, th_ref, _ = R.PlanarChain([b] * 3, (0.0, -G_BOOK)).simulate(
        [0, 0, 0], [0, 0, 0], 2.0, 2.5e-4)
    errs = []
    for dt in (0.005, 0.0025, 0.00125):
        out = D.PlanarChainDT([b] * 3, (0.0, -G_BOOK)).simulate(
            [0, 0, 0], [0, 0, 0], 2.0, dt, method="houbolt")
        k = int(round(dt / 2.5e-4))
        errs.append(np.max(np.abs(out["theta"] - th_ref[::k])))
    assert np.all(np.abs(_order(errs) - 2.0) < 0.2), errs
    with np.errstate(all="ignore"), pytest.raises((np.linalg.LinAlgError,
                                                   ValueError,
                                                   FloatingPointError,
                                                   AssertionError)):
        out = D.PlanarChainDT([b] * 3, (0.0, -G_BOOK)).simulate([0, 0, 0],
                                                                [0, 0, 0], 2.0,
                                                                0.0025)
        assert np.all(np.abs(out["theta"] - th_ref[::10]) < 0.1)


def test_spatial_chain_newmark_is_second_order():
    """CV: spatial ball chain (Example 7.2 bodies), 0.5 s, order 2 +- 0.2.

    Step sizes 2.5e-3 to 6.25e-4 (at 5e-3 the error is not yet asymptotic).
    """
    cube = _cube()
    th0 = np.zeros((3, 3))
    thd0 = np.zeros((3, 3))
    thd0[2] = [0.2, 0.4, -0.1]
    ch = R.SpatialChain([cube] * 3, [R.BallHinge()] * 3, (0.0, -G_BOOK, 0.0))
    mats = [R.rotation_xyz(t) for t in th0]
    w0 = [a @ (D.h_matrix(t) @ td) for a, t, td in zip(mats, th0, thd0)]
    ref = ch.simulate([[1, 0, 0, 0]] * 3, w0, 0.5, 1.5625e-4)
    ang = np.array([[R.angles_xyz(R.quat_to_matrix(q)) for q in qq]
                    for qq in ref["quat"]])
    errs = []
    for dt in (0.0025, 0.00125, 0.000625):
        out = D.SpatialChainDT([cube] * 3, (0.0, -G_BOOK, 0.0)).simulate(th0,
                                                                         thd0,
                                                                         0.5,
                                                                         dt)
        k = int(round(dt / 1.5625e-4))
        errs.append(np.max(np.abs(out["theta"] - ang[::k])))
    assert np.all(np.abs(_order(errs) - 2.0) < 0.2), errs


# ---------------------------------------------------------------------------
# Book examples 7.1 and 7.2 (BK)
# ---------------------------------------------------------------------------

def test_book_example_7_1_figure_7_18():
    """BK Fig. 7.18 (p. 262): three planar pendulums, 2 px beyond the band.

    Houbolt dT 2e-3. The Newmark average acceleration run diverges after a
    few seconds of this example (see
    test_houbolt_is_stable_where_newmark_is_not).
    """
    book = json.loads((DATA / "tmm_ch7_fig7_18.json").read_text())
    out = D.PlanarChainDT([R.uniform_bar(1.0, 1.0)] * 3,
                          (0.0, -G_BOOK)).simulate(
        [-np.pi / 3] * 3, [0.0] * 3, 10.0, 2e-3, method="houbolt")
    for k, name in enumerate(["theta2", "theta4", "theta6"]):
        curve = book["curves"][name]
        v, ok = _in_band(curve, out["t"], out["theta"][:, k],
                         book["read_uncertainty_px"])
        assert ok.all(), (name,
                          list(zip(curve["t"], v, curve["low"],
                                   curve["high"])))


def test_book_example_7_2_figure_7_20():
    """BK Fig. 7.20 (p. 263): theta_z of bodies 2, 4, 6; Houbolt dT 2e-3, 2 px
    beyond band."""
    book = json.loads((DATA / "tmm_ch7_fig7_20.json").read_text())
    th0 = np.zeros((3, 3))
    thd0 = np.zeros((3, 3))
    thd0[2, 1] = 0.1
    out = D.SpatialChainDT([_cube()] * 3, (0.0, -G_BOOK, 0.0)).simulate(
        th0, thd0, 10.0, 2e-3, method="houbolt")
    for k, name in enumerate(["theta_z2", "theta_z4", "theta_z6"]):
        curve = book["curves"][name]
        v, ok = _in_band(curve, out["t"], out["theta"][:, k, 2],
                         book["read_uncertainty_px"])
        assert ok.all(), (name,
                          list(zip(curve["t"], v, curve["low"],
                                   curve["high"])))


# ---------------------------------------------------------------------------
# Cross-check with the Haug engine (XM)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mbs():
    """Import the compiled Haug engine or skip."""
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def test_xm_dttmm_planar_triple_pendulum_vs_haug(mbs):
    """XM: Example 7.1 system, DT-TMM (Newmark dT 5e-4) vs Haug (RK45), 2 s.

    Tolerance 2e-5 m: the DT-TMM error at this step, estimated from the
    difference of the runs at dT and 2 dT (factor 1/3 for order 2), is
    below 1e-5 m; the Haug error is below 1e-8 m.
    """
    g, n = G_BOOK, 3
    th0 = np.array([-np.pi / 3] * n)
    z3 = [0.0, 0.0, 0.0]
    s = mbs.DynamicsSystem(1e-3)
    gr = mbs.create_ground(0)
    s.Add(gr)
    bodies = [gr]
    p = np.zeros(3)
    for k in range(n):
        u = np.array([np.cos(th0[k]), np.sin(th0[k]), 0.0])
        b = mbs.create_mobilized_body_quat(k + 1, list(p + 0.5 * u), z3, z3,
                                           [np.cos(th0[k] / 2), 0, 0,
                                            np.sin(th0[k] / 2)],
                                           z3, z3, 1.0, [1e-4, 1 / 12, 1 / 12],
                                           [0, -g, 0], z3)
        s.Add(b)
        bodies.append(b)
        p = p + u
    s.Add(mbs.create_joint(1, z3, [-0.5, 0, 0], z3, z3, [0, 0, 1], [0, 0, 1],
                           gr, bodies[1]))
    for k in range(1, n):
        s.Add(mbs.create_joint(1, [0.5, 0, 0], [-0.5, 0, 0], z3, z3, [0, 0, 1],
                               [0, 0, 1], bodies[k], bodies[k + 1]))
    s.set_integrator("rk45")
    s.init()
    for _ in range(2000):
        s.solve()
    pos = np.array(s.get_body_positions())[1:][:, :2]
    bars = [R.uniform_bar(1.0, 1.0)] * n
    finals = []
    for dt in (1e-3, 5e-4):
        out = D.PlanarChainDT(bars, (0.0, -g)).simulate(th0, [0.0] * n, 2.0,
                                                        dt)
        finals.append(R.PlanarChain(bars).positions(out["theta"][-1])[1])
    est = np.max(np.abs(finals[1] - finals[0])) / 3.0
    assert est < 1e-5
    assert np.max(np.abs(finals[1] - pos)) < 2e-5

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
