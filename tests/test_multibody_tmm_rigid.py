# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of the multi-rigid-body transfer matrix method (book Ch. 5).

Check types (docs/roadmap.md Sec. 4.2): AN analytic, IV invariant,
BK book figure, XM cross-method with the Haug engine.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import reaction_arrays
from solvcon.multibody.tmm import rigid as R

DATA = Path(__file__).parent / "data" / "multibody"
G_BOOK = 9.8  # g of the book examples; reproduces Fig. 5.5 (see test below)


def _cube():
    """Body of Secs. 5.10.1, 5.10.2: unit cube, mass 1, I at a face center."""
    return R.SpatialBody(1.0, [1 / 6, 5 / 12, 5 / 12], [1.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0])


def _agm(a, b):
    while abs(a - b) > 1e-15 * a:
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return a


def _period(t, x):
    """Mean period from upward zero crossings with linear interpolation."""
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    tc = t[idx] - x[idx] * (t[idx + 1] - t[idx]) / (x[idx + 1] - x[idx])
    return np.mean(np.diff(tc))


# ---------------------------------------------------------------------------
# Element matrices against the Newton-Euler equations (AN)
# ---------------------------------------------------------------------------

def test_planar_body_matrix_matches_newton_euler():
    """Eq. (5.18) against a direct Newton-Euler evaluation, tol 1e-12
    (round-off)."""
    rng = np.random.default_rng(1)
    m, j_i = 2.3, 0.7
    l_io, l_ic = np.array([1.2, 0.3]), np.array([0.5, -0.2])
    th, w = 0.4, -1.3
    f, mc = np.array([0.4, -9.0]), 0.25
    u = R.planar_body_matrix(m, j_i, l_io, l_ic, th, w, f, mc)
    z = np.concatenate([rng.normal(size=6), [1.0]])
    rdd, om, m_i, q_i = z[0:2], z[2], z[3], z[4:6]
    r = R.rot2(th)
    r_io, r_ic = r @ l_io, r @ l_ic

    def perp(v):
        return np.array([-v[1], v[0]])

    rdd_c = rdd + om * perp(r_ic) - w * w * r_ic
    q_o = q_i - m * rdd_c + f
    # Moment about I with m = moment of the outboard on the inboard element.
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]  # noqa: E731
    m_o = (m_i + j_i * om + m * cross(r_ic, rdd) - mc + cross(r_io, q_o)
           - cross(r_ic, f))
    z_o = u @ z
    np.testing.assert_allclose(z_o[0:2], rdd + om * perp(r_io) - w * w * r_io,
                               atol=1e-12)
    np.testing.assert_allclose(z_o[2], om, atol=1e-12)
    np.testing.assert_allclose(z_o[3], m_o, atol=1e-12)
    np.testing.assert_allclose(z_o[4:6], q_o, atol=1e-12)


def test_spatial_body_matrix_matches_newton_euler():
    """Eq. (5.27) against a direct Newton-Euler evaluation, tol 1e-12."""
    rng = np.random.default_rng(2)
    m = 1.7
    jb = np.array([[0.3, 0.02, -0.01], [0.02, 0.5, 0.03], [-0.01, 0.03, 0.4]])
    a = R.rotation_xyz([0.3, -0.5, 1.1])
    l_io, l_ic = np.array([1.0, 0.2, -0.1]), np.array([0.4, 0.1, 0.05])
    w = np.array([0.3, -1.2, 0.8])
    f, mc = np.array([0.5, -16.0, 0.2]), np.array([0.1, 0.2, -0.3])
    u = R.spatial_body_matrix(m, jb, a, l_io, l_ic, w, f, mc)
    z = np.concatenate([rng.normal(size=12), [1.0]])
    rdd, om, m_i, q_i = z[0:3], z[3:6], z[6:9], z[9:12]
    r_io, r_ic = a @ l_io, a @ l_ic
    jg = a @ jb @ a.T
    rdd_c = rdd + np.cross(om, r_ic) + np.cross(w, np.cross(w, r_ic))
    q_o = q_i - m * rdd_c + f
    m_o = (m_i + jg @ om + np.cross(w, jg @ w) + m * np.cross(r_ic, rdd) - mc
           + np.cross(r_io, q_o) - np.cross(r_ic, f))
    z_o = u @ z
    rdd_o = rdd + np.cross(om, r_io) + np.cross(w, np.cross(w, r_io))
    np.testing.assert_allclose(z_o[0:3], rdd_o, atol=1e-12)
    np.testing.assert_allclose(z_o[6:9], m_o, atol=1e-12)
    np.testing.assert_allclose(z_o[9:12], q_o, atol=1e-12)


def test_hinge_matrices_give_zero_outboard_moment():
    """Eqs. (5.38), (5.49), (5.61): m at the outboard end of the outboard body
    is zero."""
    rng = np.random.default_rng(3)
    u = R.planar_body_matrix(1.0, 0.4, [1, 0.1], [0.5, 0], 0.3, 0.7, [0, -9.8])
    z = np.concatenate([rng.normal(size=6), [1.0]])
    assert abs((u @ R.planar_pin_hinge_matrix(u) @ z)[3]) < 1e-12
    cube = _cube()
    a = R.rotation_xyz([0.2, 0.1, -0.4])
    us = R.spatial_body_matrix(1.0, cube.j_i, a, cube.l_io, cube.l_ic,
                               [0.1, 0.2, 0.3], [0, -9.8, 0])
    z = np.concatenate([rng.normal(size=12), [1.0]])
    np.testing.assert_allclose((us @ R.ball_hinge_matrix(us) @ z)[6:9], 0.0,
                               atol=1e-12)
    n, n2 = np.array([0, 0, 1.0]), a @ np.array([0, 1.0, 0])
    hp = R.spatial_pin_hinge_matrix(us, n, n2, [0.0, 0.0, 0.2],
                                    [0.1, -0.2, 0.5])
    assert abs(n2 @ (us @ hp @ z)[6:9]) < 1e-12


# ---------------------------------------------------------------------------
# Analytic checks
# ---------------------------------------------------------------------------

def test_simple_pendulum_period_matches_elliptic_integral():
    """AN: T = 4 sqrt(J_I/(m g d)) K(sin(a/2)); RK4 dt 1e-3, tol 1e-8 relative.

    The RK4 phase error over 10 s at this step is below 1e-11; the
    zero-crossing interpolation error is below 1e-9.
    """
    g = 9.81
    bar = R.uniform_bar(2.0, 1.5)
    amp = 0.8
    ch = R.PlanarChain([bar], gravity=(0.0, -g))
    t, th, _ = ch.simulate([-np.pi / 2 + amp], [0.0], 12.0, 1e-3)
    k = math.sin(amp / 2)
    period = 4 * math.sqrt(bar.j_i / (bar.mass * g * 0.75)) * math.pi / (
        2 * _agm(1.0, math.sqrt(1 - k * k)))
    assert abs(_period(t, th[:, 0] + np.pi / 2) - period) / period < 1e-8


def test_double_pendulum_small_amplitude_frequencies():
    """AN: linearized eigenfrequencies of two uniform bars, tol 1e-5 relative.

    Pure mode shapes at amplitude 1e-3 rad; the amplitude correction of the
    period is of order 1e-7.
    """
    g, m, ln = 9.81, 1.0, 1.0
    mass = np.array([[m * ln * ln / 3 + m * ln * ln, m * ln * ln / 2],
                     [m * ln * ln / 2, m * ln * ln / 3]])
    stiff = np.diag([1.5 * m * g * ln, 0.5 * m * g * ln])
    w2, vec = np.linalg.eig(np.linalg.solve(mass, stiff))
    ch = R.PlanarChain([R.uniform_bar(m, ln)] * 2, gravity=(0.0, -g))
    for k in range(2):
        v = vec[:, k] / np.max(np.abs(vec[:, k])) * 1e-3
        t, th, _ = ch.simulate(-np.pi / 2 + v, [0.0, 0.0], 12.0, 2e-3)
        period = _period(t, th[:, 0] + np.pi / 2)
        assert abs(period - 2 * np.pi / np.sqrt(w2[k])) / period < 1e-5


def test_free_fall_with_acceleration_hinge():
    """AN: a body on a zero-stiffness acceleration hinge falls freely (Eq.
    5.74).

    r(t) = r0 + g t^2 / 2 exactly; RK4 is exact for this quadratic, tol 1e-12.
    """
    g = np.array([0.0, -9.81, 0.0])
    ch = R.SpatialChain([_cube()], [R.BushingHinge([0, 0, 1], 0.0, 0.0)],
                        gravity=g)
    out = ch.simulate([[1, 0, 0, 0]], [[0.0, 0.0, 0.0]], 1.0, 0.01)
    np.testing.assert_allclose(out["r_in"][-1, 0], 0.5 * g * 1.0, atol=1e-12)


def test_hanging_chain_internal_forces():
    """AN: a chain hanging at rest has q_k = (n - k) m g at hinge k.

    Tolerance 1e-10: the only error is round-off (cos(-pi/2) = 6e-17)
    amplified by the hinge eliminations.
    """
    g = 9.81
    ch = R.PlanarChain([R.uniform_bar(1.5, 0.7)] * 4, gravity=(0.0, -g))
    states, alpha = ch.transfer([-np.pi / 2] * 4, [0.0] * 4)
    np.testing.assert_allclose(alpha, 0.0, atol=1e-10)
    for k, (z_in, _) in enumerate(states):
        np.testing.assert_allclose(z_in[4:6], [0.0, (4 - k) * 1.5 * g],
                                   atol=1e-10)
    sch = R.SpatialChain([_cube()] * 3, [R.BallHinge()] * 3,
                         gravity=(0.0, -g, 0.0))
    q = R.matrix_to_quat(R.rotation_xyz([0, 0, -np.pi / 2]))
    states, alpha, _ = sch.transfer([q] * 3, np.zeros((3, 3)))
    np.testing.assert_allclose(alpha, 0.0, atol=1e-10)
    for k, (z_in, _) in enumerate(states):
        np.testing.assert_allclose(z_in[9:12], [0.0, (3 - k) * g, 0.0],
                                   atol=1e-10)


def test_spatial_ball_chain_conserves_energy():
    """IV: energy of the 5.10.1 system over 3 s, RK4 dt 2e-3.

    The drift relative to the energy scale n m g L must stay below 1e-8; the
    RK4 energy error at this step is of order 1e-9 of that scale.
    """
    cube = _cube()
    ch = R.SpatialChain([cube] * 3, [R.BallHinge()] * 3,
                        gravity=(0.0, -G_BOOK, 0.0))
    w0 = np.zeros((3, 3))
    w0[2] = [0.3, 0.5, -0.2]
    out = ch.simulate([[1, 0, 0, 0]] * 3, w0, 3.0, 2e-3)

    def energy(q, w):
        _, rc, _ = ch.positions(q)
        mats = [R.quat_to_matrix(e) for e in q]
        v = np.zeros(3)
        e = 0.0
        for a, om, r in zip(mats, w, rc):
            vc = v + np.cross(om, a @ cube.l_ic)
            jc = a @ cube.inertia_about_c() @ a.T
            e += (0.5 * cube.mass * vc @ vc + 0.5 * om @ jc @ om
                  + cube.mass * G_BOOK * r[1])
            v = v + np.cross(om, a @ cube.l_io)
        return e

    e0 = energy(out["quat"][0], out["omega"][0])
    e1 = energy(out["quat"][-1], out["omega"][-1])
    assert abs(e1 - e0) / (3 * cube.mass * G_BOOK * 1.0) < 1e-8


def test_pin_chain_orthogonal_axes_uses_augmented_hinge():
    """IV: pin axes z, y, x (E_0 of Eq. 5.59 singular), RK4 dt 2e-3, 2 s.

    Energy drift relative to n m g L below 1e-8 (RK4 error level) and the
    relative angular velocity of hinge 1 stays along its axis (tol 1e-8).
    """
    cube = _cube()
    hs = [R.PinHinge([0, 0, 1]), R.PinHinge([0, 1, 0]), R.PinHinge([1, 0, 0])]
    ch = R.SpatialChain([cube] * 3, hs, gravity=(0.0, -G_BOOK, 0.0))
    w0 = np.array([[0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])
    out = ch.simulate([[1, 0, 0, 0]] * 3, w0, 2.0, 2e-3)

    def energy(q, w):
        _, rc, _ = ch.positions(q)
        e = 0.0
        for e_, om, r in zip(q, w, rc):
            a = R.quat_to_matrix(e_)
            e += (cube.mass * G_BOOK * r[1] + 0.5 * om
                  @ (a @ cube.inertia_about_c() @ a.T) @ om)
        _, _, ro = ch.positions(q)
        v = np.zeros(3)
        for e_, om in zip(q, w):
            a = R.quat_to_matrix(e_)
            vc = v + np.cross(om, a @ cube.l_ic)
            e += 0.5 * cube.mass * vc @ vc
            v = v + np.cross(om, a @ cube.l_io)
        return e

    e0 = energy(out["quat"][0], out["omega"][0])
    e1 = energy(out["quat"][-1], out["omega"][-1])
    assert abs(e1 - e0) / (3 * cube.mass * G_BOOK * 1.0) < 1e-8
    # The relative angular velocity stays along the hinge axes (constraint).
    q, w = out["quat"][-1], out["omega"][-1]
    a0 = R.quat_to_matrix(q[0])
    n1 = a0 @ np.array([0.0, 1.0, 0.0])
    rel = w[1] - w[0]
    assert np.linalg.norm(rel - (rel @ n1) * n1) < 1e-8


def test_bushing_chain_converges_to_pin_chain():
    """Sec. 5.10.3: stiff spring hinges reproduce the smooth pin hinges.

    The hinge deflection scales with 1/k, so the angle difference to the
    pin chain must drop by a factor 10 +- 30 % from k = 1e4 to 1e5.
    """
    cube = _cube()
    axes = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
    pin = R.SpatialChain([cube] * 3, [R.PinHinge(a) for a in axes],
                         gravity=(0, -G_BOOK, 0))
    w0 = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 1.0]])
    ref = pin.simulate([[1, 0, 0, 0]] * 3, w0, 0.5, 2e-4)
    th_ref = R.pin_relative_angles(pin, ref["quat"][-1], ref["quat"][0])
    errs = []
    for k in (1e4, 1e5):
        hs = [R.BushingHinge(a, k, k) for a in axes]
        ch = R.SpatialChain([cube] * 3, hs, gravity=(0, -G_BOOK, 0))
        out = ch.simulate([[1, 0, 0, 0]] * 3, w0, 0.5, 2e-4)
        th = R.pin_relative_angles(ch, out["quat"][-1], out["quat"][0])
        errs.append(np.max(np.abs(th - th_ref)))
    assert 7.0 < errs[0] / errs[1] < 13.0


# ---------------------------------------------------------------------------
# Book example 5.10.1 (BK)
# ---------------------------------------------------------------------------

def _in_band(curve, t, y, extra_px):
    v = np.interp(curve["t"], t, y)
    pad = extra_px / curve["px_per_unit"]
    low = np.array(curve["low"]) - pad
    high = np.array(curve["high"]) + pad
    return v, (v >= low) & (v <= high)


def test_book_5_10_1_space_angles_of_body_1():
    """BK Fig. 5.5 (p. 195): theta_3 of body 1, read uncertainty 2 px beyond
    the band."""
    book = json.loads((DATA / "tmm_ch5_fig5_5.json").read_text())
    ch = R.SpatialChain([_cube()] * 3, [R.BallHinge()] * 3,
                        gravity=(0.0, -G_BOOK, 0.0))
    w0 = np.zeros((3, 3))
    w0[2, 1] = 0.1
    out = ch.simulate([[1, 0, 0, 0]] * 3, w0, 10.0, 2.5e-3)
    ang = np.unwrap(np.array([R.angles_xyz(R.quat_to_matrix(q[0]))
                              for q in out["quat"]]), axis=0)
    curve = book["curves"]["theta3"]
    v, ok = _in_band(curve, out["t"], ang[:, 2], book["read_uncertainty_px"])
    assert ok.all(), list(zip(curve["t"], v, curve["low"], curve["high"]))


def test_book_5_10_2_pin_hinge_angles():
    """BK Fig. 5.7 (p. 196): relative angles of the three pin hinges.

    The hinge axes are not stated in the text; z (ground), y and z (inboard
    body frames) reproduce the whole figure. Read uncertainty 2 px beyond
    the band.
    """
    book = json.loads((DATA / "tmm_ch5_fig5_7.json").read_text())
    axes = [[0, 0, 1], [0, 1, 0], [0, 0, 1]]
    ch = R.SpatialChain([_cube()] * 3, [R.PinHinge(a) for a in axes],
                        gravity=(0.0, -G_BOOK, 0.0))
    w0 = np.array([[0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])
    q0 = [[1, 0, 0, 0]] * 3
    out = ch.simulate(q0, w0, 10.0, 2.5e-3)
    th = np.unwrap(np.array([R.pin_relative_angles(ch, q,
                                                   q0) for q in out["quat"]]),
                   axis=0)
    for k, name in enumerate(["Theta1", "Theta3", "Theta5"]):
        curve = book["curves"][name]
        v, ok = _in_band(curve, out["t"], th[:, k],
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


Z3 = [0.0, 0.0, 0.0]


def _haug_planar_chain(mbs, n, length, mass, theta0, g, dt):
    s = mbs.DynamicsSystem(dt)
    gr = mbs.create_ground(0)
    s.Add(gr)
    bodies = [gr]
    p = np.zeros(3)
    for k in range(n):
        u = np.array([np.cos(theta0[k]), np.sin(theta0[k]), 0.0])
        c = p + 0.5 * length * u
        q = [np.cos(theta0[k] / 2), 0, 0, np.sin(theta0[k] / 2)]
        jc = mass * length ** 2 / 12
        b = mbs.create_mobilized_body_quat(k + 1, list(c), Z3, Z3, q, Z3, Z3,
                                           mass, [1e-4, jc, jc],
                                           [0, -mass * g, 0], Z3)
        s.Add(b)
        bodies.append(b)
        p = p + length * u
    s.Add(mbs.create_joint(1, Z3, [-0.5 * length, 0, 0], Z3, Z3, [0, 0, 1],
                           [0, 0, 1], gr, bodies[1]))
    for k in range(1, n):
        s.Add(mbs.create_joint(1, [0.5 * length, 0, 0], [-0.5 * length, 0, 0],
                               Z3, Z3, [0, 0, 1], [0, 0, 1], bodies[k],
                               bodies[k + 1]))
    s.set_integrator("rk45")
    s.init()
    return s


def test_xm_planar_triple_pendulum_positions_and_reactions(mbs):
    """XM: Ch. 5 TMM vs Haug, planar triple pendulum, 2 s.

    Positions tol 1e-6 m (both RK-type integrators at dt 1e-3, errors
    below 1e-8 here); joint reactions at t = 0 tol 1e-9 N (no integration).
    """
    g, n, length, mass = 9.81, 3, 1.0, 1.0
    th0 = np.array([-np.pi / 3] * n)
    s = _haug_planar_chain(mbs, n, length, mass, th0, g, 1e-3)
    ch = R.PlanarChain([R.uniform_bar(mass, length)] * n, gravity=(0.0, -g))
    states, _ = ch.transfer(th0, np.zeros(n))
    reac = reaction_arrays(s.get_joint_reactions())
    for k in range(n):
        # q at hinge k = force of the inboard element on the outboard body.
        np.testing.assert_allclose(reac[k]["force_j"][:2], states[k][0][4:6],
                                   atol=1e-9)
    t, th, _ = ch.simulate(th0, np.zeros(n), 2.0, 1e-3)
    worst = 0.0
    for i in range(2000):
        s.solve()
        if (i + 1) % 100 == 0:
            pos = np.array(s.get_body_positions())[1:][:, :2]
            _, rc, _ = ch.positions(th[i + 1])
            worst = max(worst, np.max(np.abs(pos - rc)))
    assert worst < 1e-6


def test_xm_spatial_ball_chain_5_10_1(mbs):
    """XM: Ch. 5 TMM vs Haug, system of Sec. 5.10.1, 3 s, positions tol 1e-6
    m."""
    g = G_BOOK
    cube = _cube()
    jc = np.diag(cube.inertia_about_c())
    dt = 1e-3
    s = mbs.DynamicsSystem(dt)
    gr = mbs.create_ground(0)
    s.Add(gr)
    bodies = [gr]
    w0 = np.zeros((3, 3))
    w0[2, 1] = 0.1
    v = np.zeros(3)
    for k in range(3):
        vc = v + np.cross(w0[k], [0.5, 0, 0])
        v = v + np.cross(w0[k], [1.0, 0, 0])
        b = mbs.create_mobilized_body_quat(k + 1, [k + 0.5, 0, 0], list(vc),
                                           Z3, [1, 0, 0, 0], list(w0[k]), Z3,
                                           1.0, list(jc), [0, -g, 0], Z3)
        s.Add(b)
        bodies.append(b)
    s.Add(mbs.create_joint(0, Z3, [-0.5, 0, 0], Z3, Z3, [0, 0, 1], [0, 0, 1],
                           gr, bodies[1]))
    for k in range(1, 3):
        s.Add(mbs.create_joint(0, [0.5, 0, 0], [-0.5, 0, 0], Z3, Z3, [0, 0, 1],
                               [0, 0, 1], bodies[k], bodies[k + 1]))
    s.set_integrator("rk45")
    s.init()
    ch = R.SpatialChain([cube] * 3, [R.BallHinge()] * 3,
                        gravity=(0.0, -g, 0.0))
    out = ch.simulate([[1, 0, 0, 0]] * 3, w0, 3.0, dt)
    worst = 0.0
    for i in range(3000):
        s.solve()
        if (i + 1) % 100 == 0:
            pos = np.array(s.get_body_positions())[1:]
            _, rc, _ = ch.positions(out["quat"][i + 1])
            worst = max(worst, np.max(np.abs(pos - rc)))
    assert worst < 1e-6


def test_xm_spatial_pin_arm(mbs):
    """XM: Ch. 5 TMM vs Haug, pin arm with axes z, y, z (Sec. 5.10.2 model), 2
    s.

    Positions tol 1e-6 m.
    """
    g = G_BOOK
    cube = _cube()
    jc = np.diag(cube.inertia_about_c())
    axes = [[0, 0, 1], [0, 1, 0], [0, 0, 1]]
    w0 = np.array([[0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])
    dt = 1e-3
    s = mbs.DynamicsSystem(dt)
    gr = mbs.create_ground(0)
    s.Add(gr)
    bodies = [gr]
    # Velocity of the mass centers from the initial angular velocities.
    vel = [np.zeros(3), np.cross(w0[1], [0.5, 0, 0]),
           np.cross(w0[1], [1.0, 0, 0])
           + np.cross(w0[2], [0.5, 0, 0])]
    for k in range(3):
        b = mbs.create_mobilized_body_quat(k + 1, [k + 0.5, 0, 0],
                                           list(vel[k]), Z3, [1, 0, 0, 0],
                                           list(w0[k]), Z3, 1.0, list(jc),
                                           [0, -g, 0], Z3)
        s.Add(b)
        bodies.append(b)
    s.Add(mbs.create_joint(1, Z3, [-0.5, 0, 0], Z3, Z3, axes[0], axes[0], gr,
                           bodies[1]))
    for k in range(1, 3):
        s.Add(mbs.create_joint(1, [0.5, 0, 0], [-0.5, 0, 0], Z3, Z3, axes[k],
                               axes[k], bodies[k], bodies[k + 1]))
    s.set_integrator("rk45")
    s.init()
    ch = R.SpatialChain([cube] * 3, [R.PinHinge(a) for a in axes],
                        gravity=(0.0, -g, 0.0))
    out = ch.simulate([[1, 0, 0, 0]] * 3, w0, 2.0, dt)
    worst = 0.0
    for i in range(2000):
        s.solve()
        if (i + 1) % 100 == 0:
            pos = np.array(s.get_body_positions())[1:]
            _, rc, _ = ch.positions(out["quat"][i + 1])
            worst = max(worst, np.max(np.abs(pos - rc)))
    assert worst < 1e-6

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
