# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of the automatic deduction of the overall transfer equation (book Ch.
11).

Check types (docs/roadmap.md Sec. 4.2): AN analytic, IV invariant,
CV convergence order, BK book figure, XM cross-method (Haug engine).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tmm import dttmm as D
from solvcon.multibody.tmm import rigid as R
from solvcon.multibody.tmm import topology as T

DATA = Path(__file__).parent / "data" / "multibody"
G_BOOK = 9.8


# ---------------------------------------------------------------------------
# Matrix level: theorem of Sec. 11.6
# ---------------------------------------------------------------------------

def _figure_11_1():
    """Topology figure of Fig. 11.1 (transfer from the tips to the root 18)."""
    fig = T.TopologyFigure()
    for e, ins in [(1, ["z1,0"]), (2, ["z2,0"]), (3, ["z3,0"]), (4, ["z4,0"]),
                   (5, [1]), (6, [2]), (7, [3]), (8, [4]), (9, [5, 6]),
                   (10, [7, 8]), (11, [9]), (13, [11]), (14, [13]), (12, [10]),
                   (15, [14, 12]), (16, [15]), (17, [16]), (18, [17])]:
        fig.add(e, ins)
    return fig


def _random_matrices(seed, n=7):
    rng = np.random.default_rng(seed)
    u = {e: rng.normal(size=(n, n)) for e in range(1, 19)}
    u_in = {(9, 1): rng.normal(size=(n, n)), (10, 1): rng.normal(size=(n, n)),
            (15, 1): rng.normal(size=(n, n))}
    h = {e: rng.normal(size=(3, n)) for e in (9, 10, 15)}
    h_in = {(e, 1): rng.normal(size=(3, n)) for e in (9, 10, 15)}
    return u, u_in, h, h_in


def _prod(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = out @ m
    return out


def test_deduce_tree_reproduces_equations_11_8_to_11_21():
    """AN: T and G of Fig. 11.1 equal the hand deduction of Eqs. (11.9),
    (11.15), (11.18).

    Random element matrices; tolerance 1e-9 relative (round-off of products).
    """
    u, u_in, h, h_in = _random_matrices(5)
    fig = _figure_11_1()
    u_all, labels, t_mats, g_rows = T.deduce_tree(fig, u, u_in, h, h_in)
    assert labels == ["root", "z1,0", "z2,0", "z3,0", "z4,0"]
    up = _prod(u[18], u[17], u[16])
    t1 = _prod(up, u[15], u[14], u[13], u[11], u[9], u[5], u[1])
    t2 = _prod(up, u[15], u[14], u[13], u[11], u_in[(9, 1)], u[6], u[2])
    t3 = _prod(up, u_in[(15, 1)], u[12], u[10], u[7], u[3])
    t4 = _prod(up, u_in[(15, 1)], u[12], u_in[(10, 1)], u[8], u[4])
    expect_g = [
        {"z1,0": -h[9] @ u[5] @ u[1], "z2,0": h_in[(9, 1)] @ u[6] @ u[2]},
        {"z3,0": -h[10] @ u[7] @ u[3], "z4,0": h_in[(10, 1)] @ u[8] @ u[4]},
        {"z1,0": -_prod(h[15], u[14], u[13], u[11], u[9], u[5], u[1]),
         "z2,0": -_prod(h[15], u[14], u[13], u[11], u_in[(9, 1)], u[6], u[2]),
         "z3,0": _prod(h_in[(15, 1)], u[12], u[10], u[7], u[3]),
         "z4,0": _prod(h_in[(15, 1)], u[12], u_in[(10, 1)], u[8], u[4])},
    ]
    for tip, m in zip(labels[1:], (t1, t2, t3, t4)):
        np.testing.assert_allclose(t_mats[tip], m, rtol=1e-9)
    by_elem = dict(zip([9, 10, 15], g_rows))
    for e, exp in zip([9, 10, 15], expect_g):
        assert set(by_elem[e]) == set(exp)
        for tip in exp:
            np.testing.assert_allclose(by_elem[e][tip], exp[tip], rtol=1e-9)
    # Block structure of Eq. (11.23): [-I T; O G].
    np.testing.assert_allclose(u_all[0:7, 0:7], -np.eye(7))
    assert np.all(u_all[7:, 0:7] == 0.0)


def test_apply_cuts_merges_cut_point_states():
    """AN: Eqs. (11.28)-(11.34): columns of z_b times C are added to z_a."""
    u, u_in, h, h_in = _random_matrices(6)
    fig = _figure_11_1()
    u_all, labels, _, _ = T.deduce_tree(fig, u, u_in, h, h_in)
    c = T.cut_sign_matrix(3)
    merged, lab2 = T.apply_cuts(u_all, labels, [("z2,0", "z3,0")], c)
    assert lab2 == ["root", "z1,0", "z2,0", "z4,0"]
    np.testing.assert_allclose(merged[:, 14:21],
                               u_all[:, 14:21] + u_all[:, 21:28] @ c,
                               rtol=1e-12)
    np.testing.assert_allclose(c @ c, np.eye(7))


def test_closed_loop_ring_characteristic_equation():
    """AN: ring of Fig. 11.7, det(U_6 ... U_1 - I) = 0 at the ring
    eigenfrequencies.

    Element matrices of Eqs. (7.11) and (7.14) with xdd = -w^2 x. The
    smallest singular value of U_all at an eigenfrequency is below 1e-9
    times its value at a frequency between two eigenfrequencies.
    """
    k1, k3, k5 = 1000.0, 1500.0, 800.0
    m2, m4, m6 = 1.0, 2.0, 1.5

    def u_all(w):
        mats = [D.spring_matrix_1d(k1, 0.0),
                D.lumped_mass_matrix_1d(m2, -w * w, 0.0),
                D.spring_matrix_1d(k3, 0.0),
                D.lumped_mass_matrix_1d(m4, -w * w, 0.0),
                D.spring_matrix_1d(k5, 0.0),
                D.lumped_mass_matrix_1d(m6, -w * w, 0.0)]
        return T.closed_loop_matrix([m[:2, :2] for m in mats])

    kmat = np.array([[k1 + k3, -k3, -k1], [-k3, k3 + k5, -k5],
                     [-k1, -k5, k1 + k5]])
    mmat = np.diag([m2, m4, m6])
    w2 = np.sort(np.linalg.eigvals(np.linalg.solve(mmat, kmat)).real)
    wk = np.sqrt(np.clip(w2, 0.0, None))
    ref = np.linalg.svd(u_all(0.5 * (wk[1] + wk[2])), compute_uv=False)[-1]
    for w in wk:
        assert np.linalg.svd(u_all(w), compute_uv=False)[-1] < 1e-9 * ref


def test_solve_boundary_with_known_entries():
    """AN: Eqs. (11.113)-(11.116) on a cantilever spring chain, exact to
    1e-12."""
    us = D.spring_matrix_1d(200.0, 0.3)
    u = us @ us
    # z_n = U z_0, z_0 = [0, q0, 1] (wall), z_n = [x, -5, 1] (load 5 N).
    u_all = np.hstack([u, -np.eye(3)])
    known = np.array([0.0, np.nan, 1.0, np.nan, -5.0, 1.0])
    z, res = T.solve_boundary(u_all, known)
    assert res < 1e-12
    np.testing.assert_allclose(z[3], 2 * (0.3 + 5.0 / 200.0), atol=1e-12)


# ---------------------------------------------------------------------------
# Planar mechanisms: consistency with the chain formulations
# ---------------------------------------------------------------------------

def _chain_mechanism(n, g):
    mech = T.PlanarMechanism((0.0, -g))
    bars = [mech.add_bar(1.0, 1.0) for _ in range(n)]
    mech.add_hinge([(bars[0], 0)], ground=(0.0, 0.0))
    for k in range(1, n):
        mech.add_hinge([(bars[k - 1], 1), (bars[k], 0)])
    return mech


def test_mechanism_chain_equals_chain_formulations():
    """AN: the deduced overall equation of a chain equals Eqs. (5.8) and
    (7.24).

    Acceleration form vs rigid.PlanarChain and DT form vs
    dttmm.PlanarChainDT, tolerance 1e-10 (round-off only).
    """
    g = 9.81
    mech = _chain_mechanism(3, g)
    th0 = np.array([-np.pi / 3, -0.8, 0.4])
    w0 = np.array([0.3, -0.2, 0.5])
    acc = mech.accelerations(th0, w0)
    _, alpha = R.PlanarChain([R.uniform_bar(1.0, 1.0)] * 3,
                             (0.0, -g)).transfer(th0, w0)
    np.testing.assert_allclose([acc["theta"][b] for b in range(3)], alpha,
                               atol=1e-10)
    out = mech.simulate_dt(th0, w0, 0.5, 0.01)
    ref = D.PlanarChainDT([R.uniform_bar(1.0, 1.0)] * 3,
                          (0.0, -g)).simulate(th0, w0, 0.5, 0.01)
    np.testing.assert_allclose(out["theta"], ref["theta"], atol=1e-10)


# ---------------------------------------------------------------------------
# Closed four-bar loop and slider-crank
# ---------------------------------------------------------------------------

L4 = [1.0, 2.0, 1.5]
M4 = [1.0, 2.0, 1.5]


def _four_bar(g):
    mech = T.PlanarMechanism((0.0, -g))
    a, b, c = (mech.add_bar(m, ln) for m, ln in zip(M4, L4))
    mech.add_hinge([(a, 0)], ground=(0.0, 0.0))
    mech.add_hinge([(a, 1), (b, 0)])
    mech.add_hinge([(b, 1), (c, 0)])
    mech.add_hinge([(c, 1)], ground=(2.0, 0.0))
    t1 = np.pi / 3
    pa = np.array([np.cos(t1), np.sin(t1)])
    pd = np.array([2.0, 0.0])
    d = np.linalg.norm(pd - pa)
    ang = np.arctan2(*(pd - pa)[::-1])
    t2 = ang + np.arccos((L4[1] ** 2 + d * d - L4[2] ** 2) / (2 * L4[1] * d))
    pb = pa + L4[1] * np.array([np.cos(t2), np.sin(t2)])
    t3 = np.arctan2(*(pd - pb)[::-1])
    return mech, np.array([t1, t2, t3])


def _energy(mech, th, w):
    p_in, v_in = mech.kinematics(th, w)
    entry = {b: e for b, e, _ in mech.plan()}
    e = 0.0
    for b, body in enumerate(mech.bodies):
        r = R.rot2(th[b]) @ (body.center - body.ends[entry[b]])
        vc = v_in[b] + w[b] * np.array([-r[1], r[0]])
        e += 0.5 * body.mass * vc @ vc + 0.5 * body.j_c * w[b] ** 2
        e -= body.mass * (mech.gravity @ (p_in[b] + r))
    return e


def test_four_bar_energy_and_closure():
    """IV: four-bar, acceleration form + RK4 dt 1e-3, 2 s.

    Energy drift relative to sum m g L below 1e-8 and loop closure below
    1e-8 m (RK4 error level; closure is not enforced, only its derivative).
    """
    g = 9.81
    mech, th0 = _four_bar(g)
    t, th, w = mech.simulate_rk4(th0, np.zeros(3), 2.0, 1e-3)
    scale = g * sum(m * ln for m, ln in zip(M4, L4))
    assert abs(_energy(mech, th[-1], w[-1])
               - _energy(mech, th[0], w[0])) / scale < 1e-8
    p = mech.end_positions(th[-1])
    assert np.linalg.norm(p[(2, 1)] - np.array([2.0, 0.0])) < 1e-8


def test_four_bar_dttmm_houbolt_is_second_order():
    """CV: closed loop with the DT-TMM (Houbolt), order 2 +- 0.2 vs RK4
    reference.

    Newmark average acceleration is unstable for this loop (it diverges
    near the rocker dead point after about 1 s), so the loop examples use
    the Houbolt method of Sec. 7.3.2.4.
    """
    g = 9.81
    mech, th0 = _four_bar(g)
    _, th_ref, _ = mech.simulate_rk4(th0, np.zeros(3), 2.0, 2.5e-4)
    errs = []
    for dt in (0.005, 0.0025, 0.00125):
        out = mech.simulate_dt(th0, np.zeros(3), 2.0, dt, method="houbolt")
        k = int(round(dt / 2.5e-4))
        errs.append(np.max(np.abs(out["theta"] - th_ref[::k])))
        assert np.max(out["residual"]) < 1e-9
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - 2.0) < 0.2), errs


@pytest.fixture(scope="module")
def mbs():
    """Import the compiled Haug engine or skip."""
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


Z3 = [0.0, 0.0, 0.0]
AZ = [0.0, 0.0, 1.0]


def _haug_bars(mbs, s, starts, angles, lengths, masses, g):
    bodies = []
    for k, (p, th, ln, m) in enumerate(zip(starts, angles, lengths, masses)):
        c = np.asarray(p) + 0.5 * ln * np.array([np.cos(th), np.sin(th)])
        jc = m * ln * ln / 12
        b = mbs.create_mobilized_body_quat(k + 1, [c[0], c[1], 0.0], Z3, Z3,
                                           [np.cos(th / 2), 0, 0,
                                            np.sin(th / 2)], Z3, Z3, m,
                                           [1e-4, jc, jc], [0, -m * g, 0], Z3)
        s.Add(b)
        bodies.append(b)
    return bodies


def _haug_angles(s):
    return np.array([2 * np.arctan2(q[3], q[0])
                     for q in np.asarray(s.get_body_quaternions())[1:]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def test_xm_four_bar_vs_haug(mbs):
    """XM: closed four-bar, TMM acceleration form (RK4 dt 1e-3) vs Haug (RK45),
    1.5 s.

    Angles tol 1e-6 rad: both integrators are converged to below 1e-8 here.
    """
    g = 9.81
    mech, th0 = _four_bar(g)
    s = mbs.DynamicsSystem(1e-3)
    gr = mbs.create_ground(0)
    s.Add(gr)
    pa = L4[0] * np.array([np.cos(th0[0]), np.sin(th0[0])])
    pb = pa + L4[1] * np.array([np.cos(th0[1]), np.sin(th0[1])])
    a, b, c = _haug_bars(mbs, s, [np.zeros(2), pa, pb], th0, L4, M4, g)
    s.Add(mbs.create_joint(1, Z3, [-0.5 * L4[0], 0, 0], Z3, Z3, AZ, AZ, gr, a))
    s.Add(mbs.create_joint(1, [0.5 * L4[0], 0, 0], [-0.5 * L4[1], 0, 0], Z3,
                           Z3, AZ, AZ, a, b))
    s.Add(mbs.create_joint(1, [0.5 * L4[1], 0, 0], [-0.5 * L4[2], 0, 0], Z3,
                           Z3, AZ, AZ, b, c))
    s.Add(mbs.create_joint(1, [0.5 * L4[2], 0, 0], [2.0, 0, 0], Z3, Z3, AZ, AZ,
                           c, gr))
    s.set_integrator("rk45")
    s.init()
    t, th, _ = mech.simulate_rk4(th0, np.zeros(3), 1.5, 1e-3)
    worst = 0.0
    for i in range(1500):
        s.solve()
        if (i + 1) % 100 == 0:
            worst = max(worst,
                        np.max(np.abs(_wrap(_haug_angles(s) - th[i + 1]))))
    assert worst < 1e-6


def test_xm_slider_crank_vs_haug(mbs):
    """XM: planar slider-crank with a massive slider, TMM (RK4 dt 1e-3) vs
    Haug, 1.5 s.

    Crank 0.3 m / 2 kg pinned at the origin, rod 0.8 m / 1 kg, slider 1.5 kg
    on the x axis; released from rest at 60 degrees under gravity. Slider
    position tol 1e-7 m, angles tol 1e-6 rad.
    """
    g = 9.81
    l1, l2, m1, m2, m3 = 0.3, 0.8, 2.0, 1.0, 1.5
    mech = T.PlanarMechanism((0.0, -g))
    a = mech.add_bar(m1, l1)
    b = mech.add_bar(m2, l2)
    mech.add_hinge([(a, 0)], ground=(0.0, 0.0))
    mech.add_hinge([(a, 1), (b, 0)])
    mech.add_slider((b, 1), (0.0, 0.0), (1.0, 0.0), mass=m3)
    t1 = np.pi / 3
    t2 = -np.arcsin(l1 * np.sin(t1) / l2)
    th0 = np.array([t1, t2])
    s = mbs.DynamicsSystem(1e-3)
    gr = mbs.create_ground(0)
    s.Add(gr)
    pa = l1 * np.array([np.cos(t1), np.sin(t1)])
    pc = pa + l2 * np.array([np.cos(t2), np.sin(t2)])
    ba, bb = _haug_bars(mbs, s, [np.zeros(2), pa], th0, [l1, l2], [m1, m2], g)
    sl = mbs.create_mobilized_body_quat(3, [pc[0], 0.0, 0.0], Z3, Z3,
                                        [1, 0, 0, 0], Z3, Z3, m3,
                                        [0.01, 0.01, 0.01], [0, -m3 * g, 0],
                                        Z3)
    s.Add(sl)
    s.Add(mbs.create_joint(1, Z3, [-0.5 * l1, 0, 0], Z3, Z3, AZ, AZ, gr, ba))
    s.Add(mbs.create_joint(1, [0.5 * l1, 0, 0], [-0.5 * l2, 0, 0], Z3, Z3, AZ,
                           AZ, ba, bb))
    s.Add(mbs.create_joint(1, [0.5 * l2, 0, 0], Z3, Z3, Z3, AZ, AZ, bb, sl))
    s.Add(mbs.create_joint(2, Z3, [pc[0], 0, 0], Z3, Z3, [1, 0, 0], [1, 0, 0],
                           sl, gr))
    s.set_integrator("rk45")
    s.init()
    t, th, _ = mech.simulate_rk4(th0, np.zeros(2), 1.5, 1e-3)
    worst_a = worst_x = 0.0
    for i in range(1500):
        s.solve()
        if (i + 1) % 100 == 0:
            worst_a = max(worst_a, np.max(np.abs(
                _wrap(_haug_angles(s)[:2] - th[i + 1]))))
            x_tmm = mech.end_positions(th[i + 1])[(1, 1)][0]
            x_haug = np.asarray(s.get_body_positions())[3][0]
            worst_x = max(worst_x, abs(x_haug - x_tmm))
    assert worst_a < 1e-6
    assert worst_x < 1e-7
    # The DT-TMM (Houbolt) slider history converges to the same motion.
    out = mech.simulate_dt(th0, np.zeros(2), 1.5, 1e-3, method="houbolt")
    assert np.max(np.abs(out["theta"] - th[::1])) < 1e-3


# ---------------------------------------------------------------------------
# Book examples 11.2, 11.7, 11.10 (BK) and a tree cross-check (XM)
# ---------------------------------------------------------------------------

def example_11_2():
    """Closed loop of Fig. 11.11: bars 2, 4, 6, 8 (1 m, 1 kg) and bar 10.

    Bar 10 is sqrt(3) m long with J_I = sqrt(3) kg m^2 (the book names bar 4;
    with 1 kg and 1 m it cannot have that inertia), so it is a uniform bar of
    mass sqrt(3) kg.
    """
    m = T.PlanarMechanism((0.0, -G_BOOK))
    b2, b4, b6, b8 = (m.add_bar(1.0, 1.0) for _ in range(4))
    b10 = m.add_bar(np.sqrt(3.0), np.sqrt(3.0))
    m.add_hinge([(b2, 0)], ground=(0.0, 0.0))
    m.add_hinge([(b2, 1), (b4, 0), (b10, 1)])
    m.add_hinge([(b4, 1), (b6, 0)])
    m.add_hinge([(b6, 1), (b8, 0)])
    m.add_hinge([(b8, 1), (b10, 0)])
    return m, np.array([-np.pi / 3, -np.pi / 3, -np.pi / 3, -np.pi, np.pi / 2])


def example_11_7():
    """Branched pendulum of Fig. 11.22: bars 2, 4, 6 and bar 7 at hinge 3."""
    m = T.PlanarMechanism((0.0, -G_BOOK))
    b2, b4, b6, b7 = (m.add_bar(1.0, 1.0) for _ in range(4))
    m.add_hinge([(b2, 0)], ground=(0.0, 0.0))
    m.add_hinge([(b2, 1), (b4, 0), (b7, 0)])
    m.add_hinge([(b4, 1), (b6, 0)])
    return m, np.array([-np.pi / 3, -np.pi / 3, -np.pi / 3, -2 * np.pi / 3])


def example_11_10():
    """Network of Fig. 11.31: loop 2-4-6-8 at hinge 1, bars 9 and 10 hanging.

    The angle of bar 10 (pi) is the direction from its free end to hinge 5;
    with this reading the bar points to the right as drawn in Fig. 11.31.
    """
    m = T.PlanarMechanism((0.0, -G_BOOK))
    b2, b4, b6, b8, b9, b10 = (m.add_bar(1.0, 1.0) for _ in range(6))
    m.add_hinge([(b2, 0), (b8, 1)], ground=(0.0, 0.0))
    m.add_hinge([(b2, 1), (b4, 0), (b9, 0)])
    m.add_hinge([(b4, 1), (b6, 0), (b10, 1)])
    m.add_hinge([(b6, 1), (b8, 0)])
    return (m,
            np.array([-np.pi / 3, -2 * np.pi / 3, 2 * np.pi / 3, np.pi / 3,
                      0.0, np.pi]))


BK_CASES = [
    (example_11_2, "tmm_ch11_fig11_12.json",
     ["theta2", "theta4", "theta6", "theta8", "theta10"], 12.0),
    (example_11_7, "tmm_ch11_fig11_23.json",
     ["theta2", "theta4", "theta6", "theta7"], 12.0),
    (example_11_10, "tmm_ch11_fig11_32.json",
     ["theta2", "theta4", "theta6", "theta8", "theta9", "theta10"], 5.5),
]
_BK_RUNS = {}


def _bk_run(case):
    """Simulate a Ch. 11 book example once (DT-TMM, Houbolt, dT 2.5e-3)."""
    builder, fname, names, t_end = case
    if fname not in _BK_RUNS:
        mech, th0 = builder()
        out = mech.simulate_dt(th0, np.zeros(len(th0)), t_end, 2.5e-3,
                               method="houbolt")
        _BK_RUNS[fname] = (mech, th0, out)
    return _BK_RUNS[fname]


def _excess_px(book, name, k, out, t_read):
    """Distance of the simulation outside the read band, minus the read
    uncertainty (px)."""
    curve = book["curves"][name]
    i = curve["t"].index(t_read)
    v = np.interp(t_read, out["t"], out["theta"][:, k])
    ex = max(curve["low"][i] - v, v - curve["high"][i]) * curve["px_per_unit"]
    return ex - book["read_uncertainty_px"]


@pytest.mark.parametrize("case", BK_CASES,
                         ids=["fig11_12", "fig11_23", "fig11_32"])
def test_book_examples_chapter_11(case):
    """BK Figs. 11.12 (p. 450), 11.23 (p. 468), 11.32 (p. 485), 2 px beyond the
    band.

    DT-TMM with the Houbolt method, dT = 2.5e-3; the loop residual of
    every step is below 1e-9. Points listed in ``outside_band`` of the data
    file are tested separately as expected failures.
    """
    builder, fname, names, _ = case
    book = json.loads((DATA / fname).read_text())
    mech, th0, out = _bk_run(case)
    closure = mech.end_positions(th0)
    for h in mech.hinges:
        pts = [closure[e] for e in h["ends"]]
        if h["ground"] is not None:
            pts.append(h["ground"])
        assert max(np.linalg.norm(p - pts[0]) for p in pts) < 1e-12
    assert np.max(out["residual"]) < 1e-9
    known = [tuple(p) for p in book.get("outside_band", {}).get("points", [])]
    for k, name in enumerate(names):
        for t in book["curves"][name]["t"]:
            if (name, t) in known:
                continue
            assert _excess_px(book, name, k, out, t) <= 0.0, (name, t)


_XFAIL_POINTS = []
for _case in BK_CASES:
    _book = json.loads((DATA / _case[1]).read_text())
    _ob = _book.get("outside_band")
    if _ob:
        for (_name, _t), _ex in zip(_ob["points"], _ob["measured_excess_px"]):
            _XFAIL_POINTS.append(pytest.param(
                _case, _name, _t, id="%s-%s-%g" % (_case[1][9:17], _name, _t),
                marks=pytest.mark.xfail(strict=True, reason=(
                    "simulation %.2f px outside the 2 px read band of %s at t "
                    "= %g s; no independent analytic curve in the book for "
                    "this point"
                    % (_ex, _name, _t)))))


@pytest.mark.parametrize("case,name,t", _XFAIL_POINTS)
def test_book_examples_chapter_11_points_outside_band(case, name, t):
    """BK: read points of Figs. 11.12 and 11.23 that lie outside the read
    band."""
    book = json.loads((DATA / case[1]).read_text())
    _, _, out = _bk_run(case)
    assert _excess_px(book, name, case[2].index(name), out, t) <= 0.0


def test_xm_branched_pendulum_11_7_vs_haug(mbs):
    """XM: tree of Example 11.7, TMM acceleration form (RK4 dt 1e-3) vs Haug, 1
    s.

    Angles tol 1e-6 rad.
    """
    g = G_BOOK
    mech, th0 = example_11_7()
    s = mbs.DynamicsSystem(1e-3)
    gr = mbs.create_ground(0)
    s.Add(gr)
    p3 = np.array([np.cos(th0[0]), np.sin(th0[0])])
    p5 = p3 + np.array([np.cos(th0[1]), np.sin(th0[1])])
    b2, b4, b6, b7 = _haug_bars(mbs, s, [np.zeros(2), p3, p5, p3], th0,
                                [1.0] * 4, [1.0] * 4, g)
    s.Add(mbs.create_joint(1, Z3, [-0.5, 0, 0], Z3, Z3, AZ, AZ, gr, b2))
    s.Add(mbs.create_joint(1, [0.5, 0, 0], [-0.5, 0, 0], Z3, Z3, AZ, AZ, b2,
                           b4))
    s.Add(mbs.create_joint(1, [0.5, 0, 0], [-0.5, 0, 0], Z3, Z3, AZ, AZ, b4,
                           b6))
    s.Add(mbs.create_joint(1, [0.5, 0, 0], [-0.5, 0, 0], Z3, Z3, AZ, AZ, b2,
                           b7))
    s.set_integrator("rk45")
    s.init()
    t, th, _ = mech.simulate_rk4(th0, np.zeros(4), 1.0, 1e-3)
    worst = 0.0
    for i in range(1000):
        s.solve()
        if (i + 1) % 100 == 0:
            worst = max(worst,
                        np.max(np.abs(_wrap(_haug_angles(s) - th[i + 1]))))
    assert worst < 1e-6

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
