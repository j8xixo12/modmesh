# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of the DT-TMM for planar multi-rigid-flexible-body systems (book Ch.
8).

Check types (docs/roadmap.md Sec. 4.2): AN analytic, XM cross-method
(linear TMM of solvcon.multibody.tmm.linear / library, rigid DT-TMM), IV
invariant, CV convergence order, BK book figure.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from solvcon.multibody.tmm import beams_large as BL
from solvcon.multibody.tmm import library as LB
from solvcon.multibody.tmm import linear as L
from solvcon.multibody.tmm import rigid as R
from solvcon.multibody.tmm.topology import AffineSystem

DATA = Path(__file__).parent / "data" / "multibody"
G_BOOK = 9.8


def _example_8_1(n=3):
    """Fig. 8.6: rigid body 2, beam 4 (EI 2.22 N m^2), rigid body 6; 1 m, 1 kg
    each."""
    return BL.PlanarFlexibleChain([R.uniform_bar(1.0, 1.0),
                                   BL.PlanarBeam(1.0, 1.0, 2.22, n),
                                   R.uniform_bar(1.0, 1.0)],
                                  gravity=(0.0, -G_BOOK))


def _example_8_2(n=3):
    """Fig. 8.8: crank 1 (10 cm) at 30 rad/s, beam 3 and rigid body 5 (30 cm),
    no gravity."""
    mu = 2.7e3 * 4.0e-5
    bodies = [R.uniform_bar(mu * 0.1, 0.1),
              BL.PlanarBeam(0.3, mu * 0.3, 0.931, n),
              R.uniform_bar(mu * 0.3, 0.3)]
    return BL.PlanarFlexibleChain(bodies,
                                  hinges=[("driven", 0.0, 30.0), "smooth",
                                          "smooth"], gravity=(0.0, 0.0),
                                  tip=(0.4, 0.0))


def _linear_frequencies(chain, n):
    """Small-amplitude frequencies of a one-beam chain from the acceleration
    form."""
    def acc(y):
        r = chain.accelerations([y[0]], [0.0], [y[1:]], [np.zeros(n)])
        return np.concatenate([[r["theta"][0]], r["q"][0]])

    jac = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        e = np.zeros(n + 1)
        e[i] = 1e-6
        jac[:, i] = (acc(e) - acc(-e)) / 2e-6
    return np.sort(np.sqrt(np.abs(np.linalg.eigvals(jac).real)))


def _orders(errs):
    e = np.asarray(errs)
    return np.log2(e[:-1] / e[1:])


# ---------------------------------------------------------------------------
# Analytic limits and cross-checks
# ---------------------------------------------------------------------------

def test_pinned_free_beam_frequencies_match_linear_tmm():
    """XM: hinged-free beam, sine modes (Eq. 8.69) vs linear TMM
    (solvcon.multibody.tmm.linear).

    Rayleigh-Ritz: every frequency is an upper bound, and the linear TMM
    value lies within twice the change of the Ritz value from 16 to 32
    modes (error estimate by doubling the basis), modes 1-3.
    """
    ei, m, length = 2.22, 1.0, 1.0
    w_lin = L.Chain([LB.EulerBernoulliBeam(length, ei, m)], [0, 2],
                    [2, 3]).natural_frequencies(200.0)[1:4]
    w = {}
    for n in (16, 32):
        ch = BL.PlanarFlexibleChain([BL.PlanarBeam(length, m, ei, n)],
                                    gravity=(0.0, 0.0))
        w[n] = _linear_frequencies(ch, n)[1:4]
    assert np.all(w[16] >= w_lin * (1 - 1e-12))
    assert np.all(w[32] >= w_lin * (1 - 1e-12))
    assert np.all(np.abs(w[16] - w_lin) <= 2.0 * np.abs(w[16] - w[32]))


def test_clamped_beam_converges_first_order_in_modes():
    """XM + CV: fixed hinge (Sec. 8.7) at the ground, clamped-free beam.

    The sine modes have u''(0) = 0, so the clamped root moment is not in
    the basis: the first frequency converges from above to the linear TMM
    value with order 1 +- 0.2 in the number of modes.
    """
    ei, m, length = 2.22, 1.0, 1.0
    w1 = L.Chain([LB.EulerBernoulliBeam(length, ei, m)], [0, 1],
                 [2, 3]).natural_frequencies(20.0)[0]
    errs = []
    for n in (8, 16, 32):
        ch = BL.PlanarFlexibleChain([BL.PlanarBeam(length, m, ei, n)],
                                    hinges=["fixed"], gravity=(0.0, 0.0))
        w = _linear_frequencies(ch, n)
        errs.append(w[w > 1.0][0] - w1)
    assert np.all(np.array(errs) > 0)
    assert np.all(np.abs(_orders(errs) - 1.0) < 0.2), errs


def test_simply_supported_beam_static_deflection():
    """AN: beam pinned at both ends (tip pin), uniform gravity load.

    The axial force between the two pins is redundant (minimum-norm
    solution). At the series solution q^k = -4 w l^4 / (EI (k pi)^5) (odd k)
    every acceleration is zero (round-off, tol 1e-9 of g), and the mid
    deflection equals 5 w l^4 / (384 EI) within the series tail bound.
    """
    g, mu, ei, length, n = 9.81, 2.0, 50.0, 1.5, 15
    beam = BL.PlanarBeam(length, mu * length, ei, n)
    ch = BL.PlanarFlexibleChain([beam], gravity=(0.0, -g), tip=(length, 0.0),
                                redundant=True)
    k = np.arange(1, n + 1)
    w = mu * g
    q = np.where(k % 2 == 1, -4 * w * length ** 4 / (ei * (k * np.pi) ** 5),
                 0.0)
    res = ch.accelerations([0.0], [0.0], [q], [np.zeros(n)])
    assert abs(res["theta"][0]) < 1e-9 * g
    assert np.max(np.abs(res["q"][0])) < 1e-9 * g
    mid = beam.deflection(0.5 * length, q)[0]
    exact = -5 * w * length ** 4 / (384 * ei)
    tail = sum(4 * w * length ** 4 / (ei * (kk * np.pi) ** 5) for kk
               in range(n + 1, 4000, 2))
    assert abs(mid - exact) <= tail * 1.0000001


def test_rigid_limit_equals_rigid_chain():
    """AN: as EI -> infinity the beam chain tends to the rigid chain of Ch. 5.

    Acceleration form with RK4 (dt 2.5e-4, 0.3 s) for both models. The
    difference comes from the elastic deflection and scales with 1/EI: from
    EI = 1e3 to 1e4 it drops by 10 +- 30 %.
    """
    th0 = [-np.pi / 3] * 3
    _, ref, _ = R.PlanarChain([R.uniform_bar(1.0, 1.0)] * 3,
                              (0.0, -G_BOOK)).simulate(th0, [0.0] * 3, 0.3,
                                                       2.5e-4)
    diffs = []
    for ei in (1e3, 1e4):
        ch = BL.PlanarFlexibleChain([R.uniform_bar(1.0, 1.0),
                                     BL.PlanarBeam(1.0, 1.0, ei, 3),
                                     R.uniform_bar(1.0, 1.0)],
                                    gravity=(0.0, -G_BOOK))
        out = ch.simulate_rk4(th0, [0.0] * 3, 0.3, 2.5e-4)
        diffs.append(np.max(np.abs(out["theta"] - ref)))
    assert 7.0 < diffs[0] / diffs[1] < 13.0


def test_smooth_hinge_beam_matrix_removes_angle_and_modes():
    """AN: Eq. (8.98) gives theta and q^k with zero vibration residual and zero
    m_O."""
    n = 3
    beam = BL.PlanarBeam(1.0, 1.0, 2.22, n)
    ch = BL.PlanarFlexibleChain([beam], gravity=(0.0, -G_BOOK))
    ctx = {"theta": np.array([-0.7]), "omega": np.array([0.4]),
           "q": [np.array([0.01, -0.002, 0.001])],
           "qd": [np.array([0.1, 0.0, -0.05])]}
    coef = ch._coefficients(0, "acc", ctx)
    sysm = AffineSystem()
    z = sysm.unknown(6 + n)  # [x, y, theta, m, qx, qy, q1..q3]

    def rows(idx):
        return type(z)(z.m[idx], z.c[idx])

    _, _, m_out, _ = ch._beam_relations(sysm, beam, rows([0, 1]), rows([2]),
                                        rows(list(range(6, 6 + n))), rows([3]),
                                        rows([4, 5]), coef, "acc")
    modal = sysm.rows[0]
    beam_rows = np.hstack([np.vstack([modal.width(6 + n), m_out.width(6 + n)]),
                           np.concatenate([modal.c, m_out.c])[:, None]])
    u = BL.smooth_hinge_beam_matrix(beam_rows, n)
    z_in = np.array([0.3, -0.2, 0.0, 0.0, 1.5, -4.0, 0.0, 0.0, 0.0, 1.0])
    z_out = u @ z_in
    np.testing.assert_allclose(beam_rows @ z_out, 0.0, atol=1e-10)
    np.testing.assert_allclose(z_out[[0, 1, 4, 5]], z_in[[0, 1, 4, 5]])


# ---------------------------------------------------------------------------
# Invariants and convergence
# ---------------------------------------------------------------------------

def _final_energy(ch, out):
    q = [None if k not in out["q"] else out["q"][k][-1] for k
         in range(len(ch.bodies))]
    qd = [None if k not in out["qd"] else out["qd"][k][-1] for k
          in range(len(ch.bodies))]
    return ch.energy(out["theta"][-1], out["omega"][-1], q, qd)


def test_energy_conservation_rk4_is_fourth_order():
    """IV + CV: Example 8.1 system, energy drift of the RK4 reference over 1 s.

    No damping: the drift is the RK4 error; RK4 is at least fourth order,
    so the observed order must be above 4 - 0.2.
    """
    ch = _example_8_1()
    th0 = [-np.pi / 3] * 3
    e0 = ch.energy(th0, [0.0] * 3)
    drifts = []
    for dt in (4e-3, 2e-3, 1e-3):
        out = ch.simulate_rk4(th0, [0.0] * 3, 1.0, dt)
        drifts.append(abs(_final_energy(ch, out) - e0))
    assert np.all(_orders(drifts) > 3.8), drifts


@pytest.mark.parametrize("method", ["houbolt", "newmark"])
def test_dttmm_flexible_chain_is_second_order(method):
    """CV: Example 8.1 system, DT-TMM angles and modal coordinates vs RK4, 0.5
    s.

    Order 2 +- 0.2 for the Houbolt and the Newmark (beta 1/4) methods.
    """
    ch = _example_8_1()
    th0 = [-np.pi / 3] * 3
    ref = ch.simulate_rk4(th0, [0.0] * 3, 0.5, 2.5e-4)
    errs = []
    for dt in (4e-3, 2e-3, 1e-3):
        out = ch.simulate_dt(th0, [0.0] * 3, 0.5, dt, method=method)
        k = int(round(dt / 2.5e-4))
        errs.append(max(np.max(np.abs(out["theta"] - ref["theta"][::k])),
                        np.max(np.abs(out["q"][1] - ref["q"][1][::k]))))
    assert np.all(np.abs(_orders(errs) - 2.0) < 0.2), errs


def test_spring_hinge_energy_and_dttmm_convergence():
    """IV + CV: beam on a rotary spring hinge (Sec. 8.6) at a pinned base, no
    gravity.

    The RK4 energy drift (spring energy included) drops with an order above
    4 - 0.2 (dt 2e-3 to 5e-4; the third beam mode at 200 rad/s is not
    resolved at 4e-3) and the DT-TMM (Newmark) angles converge with order
    2 +- 0.2.
    """
    beam = BL.PlanarBeam(1.0, 1.0, 5.0, 3)
    ch = BL.PlanarFlexibleChain([beam, R.uniform_bar(0.5, 0.5)],
                                hinges=["smooth", ("spring", 20.0, 0.0)],
                                gravity=(0.0, 0.0))
    th0, w0 = [0.0, 0.3], [1.0, 0.0]
    e0 = ch.energy(th0, w0)
    drifts = []
    for dt in (2e-3, 1e-3, 5e-4):
        drifts.append(abs(_final_energy(ch,
                                        ch.simulate_rk4(th0, w0, 1.0,
                                                        dt)) - e0))
    assert np.all(_orders(drifts) > 3.8), drifts
    # Short horizon: the tip body on the spring makes this motion sensitive;
    # beyond 0.2 s the error growth is no longer linear in time.
    ref = ch.simulate_rk4(th0, w0, 0.1, 1.25e-4)
    errs = []
    for dt in (1e-3, 5e-4, 2.5e-4):
        out = ch.simulate_dt(th0, w0, 0.1, dt, method="newmark")
        k = int(round(dt / 1.25e-4))
        errs.append(np.max(np.abs(out["theta"] - ref["theta"][::k])))
    assert np.all(np.abs(_orders(errs) - 2.0) < 0.2), errs


# ---------------------------------------------------------------------------
# Book examples 8.1 and 8.2 (BK)
# ---------------------------------------------------------------------------

def _excess_px(curve, t_read, t, y, pad_px):
    i = curve["t"].index(t_read)
    v = np.interp(t_read, t, y)
    return (max(curve["low"][i] - v,
                v - curve["high"][i]) * curve["px_per_unit"] - pad_px)


_RUNS = {}


def _run(name):
    if name not in _RUNS:
        if name == "8.1":
            ch = _example_8_1()
            out = ch.simulate_dt([-np.pi / 3] * 3, [0.0] * 3, 20.0, 2e-3,
                                 method="houbolt")
            u = np.array([ch.bodies[1].deflection(0.5,
                                                  q)[0] for q in out["q"][1]])
            _RUNS[name] = {"t": out["t"], "theta2": out["theta"][:, 0],
                           "theta4": out["theta"][:, 1],
                           "theta6": out["theta"][:, 2], "u_mid": u,
                           "residual": out["residual"]}
        else:
            ch = _example_8_2()
            out = ch.simulate_dt([0.0, np.pi / 3, -np.pi / 3],
                                 [30.0, -10.0, -10.0], 3.0, 1e-4,
                                 method="houbolt")
            u = 1e3 * np.array([ch.bodies[1].deflection(0.15, q)[0]
                                for q in out["q"][1]])
            _RUNS[name] = {"t": out["t"], "theta5": out["theta"][:, 2],
                           "u_mid_mm": (30.0 * out["t"], u),
                           "residual": out["residual"]}
    return _RUNS[name]


BK_FILES = {"8.1": "tmm_ch8_fig8_7.json", "8.2": "tmm_ch8_fig8_9.json"}


def _series(run, name):
    val = run[name]
    return val if isinstance(val, tuple) else (run["t"], val)


@pytest.mark.parametrize("example", ["8.1", "8.2"])
def test_book_examples_chapter_8(example):
    """BK Figs. 8.7 (p. 321) and 8.9 (p. 323), 2 px beyond the read band.

    DT-TMM with the Houbolt method (Newmark diverges for these systems,
    as for the rigid chains of Ch. 7): Example 8.1 dT 2e-3, Example 8.2
    dT 1e-4. Points listed in ``outside_band`` are tested separately.
    """
    book = json.loads((DATA / BK_FILES[example]).read_text())
    run = _run(example)
    known = [tuple(p) for p in book.get("outside_band", {}).get("points", [])]
    for name, curve in book["curves"].items():
        t, y = _series(run, name)
        for tr in curve["t"]:
            if (name, tr) in known:
                continue
            assert _excess_px(curve, tr, t, y,
                              book["read_uncertainty_px"]) <= 0.0, (name, tr)


def _outside_points():
    out = []
    for example, fname in BK_FILES.items():
        book = json.loads((DATA / fname).read_text())
        ob = book.get("outside_band")
        if not ob:
            continue
        for (name, tr), ex in zip(ob["points"], ob["measured_excess_px"]):
            reason = ("%.2f px outside the 2 px read band; %s"
                      % (ex, ob["note"]))
            out.append(pytest.param(
                example, name, tr, id="%s-%s-%g" % (example, name, tr),
                marks=pytest.mark.xfail(strict=True, reason=reason)))
    return out


@pytest.mark.parametrize("example,name,t_read", _outside_points())
def test_book_examples_chapter_8_points_outside_band(example, name, t_read):
    """BK: read points of Figs. 8.7 and 8.9 outside the read band (expected
    failures)."""
    book = json.loads((DATA / BK_FILES[example]).read_text())
    t, y = _series(_run(example), name)
    assert _excess_px(book["curves"][name], t_read, t, y,
                      book["read_uncertainty_px"]) <= 0.0


# ---------------------------------------------------------------------------
# Clamped-compatible basis (Sec. 8.4: modes that satisfy the boundary
# condition) #
# ---------------------------------------------------------------------------

def test_cantilever_basis_gives_exact_clamped_frequencies():
    """XM: fixed base hinge with the clamped-free eigenfunctions as modes.

    The basis is the exact eigenfunction set of the linearized problem, so
    the first four frequencies equal the linear TMM values to 1e-8
    relative (root finding and quadrature round-off).
    """
    ei, m, length = 2.22, 1.0, 1.0
    w_lin = L.Chain([LB.EulerBernoulliBeam(length, ei, m)], [0, 1],
                    [2, 3]).natural_frequencies(700.0)[:4]
    ch = BL.PlanarFlexibleChain([BL.PlanarBeam(length, m, ei, 4,
                                               basis="cantilever")],
                                hinges=["fixed"], gravity=(0.0, 0.0))
    w = _linear_frequencies(ch, 4)
    np.testing.assert_allclose(w[w > 1.0][:4], w_lin, rtol=1e-8)


# ---------------------------------------------------------------------------
# Spatial beams (Secs. 8.8-8.12)
# ---------------------------------------------------------------------------

def _cube():
    """Rigid bodies 2 and 6 of Example 8.3 (as in Example 7.2)."""
    return R.SpatialBody(1.0, [1 / 6, 5 / 12, 5 / 12], [1.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0])


def _bar3():
    """Uniform bar 1 m, 1 kg, small axial inertia, as a spatial rigid body."""
    return R.SpatialBody(1.0, [1e-3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0])


def _spatial_freqs(ch, nq):
    """Small-amplitude frequencies of a one-beam spatial chain."""
    ns = 3 + nq

    def acc(y):
        r = ch.accelerations([y[:3]], [np.zeros(3)], [y[3:]], [np.zeros(nq)])
        return np.concatenate([r["theta"][0], r["q"][0]])

    jac = np.zeros((ns, ns))
    for i in range(ns):
        e = np.zeros(ns)
        e[i] = 1e-6
        jac[:, i] = (acc(e) - acc(-e)) / 2e-6
    return np.sort(np.sqrt(np.abs(np.linalg.eigvals(jac).real)))


def test_spatial_beam_reduces_to_planar_beam():
    """AN: planar motion of the spatial chain equals the planar chain, tol
    1e-10.

    Accelerations at one state and an RK4 trajectory (dt 1e-3, 0.3 s); the
    out-of-plane modal coordinates stay zero.
    """
    pb = BL.PlanarBeam(1.0, 1.0, 2.22, 3)
    sb = BL.SpatialBeam(1.0, 1.0, 2.22, n_modes=3, j_axial=1e-3)
    pc = BL.PlanarFlexibleChain([R.uniform_bar(1.0, 1.0), pb,
                                 R.uniform_bar(1.0, 1.0)],
                                gravity=(0.0, -G_BOOK))
    sc = BL.SpatialFlexibleChain([_bar3(), sb, _bar3()],
                                 gravity=(0.0, -G_BOOK, 0.0))
    th, w = np.array([-1.0, -0.5, 0.3]), np.array([0.2, -0.4, 0.5])
    q, qd = np.array([0.01, -0.003, 0.002]), np.array([0.1, 0.05, -0.02])
    ra = pc.accelerations(th, w, [None, q, None], [None, qd, None])
    t3 = np.zeros((3, 3))
    t3[:, 2] = th
    w3 = np.zeros((3, 3))
    w3[:, 2] = w
    zero = np.zeros(3)
    rs = sc.accelerations(t3, w3, [None, np.concatenate([q, zero]), None],
                          [None, np.concatenate([qd, zero]), None])
    np.testing.assert_allclose([x[2] for x in rs["theta"]], ra["theta"],
                               atol=1e-10)
    np.testing.assert_allclose(rs["q"][1][:3], ra["q"][1], atol=1e-10)
    np.testing.assert_allclose(rs["q"][1][3:], 0.0, atol=1e-10)
    t0 = np.zeros((3, 3))
    t0[:, 2] = -np.pi / 3
    out_s = sc.simulate_rk4(t0, np.zeros((3, 3)), 0.3, 1e-3)
    out_p = pc.simulate_rk4([-np.pi / 3] * 3, [0.0] * 3, 0.3, 1e-3)
    np.testing.assert_allclose(out_s["theta"][:, :, 2], out_p["theta"],
                               atol=1e-10)
    np.testing.assert_allclose(out_s["q"][1][:, :3], out_p["q"][1], atol=1e-10)


def test_spatial_hinged_free_frequencies_match_linear_tmm():
    """XM: ball-hinged free beam, both bending planes vs linear TMM
    (hinged-free).

    Each linear frequency appears twice; Ritz upper bound and the doubling
    error estimate of the planar test (16 and 32 modes per plane).
    """
    ei, m, length = 2.22, 1.0, 1.0
    w_lin = L.Chain([LB.EulerBernoulliBeam(length, ei, m)], [0, 2],
                    [2, 3]).natural_frequencies(200.0)[1:3]
    w = {}
    for n in (16, 32):
        ch = BL.SpatialFlexibleChain([BL.SpatialBeam(length, m, ei, n_modes=n,
                                                     j_axial=1e-3)],
                                     gravity=(0.0, 0.0, 0.0))
        f = _spatial_freqs(ch, 2 * n)
        f = f[f > 1.0]
        w[n] = f[:4].reshape(2, 2)  # pairs (v, w) of modes 1 and 2
    for k in range(2):
        assert np.all(w[16][k] >= w_lin[k] * (1 - 1e-12))
        assert np.all(np.abs(w[16][k] - w_lin[k])
                      <= 2.0 * np.abs(w[16][k] - w[32][k]))


def test_spatial_clamped_beam_cantilever_basis():
    """XM: spatial beam clamped at the ground (Sec. 8.10) with the cantilever
    basis.

    The first three frequencies of both planes equal the linear TMM
    clamped-free values to 1e-8 relative (exact eigenfunctions).
    """
    ei, m, length = 2.22, 1.0, 1.0
    w_lin = L.Chain([LB.EulerBernoulliBeam(length, ei, m)], [0, 1],
                    [2, 3]).natural_frequencies(300.0)[:3]
    ch = BL.SpatialFlexibleChain([BL.SpatialBeam(length, m, ei, n_modes=4,
                                                 basis="cantilever")],
                                 hinges=["fixed"], gravity=(0.0, 0.0, 0.0))

    def acc(y):
        r = ch.accelerations([np.zeros(3)], [np.zeros(3)], [y], [np.zeros(8)])
        return r["q"][0]

    jac = np.zeros((8, 8))
    for i in range(8):
        e = np.zeros(8)
        e[i] = 1e-6
        jac[:, i] = (acc(e) - acc(-e)) / 2e-6
    f = np.sort(np.sqrt(np.abs(np.linalg.eigvals(jac).real)))
    np.testing.assert_allclose(f[:6].reshape(3, 2),
                               np.column_stack([w_lin, w_lin]), rtol=1e-8)


def test_spatial_rigid_limit_equals_rigid_chain():
    """AN: spatial beam chain tends to the rigid spatial chain of Ch. 5 as
    1/EI.

    RK4 dt 2.5e-4, 0.3 s, out-of-plane motion; the angle difference drops
    by 10 +- 30 % from EI = 1e3 to 1e4.
    """
    w0 = np.zeros((3, 3))
    w0[2, 1] = 0.5
    ref = R.SpatialChain([_bar3(), _bar3(), _bar3()], [R.BallHinge()] * 3,
                         gravity=(0.0, -G_BOOK, 0.0)).simulate(
        [[1, 0, 0, 0]] * 3, w0, 0.3, 2.5e-4)
    ang = np.array([[R.angles_xyz(R.quat_to_matrix(q)) for q in qq]
                    for qq in ref["quat"]])
    diffs = []
    for ei in (1e3, 1e4):
        ch = BL.SpatialFlexibleChain(
            [_bar3(), BL.SpatialBeam(1.0, 1.0, ei, n_modes=3, j_axial=1e-3),
             _bar3()],
            gravity=(0.0, -G_BOOK, 0.0))
        out = ch.simulate_rk4(np.zeros((3, 3)), w0, 0.3, 2.5e-4)
        diffs.append(np.max(np.abs(out["theta"] - ang)))
    assert 7.0 < diffs[0] / diffs[1] < 13.0


def _soft_spatial():
    beam = BL.SpatialBeam(1.0, 1.32, 20.0, n_modes=3, j_axial=8.8e-5)
    return BL.SpatialFlexibleChain([_cube(), beam, _cube()],
                                   gravity=(0.0, -G_BOOK, 0.0))


def test_spatial_energy_rk4_and_dttmm_orders():
    """IV + CV: Example 8.3 layout with EI = 20 N m^2.

    RK4 energy drift over 0.5 s with order above 4 - 0.2. DT-TMM over
    0.25 s, dT 1e-3 to 2.5e-4: Newmark angles and modal coordinates, and
    Houbolt angles, order 2 +- 0.2. (The Houbolt modal coordinates are not
    asymptotic at these steps: the numerical damping of the third mode,
    350 rad/s, dominates.)
    """
    ch = _soft_spatial()
    th0 = np.zeros((3, 3))
    thd0 = np.zeros((3, 3))
    thd0[2, 1] = 0.5
    e0 = ch.energy(th0, thd0)
    drifts = []
    for dt in (2e-3, 1e-3, 5e-4):
        out = ch.simulate_rk4(th0, thd0, 0.5, dt)
        q = [None, out["q"][1][-1], None]
        qd = [None, out["qd"][1][-1], None]
        drifts.append(abs(ch.energy(out["theta"][-1], out["theta_d"][-1], q,
                                    qd) - e0))
    assert np.all(_orders(drifts) > 3.8), drifts
    ref = ch.simulate_rk4(th0, thd0, 0.25, 1.25e-4)
    for method in ("newmark", "houbolt"):
        errs = []
        for dt in (1e-3, 5e-4, 2.5e-4):
            out = ch.simulate_dt(th0, thd0, 0.25, dt, method=method)
            k = int(round(dt / 1.25e-4))
            e = np.max(np.abs(out["theta"] - ref["theta"][::k]))
            if method == "newmark":
                e = max(e, np.max(np.abs(out["q"][1] - ref["q"][1][::k])))
            errs.append(e)
        assert np.all(np.abs(_orders(errs) - 2.0) < 0.2), (method, errs)


_SP_RUN = {}


def _example_8_3_run():
    if not _SP_RUN:
        ei = 2.5e10 * 8.8e-5 / (2 * 3.3e3 * 1.0)
        beam = BL.SpatialBeam(1.0, 1.32, ei, n_modes=3, j_axial=8.8e-5)
        ch = BL.SpatialFlexibleChain([_cube(), beam, _cube()],
                                     gravity=(0.0, -G_BOOK, 0.0))
        thd0 = np.zeros((3, 3))
        thd0[2, 1] = 0.1
        out = ch.simulate_dt(np.zeros((3, 3)), thd0, 9.0, 1e-3,
                             method="houbolt")
        defl = np.array([beam.deflection(0.5, q) for q in out["q"][1]])[:, :,
                                                                        0]
        _SP_RUN.update({"t": out["t"], "v_mid": defl[:, 0],
                        "w_mid": defl[:, 1], "theta_z2": out["theta"][:, 0, 2],
                        "theta_z4": out["theta"][:, 1, 2],
                        "theta_z6": out["theta"][:, 2, 2],
                        "residual": out["residual"]})
    return _SP_RUN


def test_book_example_8_3_figure_8_11():
    """BK Fig. 8.11 (pp. 324-325), 2 px beyond the read band.

    EI = E J_4x / (2 rho l) = 333.3 N m^2 (circular section, J_4 of
    Example 8.3), 3 sine modes per plane, Houbolt dT 1e-3. Points listed in
    ``outside_band`` are tested separately.
    """
    book = json.loads((DATA / "tmm_ch8_fig8_11.json").read_text())
    run = _example_8_3_run()
    assert np.max(run["residual"]) < 1e-9
    known = [tuple(p) for p in book["outside_band"]["points"]]
    for name, curve in book["curves"].items():
        for tr in curve["t"]:
            if (name, tr) in known:
                continue
            assert _excess_px(curve, tr, run["t"], run[name], 2) <= 0.0, (
                name, tr)


_FIG811 = json.loads(
    (DATA / "tmm_ch8_fig8_11.json").read_text())["outside_band"]


@pytest.mark.parametrize("name,t_read", [
    pytest.param(n, t, id="%s-%g" % (n, t), marks=pytest.mark.xfail(
        strict=True, reason=("%.2f px outside the 2 px read band; %s"
                             % (e, _FIG811["note"]))))
    for (n, t), e in zip(_FIG811["points"], _FIG811["measured_excess_px"])])
def test_book_example_8_3_points_outside_band(name, t_read):
    """BK: read points of Fig. 8.11 outside the read band (expected
    failures)."""
    book = json.loads((DATA / "tmm_ch8_fig8_11.json").read_text())
    run = _example_8_3_run()
    assert _excess_px(book["curves"][name], t_read, run["t"], run[name],
                      2) <= 0.0


# ---------------------------------------------------------------------------
# Interior fixed and spring hinges on spatial beams (Secs. 8.10-8.12, 15.29)
# ---------------------------------------------------------------------------

def _chain_freqs(ch):
    """Small-amplitude frequencies of a spatial chain about the rest state."""
    sizes = [2 * b.n if ch._is_beam(b) else 0 for b in ch.bodies]
    nb = len(ch.bodies)
    ns = 3 * nb + sum(sizes)

    def acc(y):
        th = [y[3 * i:3 * i + 3] for i in range(nb)]
        qs, i = [], 3 * nb
        for s in sizes:
            qs.append(y[i:i + s] if s else None)
            i += s
        zq = [None if x is None else np.zeros_like(x) for x in qs]
        r = ch.accelerations(th, [np.zeros(3)] * nb, qs, zq)
        return np.concatenate(r["theta"]
                              + [x for x in r["q"] if x is not None])

    jac = np.zeros((ns, ns))
    for i in range(ns):
        e = np.zeros(ns)
        e[i] = 1e-6
        jac[:, i] = (acc(e) - acc(-e)) / 2e-6
    f = np.sort(np.sqrt(np.abs(np.linalg.eigvals(jac).real)))
    return f[f > 0.5]


def _poly_beam(length, ei, n):
    return BL.SpatialBeam(length, length, ei, n_modes=n, j_axial=1e-3,
                          basis="polynomial")


def _tip(ch, out, i):
    th = out["theta"][i]
    q = [out["q"][k][i] for k in range(len(ch.bodies))]
    ps, _ = ch.kinematics(th, np.zeros_like(th), q,
                          [np.zeros_like(x) for x in q])
    b = ch.bodies[-1]
    a = R.rotation_xyz(th[-1])
    return (ps[-1] + a
            @ np.array([b.length, b.phil @ q[-1][:b.n], b.phil @ q[-1][b.n:]]))


def test_fixed_hinge_between_beams_equals_continuous_beam():
    """AN: two 1 m beams welded by a fixed hinge (Sec. 8.10.2) = one 2 m beam.

    Clamped base, polynomial basis (u(0) = u'(0) = 0, free curvature at the
    weld), 8 modes per beam and plane vs 16 for the single beam.
    Frequencies: both equal the linear TMM clamped-free 2 m beam to 1e-6
    relative. Transient (gravity, DT-TMM Newmark dT 1e-3, 0.1 s, EI = 2e5):
    tip positions agree to 1e-6 m (the second-order kinematics of the two
    floating frames differ by about (deflection)^2 / l = 1e-8 m).
    """
    ei = 2.22
    w_lin = L.Chain([LB.EulerBernoulliBeam(2.0, ei, 1.0)], [0, 1],
                    [2, 3]).natural_frequencies(30.0)[:3]
    one = BL.SpatialFlexibleChain([_poly_beam(2.0, ei, 16)], hinges=["fixed"],
                                  gravity=(0.0, 0.0, 0.0))
    two = BL.SpatialFlexibleChain([_poly_beam(1.0, ei, 8),
                                   _poly_beam(1.0, ei, 8)],
                                  hinges=["fixed", "fixed"],
                                  gravity=(0.0, 0.0, 0.0))
    for ch in (one, two):
        f = _chain_freqs(ch)[:6].reshape(3, 2)
        np.testing.assert_allclose(f, np.column_stack([w_lin, w_lin]),
                                   rtol=1e-6)
    ei = 2e5
    one = BL.SpatialFlexibleChain([_poly_beam(2.0, ei, 16)], hinges=["fixed"],
                                  gravity=(0.0, -G_BOOK, 0.0))
    two = BL.SpatialFlexibleChain([_poly_beam(1.0, ei, 8),
                                   _poly_beam(1.0, ei, 8)],
                                  hinges=["fixed", "fixed"],
                                  gravity=(0.0, -G_BOOK, 0.0))
    o1 = one.simulate_dt(np.zeros((1, 3)), np.zeros((1, 3)), 0.1, 1e-3)
    o2 = two.simulate_dt(np.zeros((2, 3)), np.zeros((2, 3)), 0.1, 1e-3)
    diff = max(np.linalg.norm(_tip(one, o1, i) - _tip(two, o2, i)) for i
               in range(0, 101, 5))
    assert diff < 1e-6


def test_stiff_spring_hinge_tends_to_fixed_hinge():
    """AN: spring hinge (Sec. 8.12) with stiffness k -> infinity tends to the
    fixed hinge.

    First frequency of two welded polynomial beams; the difference falls as
    1/k: from k = 1e3 to 1e4 N m/rad by 10 +- 30 %.
    """
    ei = 2.22

    def chain(h):
        return BL.SpatialFlexibleChain([_poly_beam(1.0, ei, 6),
                                        _poly_beam(1.0, ei, 6)],
                                       hinges=["fixed", h],
                                       gravity=(0.0, 0.0, 0.0))

    f_fix = _chain_freqs(chain("fixed"))[0]
    diffs = [abs(_chain_freqs(chain(("spring", k, 0.0)))[0] - f_fix) for k
             in (1e3, 1e4)]
    assert 7.0 < diffs[0] / diffs[1] < 13.0


def _spring_chain(k1, k2, c1=0.0):
    beam = BL.SpatialBeam(1.0, 1.0, 5.0, n_modes=3, j_axial=1e-3)
    return BL.SpatialFlexibleChain([_cube(), beam, _cube()],
                                   hinges=["ball", ("spring", k1, c1),
                                           ("spring", k2, 0.0)],
                                   gravity=(0.0, -G_BOOK, 0.0))


def test_zero_stiffness_spring_hinge_equals_ball_hinge():
    """AN: k = c = 0 gives the ball hinge exactly (accelerations and RK4, tol
    1e-10)."""
    th = np.array([[0.1, -0.2, -1.0], [0.05, 0.1, -0.5], [0.2, 0.0, 0.3]])
    td = np.array([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.2], [0.3, -0.2, 0.1]])
    beam = BL.SpatialBeam(1.0, 1.0, 5.0, n_modes=3, j_axial=1e-3)
    ball = BL.SpatialFlexibleChain([_cube(), beam, _cube()],
                                   gravity=(0.0, -G_BOOK, 0.0))
    spr = _spring_chain(0.0, 0.0)
    ra = ball.accelerations(th, td)
    rb = spr.accelerations(th, td)
    for x, y in zip(ra["theta"], rb["theta"]):
        np.testing.assert_allclose(x, y, atol=1e-10)
    oa = ball.simulate_rk4(th, td, 0.2, 1e-3)
    ob = spr.simulate_rk4(th, td, 0.2, 1e-3)
    np.testing.assert_allclose(oa["theta"], ob["theta"], atol=1e-10)
    np.testing.assert_allclose(oa["q"][1], ob["q"][1], atol=1e-10)


def test_spatial_spring_hinge_reduces_to_planar_spring_hinge():
    """AN: planar motion with spring hinges equals the planar chain (Sec. 8.6),
    tol 1e-10."""
    pb = BL.PlanarBeam(1.0, 1.0, 5.0, 3)
    sb = BL.SpatialBeam(1.0, 1.0, 5.0, n_modes=3, j_axial=1e-3)
    hp = ["smooth", ("spring", 30.0, 0.5), ("spring", 20.0, 0.0)]
    pc = BL.PlanarFlexibleChain([R.uniform_bar(1.0, 1.0), pb,
                                 R.uniform_bar(1.0, 1.0)],
                                hinges=hp, gravity=(0.0, -G_BOOK))
    sc = BL.SpatialFlexibleChain([_bar3(), sb, _bar3()],
                                 hinges=["ball", ("spring", 30.0, 0.5),
                                         ("spring", 20.0, 0.0)],
                                 gravity=(0.0, -G_BOOK, 0.0))
    t0 = np.zeros((3, 3))
    t0[:, 2] = [-1.0, -0.6, -0.2]
    w0 = np.zeros((3, 3))
    w0[:, 2] = [0.2, -0.4, 0.5]
    out_s = sc.simulate_rk4(t0, w0, 0.2, 1e-3)
    out_p = pc.simulate_rk4(t0[:, 2], w0[:, 2], 0.2, 1e-3)
    np.testing.assert_allclose(out_s["theta"][:, :, 2], out_p["theta"],
                               atol=1e-10)
    np.testing.assert_allclose(out_s["q"][1][:, :3], out_p["q"][1], atol=1e-10)
    np.testing.assert_allclose(out_s["q"][1][:, 3:], 0.0, atol=1e-10)


def test_spring_hinges_energy_and_dttmm_order():
    """IV + CV: spatial chain with two rotary spring hinges, out-of-plane
    motion.

    RK4 energy (spring potential 1/2 k |Delta|^2 included) drift over 0.5 s
    with order above 4 - 0.2; DT-TMM Newmark angles and modal coordinates
    over 0.25 s, dT 1e-3 to 2.5e-4, order 2 +- 0.2.
    """
    ch = _spring_chain(30.0, 20.0)
    th0 = np.zeros((3, 3))
    th0[2] = [0.0, 0.2, 0.1]
    thd0 = np.zeros((3, 3))
    thd0[2] = [0.3, 0.5, 0.0]
    e0 = ch.energy(th0, thd0)
    drifts = []
    for dt in (2e-3, 1e-3, 5e-4):
        out = ch.simulate_rk4(th0, thd0, 0.5, dt)
        q = [None, out["q"][1][-1], None]
        qd = [None, out["qd"][1][-1], None]
        drifts.append(abs(ch.energy(out["theta"][-1], out["theta_d"][-1], q,
                                    qd) - e0))
    assert np.all(_orders(drifts) > 3.8), drifts
    ref = ch.simulate_rk4(th0, thd0, 0.25, 1.25e-4)
    errs = []
    for dt in (1e-3, 5e-4, 2.5e-4):
        out = ch.simulate_dt(th0, thd0, 0.25, dt)
        k = int(round(dt / 1.25e-4))
        errs.append(max(np.max(np.abs(out["theta"] - ref["theta"][::k])),
                        np.max(np.abs(out["q"][1] - ref["q"][1][::k]))))
    assert np.all(np.abs(_orders(errs) - 2.0) < 0.2), errs

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
