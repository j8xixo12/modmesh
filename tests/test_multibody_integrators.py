# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the time integrators (Haug 7.3-7.4, Rui TMM 7.3).

Rules under test:

* Every integrator converges with at least its theoretical order: rk4, abm4,
  partition, hybrid are 4th order; newmark and houbolt are 2nd order.
* partition, hybrid, newmark, and houbolt keep the position constraints at
  solver tolerance; rk4 and abm4 keep them at truncation-error level.
* On a stiff spring (omega h > 2.8), explicit rk4 diverges, the trapezoidal
  Newmark rule keeps the amplitude, and Houbolt damps it.
* All integrators agree on a tumbling 3D double pendulum.
* On a gyroscopic case with fast transients (revolute-spherical body), the
  adaptive rk45 conserves energy to 1e-6 where fixed-step rk4 loses 6e-3,
  hht stays stable with exact constraints, and classic newmark reports a
  Newton failure instead of returning a wrong state.
"""

import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

Z = [0.0, 0.0, 0.0]
AZ = [0.0, 0.0, 1.0]
G = 9.81
ALL = ["rk4", "rk45", "abm4", "partition", "hybrid", "newmark", "hht",
       "houbolt"]
ORDER = {"rk4": 4, "abm4": 4, "partition": 4, "hybrid": 4, "newmark": 2,
         "hht": 2, "houbolt": 2}
FIXED_STEP = [n for n in ALL if n in ORDER]


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def pendulum(mbs, dt, integrator):
    """Planar pendulum on a revolute joint, released at 60 degrees.

    :return: system
    :rtype: solvcon.multibody.DynamicsSystem
    """
    a = math.radians(60.0)
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    q = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
    b = mbs.create_mobilized_body_quat(1, [0.5 * math.cos(a - math.pi / 2),
                                           0.5 * math.sin(a - math.pi / 2), 0],
                                       Z, Z, q, Z, Z, 1.0, [0.01, 0.01, 0.02],
                                       [0, -G, 0], Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(1, Z, [0, 0.5, 0], Z, Z, AZ, AZ, g, b))
    s.set_integrator(integrator)
    s.init()
    return s


def angle_at(mbs, dt, integrator, t_end=1.0):
    s = pendulum(mbs, dt, integrator)
    for _ in range(int(round(t_end / dt))):
        s.solve()
    qq = np.asarray(s.get_body_quaternions())[1]
    return 2.0 * math.atan2(qq[3], qq[0])


@pytest.fixture(scope="module")
def reference():
    return angle_at(multibody, 2.5e-4, "rk4")


@pytest.mark.parametrize("name", FIXED_STEP)
def test_convergence_order(mbs, reference, name):
    h = 0.02 if ORDER[name] == 4 else 0.01
    e1 = abs(angle_at(mbs, h, name) - reference)
    e2 = abs(angle_at(mbs, h / 2, name) - reference)
    order = math.log2(e1 / e2)
    # At least the theoretical order; on this case abm4 and partition show
    # pre-asymptotic orders near 4.7 and 5.6 with errors below 1e-6.
    assert order > ORDER[name] - 0.3, (name, e1, e2, order)


def double_pendulum_3d(mbs, dt, integrator):
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    b1 = mbs.create_mobilized_body_quat(1, [0.4, -0.3, 0.2], Z, Z,
                                        [1, 0, 0, 0], [0.5, 1.0, -0.7], Z, 1.5,
                                        [0.02, 0.05, 0.04], [0, -G * 1.5, 0],
                                        Z)
    b2 = mbs.create_mobilized_body_quat(2, [0.9, -0.6, 0.5], Z, Z,
                                        [1, 0, 0, 0], [-0.3, 0.4, 1.2], Z, 0.8,
                                        [0.03, 0.01, 0.02], [0, -G * 0.8, 0],
                                        Z)
    for b in (g, b1, b2):
        s.Add(b)
    s.Add(mbs.create_joint(0, Z, [-0.4, 0.3, -0.2], Z, Z, AZ, AZ, g, b1))
    s.Add(mbs.create_joint(0, [0.25, -0.15, 0.15], [-0.25, 0.15, -0.15], Z, Z,
                           AZ, AZ, b1, b2))
    s.set_integrator(integrator)
    s.init()
    return s


@pytest.mark.parametrize("name", ALL)
def test_constraint_drift(mbs, name):
    s = double_pendulum_3d(mbs, 2e-3, name)
    worst = 0.0
    for _ in range(1000):
        s.solve()
        c = result_arrays(s.evaluate_constraints(s.get_sim_time()))["C"]
        worst = max(worst, np.max(np.abs(c)))
    limit = 1e-5 if name in ("rk4", "rk45", "abm4") else 1e-10
    assert worst < limit, (name, worst)


@pytest.mark.parametrize("name", ALL)
def test_integrators_agree_on_tumbling_double_pendulum(mbs, name):
    ref = double_pendulum_3d(mbs, 2.5e-4, "rk4")
    for _ in range(2000):
        ref.solve()
    s = double_pendulum_3d(mbs, 1e-3, name)
    for _ in range(500):
        s.solve()
    tol = 1e-3 if ORDER.get(name, 4) == 2 else 1e-7
    np.testing.assert_allclose(np.asarray(s.get_body_positions()),
                               np.asarray(ref.get_body_positions()), atol=tol)


def stiff_spring(mbs, integrator, dt=1e-3):
    # k = 1e7, m = 1: omega = 3162 rad/s, omega h = 3.16 > 2.78 (RK4 limit).
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [0, -0.501, 0], Z, Z, [1, 0, 0, 0],
                                       Z, Z, 1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, [0, 1, 0], [0, 1, 0], g, b))
    s.Add(mbs.create_spring_damper(g, b, Z, Z, 1e7, 0.0, 0.5))
    s.set_integrator(integrator)
    s.init()
    return s


def test_stiff_spring_rk4_diverges(mbs):
    s = stiff_spring(mbs, "rk4")
    with pytest.raises(RuntimeError):
        for _ in range(1000):
            s.solve()


def test_stiff_spring_newmark_keeps_amplitude(mbs):
    s = stiff_spring(mbs, "newmark")
    amp = 0.0
    for _ in range(1000):
        s.solve()
        amp = max(amp, abs(np.asarray(s.get_body_positions())[1][1] + 0.5))
    assert 0.9e-3 < amp < 1.001e-3


def test_stiff_spring_houbolt_damps_high_frequency(mbs):
    s = stiff_spring(mbs, "houbolt")
    for _ in range(1000):
        s.solve()
    assert abs(np.asarray(s.get_body_positions())[1][1] + 0.5) < 0.1e-3


def test_unknown_integrator_is_rejected(mbs):
    s = mbs.DynamicsSystem(1e-3)
    with pytest.raises(ValueError):
        s.set_integrator("euler")


def rev_sph_body(mbs, integrator, dt=1e-3):
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [0, 0, 0.8], Z, Z, [1, 0, 0, 0],
                                       [0.4, -0.7, 1.1], Z, 1.0,
                                       [0.02, 0.03, 0.04], [0, -G, 0], Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_revolute_spherical_joint(g, b, Z, [0.6, 0.0, -0.8], AZ))
    s.set_integrator(integrator)
    s.init()
    return s


def rot_body_to_global(q):
    e0, e1, e2, e3 = q
    e = np.array([e1, e2, e3])
    et = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return (2 * e0 * e0 - 1) * np.eye(3) + 2 * (np.outer(e, e) + e0 * et)


def body_energy(s, inertia=np.array([0.02, 0.03, 0.04])):
    r = np.array(s.get_body_positions())[1]
    v = np.array(s.get_body_velocities())[1]
    w = np.array(s.get_body_angular_velocities())[1]
    wb = rot_body_to_global(np.asarray(s.get_body_quaternions())[1]).T @ w
    return 0.5 * v @ v + 0.5 * wb @ (inertia * wb) + G * r[1]


def test_rk45_resolves_fast_gyroscopic_transient(mbs):
    s = rev_sph_body(mbs, "rk45")
    e0 = body_energy(s)
    drift = 0.0
    for _ in range(2000):
        s.solve()
        drift = max(drift, abs(body_energy(s) - e0))
    assert drift < 1e-6


def test_rk45_error_follows_tolerance(mbs, reference):
    errs = []
    for rtol in (1e-6, 1e-9):
        s = pendulum(mbs, 0.02, "rk45")
        s.set_tolerances(rtol, rtol * 1e-3)
        for _ in range(50):
            s.solve()
        qq = np.asarray(s.get_body_quaternions())[1]
        errs.append(abs(2.0 * math.atan2(qq[3], qq[0]) - reference))
    assert errs[1] < errs[0] / 50


def test_hht_is_stable_on_fast_gyroscopic_transient(mbs):
    s = rev_sph_body(mbs, "hht")
    for _ in range(2000):
        s.solve()
        c = result_arrays(s.evaluate_constraints(s.get_sim_time()))["C"]
        assert np.max(np.abs(c)) < 1e-12


def test_newmark_reports_failure_on_fast_gyroscopic_transient(mbs):
    s = rev_sph_body(mbs, "newmark")
    with pytest.raises(RuntimeError, match="Newton did not converge"):
        for _ in range(2000):
            s.solve()


def test_hht_alpha_range_is_checked(mbs):
    s = mbs.DynamicsSystem(1e-3)
    with pytest.raises(ValueError):
        s.set_hht_alpha(-0.5)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
