# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for P2: inertia tensor, RSDA, joint reactions, inverse dynamics, and
static equilibrium (Haug 11.2, 11.4.2, 11.5, 6.4-6.6, 7.5).

Rules under test:

* A body frame choice does not change the physics: a body with products of
  inertia moves exactly like the same body described in principal axes.
* A torsional RSDA oscillator has the natural frequency sqrt(k / J).
* Joint reactions satisfy Newton-Euler for every body:
  m a = F_applied + sum of reaction forces.
* The driver effort of a kinematically driven body equals the analytic
  actuator torque.
* Static equilibrium from Newton iteration equals the analytic result and
  the result of dynamic settling.
"""

import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import reaction_arrays

Z = [0.0, 0.0, 0.0]
AZ = [0.0, 0.0, 1.0]
G = 9.81


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def quat_mul(a, b):
    """Hamilton product a * b of quaternions [w, x, y, z].

    :return: product quaternion
    :rtype: numpy.ndarray
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot(q):
    """Haug A(p), body-to-global.

    :return: 3x3 matrix
    :rtype: numpy.ndarray
    """
    e0, e1, e2, e3 = q
    e = np.array([e1, e2, e3])
    et = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return (2 * e0 * e0 - 1) * np.eye(3) + 2 * (np.outer(e, e) + e0 * et)


# -------------------------------------------------------------- P2.1 inertia

def test_products_of_inertia_match_principal_description(mbs):
    principal = np.array([0.3, 0.5, 0.9])
    q_a = np.array([0.9, 0.1, -0.3, 0.3])
    q_a /= np.linalg.norm(q_a)
    q_r = np.array([0.8, 0.4, 0.2, -0.4])
    q_r /= np.linalg.norm(q_r)
    r_mat = rot(q_r)
    # Body frame B = frame A rotated by R: A_B = A_A R, J_B = R^T J_A R.
    j_b = r_mat.T @ np.diag(principal) @ r_mat
    q_b = quat_mul(q_a, q_r)
    omega = [1.0, 2.0, -1.5]
    sys_list = []
    for q, inertia in ((q_a, list(principal)),
                       (q_b, [j_b[0, 0], j_b[1, 1], j_b[2, 2],
                              j_b[0, 1], j_b[0, 2], j_b[1, 2]])):
        s = mbs.DynamicsSystem(1e-3)
        s.Add(mbs.create_ground(0))
        s.Add(mbs.create_mobilized_body_quat(1, Z, Z, Z, list(q), omega, Z,
                                             2.0, inertia, Z, Z))
        s.init()
        sys_list.append(s)
    for _ in range(1500):
        for s in sys_list:
            s.solve()
    w_a = np.array(sys_list[0].get_body_angular_velocities())[1]
    w_b = np.array(sys_list[1].get_body_angular_velocities())[1]
    np.testing.assert_allclose(w_b, w_a, atol=1e-10)
    a_a = rot(np.asarray(sys_list[0].get_body_quaternions())[1])
    a_b = rot(np.asarray(sys_list[1].get_body_quaternions())[1]) @ r_mat.T
    np.testing.assert_allclose(a_b, a_a, atol=1e-10)


def test_inertia_input_rejects_wrong_length(mbs):
    with pytest.raises(ValueError):
        mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z, 1.0,
                                       [0.1, 0.2], Z, Z)


# ----------------------------------------------------------------- P2.2 RSDA

def rotor_with_rsda(mbs, k, c, omega0, dt=1e-3, j_zz=0.25, torque=None):
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0],
                                       [0, 0, omega0], Z, 1.0,
                                       [0.1, 0.1, j_zz], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AZ, AZ, g, b))
    s.Add(mbs.create_rsda(g, b, AZ, AZ, k=k, c=c, torque=torque))
    return s


def z_angle(s):
    q = np.asarray(s.get_body_quaternions())[1]
    return 2.0 * math.atan2(q[3], q[0])


def test_rsda_torsional_oscillator_matches_analytic(mbs):
    k, j_zz, w0 = 4.0, 0.25, 1.0
    wn = math.sqrt(k / j_zz)
    s = rotor_with_rsda(mbs, k, 0.0, w0)
    s.init()
    for n in range(1, 2001):
        s.solve()
        if n % 100 == 0:
            t = n * 1e-3
            assert z_angle(s) == pytest.approx(w0 / wn * math.sin(wn * t),
                                               abs=1e-9)


def test_rsda_damping_matches_analytic_decay(mbs):
    k, c, j_zz, w0 = 4.0, 0.3, 0.25, 1.0
    wn = math.sqrt(k / j_zz)
    zeta = c / (2 * math.sqrt(k * j_zz))
    wd = wn * math.sqrt(1 - zeta ** 2)
    s = rotor_with_rsda(mbs, k, c, w0)
    s.init()
    for n in range(1, 2001):
        s.solve()
    t = 2.0
    expect = w0 / wd * math.exp(-zeta * wn * t) * math.sin(wd * t)
    assert z_angle(s) == pytest.approx(expect, abs=1e-9)


def test_rsda_actuator_torque_spins_free_rotor(mbs):
    s = rotor_with_rsda(mbs, 0.0, 0.0, 0.0,
                        torque=mbs.DriveFunction.polynomial([0.5]))
    s.init()
    for _ in range(1000):
        s.solve()
    # theta = 0.5 * (T / J) t^2 at t = 1 s: the torque acts on body j
    # in the positive (counterclockwise) sense.
    assert z_angle(s) == pytest.approx(0.5 * (0.5 / 0.25) * 1.0, abs=1e-9)


# ------------------------------------------------------------ P2.3 reactions

def test_hanging_pendulum_reaction_balances_gravity(mbs):
    m = 2.0
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [0, -0.5, 0], Z, Z, [1, 0, 0, 0],
                                       Z, Z, m, [0.05, 0.05, 0.05],
                                       [0, -G * m, 0], Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(1, Z, [0, 0.5, 0], Z, Z, AZ, AZ, g, b))
    s.init()
    s.solve()
    r = reaction_arrays(s.get_joint_reactions())[0]
    np.testing.assert_allclose(r["force_j"], [0, G * m, 0], atol=1e-8)
    np.testing.assert_allclose(r["torque_j"], Z, atol=1e-8)
    np.testing.assert_allclose(r["force_i"], [0, -G * m, 0], atol=1e-8)


def test_reactions_satisfy_newton_euler_for_tumbling_chain(mbs):
    # Two bodies, spherical joints, 3D motion: for each body
    # m a = F_applied + sum(F_reaction) and
    # dH/dt (about COM, global) = sum(T_reaction at point + r x F).
    m = [1.5, 0.8]
    inertia = [np.array([0.02, 0.05, 0.04]), np.array([0.03, 0.01, 0.02])]
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b1 = mbs.create_mobilized_body_quat(1, [0.4, -0.3, 0.2], Z, Z,
                                        [1, 0, 0, 0], [0.5, 1.0, -0.7], Z,
                                        m[0], list(inertia[0]),
                                        [0, -G * m[0], 0], Z)
    b2 = mbs.create_mobilized_body_quat(2, [0.9, -0.6, 0.5], Z, Z,
                                        [1, 0, 0, 0], [-0.3, 0.4, 1.2], Z,
                                        m[1], list(inertia[1]),
                                        [0, -G * m[1], 0], Z)
    for b in (g, b1, b2):
        s.Add(b)
    s.Add(mbs.create_joint(0, Z, [-0.4, 0.3, -0.2], Z, Z, AZ, AZ, g, b1))
    s.Add(mbs.create_joint(0, [0.25, -0.15, 0.15], [-0.25, 0.15, -0.15],
                           Z, Z, AZ, AZ, b1, b2))
    s.init()
    for _ in range(400):
        s.solve()
    reac = reaction_arrays(s.get_joint_reactions())
    acc = np.asarray(s.get_body_accelerations())
    pos = np.asarray(s.get_body_positions())
    for body, mass in ((1, m[0]), (2, m[1])):
        total = np.array([0.0, -G * mass, 0.0])
        for r in reac:
            if r["body_i"] == body:
                total += r["force_i"]
            if r["body_j"] == body:
                total += r["force_j"]
        np.testing.assert_allclose(mass * np.array(acc[body]), total,
                                   atol=1e-7)
    # Angular check for body 2 (one joint): torque about COM from the joint.
    w = np.array(s.get_body_angular_velocities())[2]
    alpha = np.array(s.get_body_angular_accelerations())[2]
    a_mat = rot(np.asarray(s.get_body_quaternions())[2])
    j_glob = a_mat @ np.diag(inertia[1]) @ a_mat.T
    h_dot = j_glob @ alpha + np.cross(w, j_glob @ w)
    r = reac[1]
    lever = np.array(r["point_j"]) - np.array(pos[2])
    torque = np.array(r["torque_j"]) + np.cross(lever, r["force_j"])
    np.testing.assert_allclose(h_dot, torque, atol=1e-7)


# --------------------------------------------------- P2.4 inverse dynamics

def test_driver_effort_equals_analytic_crank_torque(mbs):
    m, rr, jc = 1.2, 0.3, 0.02
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [rr, 0, 0], Z, Z, [1, 0, 0, 0], Z,
                                       Z, m, [jc, jc, jc], [0, -G * m, 0], Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(1, Z, [-rr, 0, 0], Z, Z, AZ, AZ, g, b))
    f = mbs.DriveFunction.polynomial([0.0, 2.0, 1.5])  # theta = 2 t + 1.5 t^2
    drv = mbs.create_rotational_driver(body_i=g, body_j=b, axis_i=AZ,
                                       axis_j=AZ, func=f)
    s.Add(drv)
    s.init()
    for n in range(1, 501):
        s.solve()
        th, _, thdd = f.eval(n * 1e-3)
        expect = (jc + m * rr * rr) * thdd + m * G * rr * math.cos(th)
        effort = reaction_arrays(s.get_joint_reactions())[1]["effort"]
        assert effort == pytest.approx(expect, abs=1e-8)


# ------------------------------------------------------ P2.5 equilibrium

def hanging_spring_mass(mbs, c=0.0, x0=-0.5):
    m, k, l0 = 2.0, 50.0, 0.5
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [0, x0, 0], Z, Z, [1, 0, 0, 0], Z,
                                       Z, m, [0.05, 0.05, 0.05],
                                       [0, -G * m, 0], Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, [0, 1, 0], [0, 1, 0], g, b))
    s.Add(mbs.create_spring_damper(g, b, Z, Z, k, c, l0))
    return s, -(l0 + m * G / k)


def test_equilibrium_matches_analytic_spring_extension(mbs):
    s, y_eq = hanging_spring_mass(mbs)
    s.init()
    res = s.find_equilibrium()
    assert res["converged"]
    assert np.asarray(s.get_body_positions())[1][1] == pytest.approx(y_eq,
                                                                     abs=1e-10)
    np.testing.assert_allclose(np.asarray(s.get_body_velocities())[1], Z,
                               atol=1e-14)


def test_equilibrium_matches_dynamic_settling(mbs):
    s, y_eq = hanging_spring_mass(mbs, c=8.0)
    s.init()
    for _ in range(20000):
        s.solve()
    y_settle = np.asarray(s.get_body_positions())[1][1]
    s2, _ = hanging_spring_mass(mbs, c=8.0)
    s2.init()
    s2.find_equilibrium()
    assert np.asarray(s2.get_body_positions())[1][1] == pytest.approx(y_settle,
                                                                      abs=1e-8)


def test_equilibrium_of_pendulum_under_offset_load(mbs):
    # A pendulum with a horizontal load F hangs at tan(phi) = F / (m g).
    m, length, f = 1.0, 0.5, 3.0
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [0, -length, 0], Z, Z, [1, 0, 0, 0],
                                       Z, Z, m, [0.01, 0.01, 0.01],
                                       [f, -G * m, 0], Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(1, Z, [0, length, 0], Z, Z, AZ, AZ, g, b))
    s.init()
    res = s.find_equilibrium()
    assert res["converged"]
    x, y, _ = np.asarray(s.get_body_positions())[1]
    assert math.atan2(x, -y) == pytest.approx(math.atan2(f, m * G), abs=1e-10)


def test_tsda_spring_mass_oscillates_at_sqrt_k_over_m(mbs):
    # Spring stretched by 0.1 from free length, no gravity: x(t) = 0.1 cos(wn
    # t).
    m, k, l0 = 2.0, 50.0, 0.5
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [0, -0.6, 0], Z, Z, [1, 0, 0, 0], Z,
                                       Z, m, [0.05, 0.05, 0.05], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, [0, 1, 0], [0, 1, 0], g, b))
    s.Add(mbs.create_spring_damper(g, b, Z, Z, k, 0.0, l0))
    s.init()
    wn = math.sqrt(k / m)
    for n in range(1, 2001):
        s.solve()
    y = np.asarray(s.get_body_positions())[1][1]
    assert y == pytest.approx(-0.5 - 0.1 * math.cos(wn * 2.0), abs=1e-9)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
