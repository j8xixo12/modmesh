# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the 6x6 bushing and the massless beam element (P7.1, P7.4).

Rules under test:

* A bushing between ground and a body gives the analytic static deflection
  m g / k and the natural frequencies sqrt(k / m) and sqrt(k_r / J).
* Bushing stiffness acts in the bushing frame, not in the body frame.
* An undamped bushing with isotropic stiffness conserves energy in 3D.
* A cantilever beam element gives the analytic tip deflection and slope
  F L^3 / (3 E I) and F L^2 / (2 E I).
"""

import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.elements import beam_stiffness, bushing_matrix

Z = [0.0, 0.0, 0.0]
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


def mounted(mbs, k_t, k_r, c_t=(0, 0, 0), c_r=(0, 0, 0), gravity=Z, x0=Z,
            omega=Z, inertia=(0.02, 0.03, 0.04), frame=(1, 0, 0, 0), m=2.0):
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], list(omega),
                                       Z, m, list(inertia), list(gravity), Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_bushing(g, b, Z, Z, bushing_matrix(k_t, k_r),
                             bushing_matrix(c_t, c_r), frame_quat=list(frame)))
    s.set_integrator("rk45")
    s.init()
    if any(x0):
        q = np.asarray(s.get_body_quaternions())[1]
        s.set_body_state(1, list(x0), Z, q, [0, 0, 0, 0])
    return s


def test_static_deflection(mbs):
    s = mounted(mbs, (1e4, 2e4, 3e4), (50, 50, 50), gravity=[0, -2.0 * G, 0])
    assert s.find_equilibrium()["converged"]
    assert np.asarray(s.get_body_positions())[1][1] == pytest.approx(
        -2.0 * G / 2e4, abs=1e-12)


def test_translational_frequency(mbs):
    s = mounted(mbs, (1e4, 2e4, 3e4), (50, 50, 50), x0=[0.01, 0, 0])
    wn = math.sqrt(1e4 / 2.0)
    for _ in range(500):
        s.solve()
    assert np.asarray(s.get_body_positions())[1][0] == pytest.approx(
        0.01 * math.cos(wn * 0.5), abs=1e-9)


def test_torsional_frequency(mbs):
    k_rz, j_zz = 8.0, 0.04
    s = mounted(mbs, (1e4, 1e4, 1e4), (5, 5, k_rz), omega=[0, 0, 0.5])
    wn = math.sqrt(k_rz / j_zz)
    for _ in range(500):
        s.solve()
    q = np.asarray(s.get_body_quaternions())[1]
    assert 2 * math.atan2(q[3],
                          q[0]) == pytest.approx(0.5 / wn * math.sin(wn * 0.5),
                                                 abs=1e-8)


def test_stiffness_acts_in_bushing_frame(mbs):
    # Frame rotated 90 degrees about z: bushing x (soft) is global y.
    c45 = math.cos(math.pi / 4)
    s = mounted(mbs, (1e3, 1e5, 1e5), (50, 50, 50), gravity=[0, -2.0 * G, 0],
                frame=(c45, 0, 0, c45))
    s.find_equilibrium()
    assert np.asarray(s.get_body_positions())[1][1] == pytest.approx(
        -2.0 * G / 1e3, abs=1e-12)


def test_damped_translation_matches_analytic(mbs):
    m, k, c = 2.0, 1e4, 20.0
    s = mounted(mbs, (k, 1e5, 1e5), (50, 50, 50), c_t=(c, 0, 0),
                x0=[0.01, 0, 0])
    wn = math.sqrt(k / m)
    zeta = c / (2 * math.sqrt(k * m))
    wd = wn * math.sqrt(1 - zeta ** 2)
    for _ in range(300):
        s.solve()
    t = 0.3
    x = 0.01 * math.exp(-zeta * wn * t) * (math.cos(wd * t)
                                           + zeta * wn / wd * math.sin(wd * t))
    assert np.asarray(s.get_body_positions())[1][0] == pytest.approx(x,
                                                                     abs=1e-9)


def rot(q):
    e0, e1, e2, e3 = q
    e = np.array([e1, e2, e3])
    et = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return (2 * e0 * e0 - 1) * np.eye(3) + 2 * (np.outer(e, e) + e0 * et)


def test_isotropic_bushing_conserves_energy_in_3d(mbs):
    k_t, k_r, m = 4e3, 30.0, 2.0
    inertia = np.array([0.02, 0.03, 0.04])
    s = mounted(mbs, (k_t,) * 3, (k_r,) * 3, x0=[0.005, -0.003, 0.004],
                omega=[1.5, -2.0, 2.5])

    def energy():
        r = np.array(s.get_body_positions())[1]
        v = np.array(s.get_body_velocities())[1]
        q = np.asarray(s.get_body_quaternions())[1]
        w = rot(q).T @ np.array(s.get_body_angular_velocities())[1]
        # rotation vector of A for the potential 0.5 k_r |theta|^2
        a = rot(q)
        ang = math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(a) - 1))))
        vec = np.array([a[2, 1] - a[1, 2], a[0, 2] - a[2, 0],
                        a[1, 0] - a[0, 1]])
        th = vec * (ang / (2 * math.sin(ang)) if ang > 1e-8 else 0.5)
        return (0.5 * m * v @ v + 0.5 * w @ (inertia * w) + 0.5 * k_t * r @ r
                + 0.5 * k_r * th @ th)

    e0 = energy()
    for _ in range(1000):
        s.solve()
    assert abs(energy() - e0) < 1e-7 * e0


def test_cantilever_beam_tip_deflection(mbs):
    e_mod, length, d = 2.1e11, 0.5, 0.02
    i_a = math.pi * d ** 4 / 64
    k = beam_stiffness(e_mod, 8.1e10, math.pi * d ** 2 / 4, i_a, i_a, 2 * i_a,
                       length)
    f = 200.0
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    tip = mbs.create_mobilized_body_quat(1, [length, 0, 0], Z, Z, [1, 0, 0, 0],
                                         Z, Z, 1.0, [1e-3, 1e-3, 1e-3],
                                         [0, -f, 0], Z)
    s.Add(g)
    s.Add(tip)
    s.Add(mbs.create_bushing(g, tip, Z, Z, k, bushing_matrix(Z, Z)))
    s.init()
    assert s.find_equilibrium(tol=1e-8)["converged"]
    v = np.asarray(s.get_body_positions())[1][1]
    q = np.asarray(s.get_body_quaternions())[1]
    slope = 2 * math.atan2(q[3], q[0])
    # Small-deflection theory; the solver uses the exact rotation vector, so
    # the comparison holds to the geometric nonlinearity (v / L)^2 ~ 1e-4.
    assert v == pytest.approx(-f * length ** 3 / (3 * e_mod * i_a), rel=2e-3)
    assert slope == pytest.approx(-f * length ** 2 / (2 * e_mod * i_a),
                                  rel=2e-3)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
