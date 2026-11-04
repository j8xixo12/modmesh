# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the point-contact tire (Haug Sec. 12.5) and the Python force element.

Rules under test:

* The tire carries m g with deflection m g / k and never pulls (no tension).
* At slip angle alpha the lateral force is -C_alpha alpha, capped at mu F_z.
* A Python force that implements a torsional spring reproduces the analytic
  oscillator, so the callback path gives the same physics as built-in elements.
"""

import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

Z = [0.0, 0.0, 0.0]
G = 9.81
R = 0.3


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def test_tire_static_deflection(mbs):
    m, k = 20.0, 2e5
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    w = mbs.create_mobilized_body_quat(1, [0, 0, R], Z, Z, [1, 0, 0, 0], Z, Z,
                                       m, [0.5, 0.8, 0.5], [0, 0, -m * G], Z)
    s.Add(g)
    s.Add(w)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, [0, 0, 1], [0, 0, 1], g, w))
    tire = mbs.create_tire(g, w, [0, 1, 0], R, k)
    s.Add(tire)
    s.init()
    assert s.find_equilibrium()["converged"]
    assert np.asarray(s.get_body_positions())[1][2] == pytest.approx(
        R - m * G / k, abs=1e-12)
    assert tire.get_vertical_force() == pytest.approx(m * G, rel=1e-9)


def test_tire_has_no_tension_and_lands(mbs):
    m, k = 20.0, 2e5
    s = mbs.DynamicsSystem(1e-4)
    g = mbs.create_ground(0)
    w = mbs.create_mobilized_body_quat(1, [0, 0, R + 0.02], Z, Z, [1, 0, 0, 0],
                                       Z, Z, m, [0.5, 0.8, 0.5],
                                       [0, 0, -m * G], Z)
    s.Add(g)
    s.Add(w)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, [0, 0, 1], [0, 0, 1], g, w))
    tire = mbs.create_tire(g, w, [0, 1, 0], R, k, c=200.0)
    s.Add(tire)
    s.set_integrator("rk45")
    s.init()
    fz = []
    for _ in range(3000):
        s.solve()
        fz.append(tire.get_vertical_force())
    assert min(fz) == 0.0  # airborne at the start
    assert max(fz) > m * G  # impact
    assert all(f >= 0.0 for f in fz)


def slipping_wheel(mbs, alpha, c_alpha, mu, delta=0.01, k=2e5, speed=10.0):
    # Heading +x (spin axis y). The wheel slides along d = (cos a, sin a, 0)
    # at constant speed: a prismatic joint and a translational driver.
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    w = mbs.create_mobilized_body_quat(1, [0, 0, R - delta], Z, Z,
                                       [1, 0, 0, 0], Z, Z, 20.0,
                                       [0.5, 0.8, 0.5], Z, Z)
    s.Add(g)
    s.Add(w)
    d = [math.cos(alpha), math.sin(alpha), 0.0]
    s.Add(mbs.create_joint(2, [0, 0, R - delta], Z, Z, Z, d, d, g, w))
    s.Add(mbs.create_translational_driver(
        body_i=g, body_j=w, point_i=[0, 0, R - delta], point_j=Z, axis_i=d,
        func=mbs.DriveFunction.polynomial([0.0, speed])))
    tire = mbs.create_tire(g, w, [0, 1, 0], R, k, c_alpha=c_alpha, mu=mu)
    s.Add(tire)
    s.init()
    s.solve()
    return tire, result_arrays(mbs.get_force_element_loads(tire))


def test_lateral_force_is_linear_in_slip(mbs):
    alpha, c_alpha = math.radians(2.0), 5e4
    tire, loads = slipping_wheel(mbs, alpha, c_alpha, mu=10.0)
    assert tire.get_slip_angle() == pytest.approx(alpha, abs=1e-12)
    assert loads["force_j"][1] == pytest.approx(-c_alpha * alpha, rel=1e-9)
    assert loads["force_j"][2] == pytest.approx(2e5 * 0.01, rel=1e-9)


def test_lateral_force_is_capped_by_friction(mbs):
    tire, loads = slipping_wheel(mbs, math.radians(10.0), 5e4, mu=1.2)
    assert loads["force_j"][1] == pytest.approx(-1.2 * 2e5 * 0.01, rel=1e-9)


def test_python_force_torsional_oscillator(mbs):
    k, j_zz, w0 = 4.0, 0.25, 1.0
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], [0, 0, w0], Z,
                                       1.0, [0.1, 0.1, j_zz], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, [0, 0, 1], [0, 0, 1], g, b))

    def torsion(t, si, sj):
        q = sj["p"]
        theta = 2.0 * math.atan2(q[3], q[0])
        return Z, Z, Z, [0.0, 0.0, -k * theta]

    s.Add(mbs.create_python_force(g, b, torsion))
    s.init()
    wn = math.sqrt(k / j_zz)
    for _ in range(1000):
        s.solve()
    q = np.asarray(s.get_body_quaternions())[1]
    assert 2 * math.atan2(q[3],
                          q[0]) == pytest.approx(w0 / wn * math.sin(wn * 1.0),
                                                 abs=1e-9)


def test_python_force_works_inside_modal_analysis(mbs):
    # modal_analysis releases the GIL; the callback must acquire it.
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z, 2.0,
                                       [0.1, 0.1, 0.1], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, [1, 0, 0], [1, 0, 0], g, b))
    s.Add(mbs.create_python_force(
        g, b, lambda t, si, sj: (Z, Z, [-800.0 * sj["r"][0], 0, 0], Z)))
    s.init()
    f = [x for x in np.asarray(mbs.modal_analysis(s).frequencies) if x > 1e-6]
    assert f[0] == pytest.approx(math.sqrt(800.0 / 2.0) / (2 * math.pi),
                                 rel=1e-6)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
