# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the point-contact tire (Haug Sec. 12.5) and the Python force element.

Rules under test:

* The tire carries m g with deflection m g / k and never pulls (no tension).
* At slip angle alpha the lateral force is -C_alpha alpha, capped at mu F_z.
* Load sensitivity: C_alpha ~ F_z^n and mu ~ F_z^(n-1) about F_z0.
* The brush (Fiala) curve has the slope C_alpha at zero slip and slides at
  tan(alpha) = 3 mu F_z / C_alpha (Pacejka 2012, Sec. 3.2.2).
* A longitudinal demand leaves the side force the rest of the friction
  circle; the relaxation length lags the slip angle over the rolled
  distance.
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


def slipping_wheel(mbs, alpha, c_alpha, mu, delta=0.01, k=2e5, speed=10.0,
                   commit_first=False, demand=None, **options):
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
    tire = mbs.create_tire(g, w, [0, 1, 0], R, k, c_alpha=c_alpha, mu=mu,
                           **options)
    s.Add(tire)
    if demand is not None:
        mbs.set_tire_longitudinal(tire, demand)
    if commit_first:
        # The lagged slip angle starts at zero (a slip step at t = 0).
        mbs.commit_tire(tire)
    s.init()
    s.solve()
    return tire, result_arrays(mbs.get_force_element_loads(tire))


FZ = 2e5 * 0.01


def test_load_sensitivity_scales_stiffness_and_friction(mbs):
    n, alpha, c_alpha = 0.8, math.radians(1.0), 5e4
    _, loads = slipping_wheel(mbs, alpha, c_alpha, mu=10.0, fz0=FZ / 2,
                              load_exponent=n)
    assert loads["force_j"][1] == pytest.approx(
        -c_alpha * 2.0 ** n * alpha, rel=1e-9)
    _, loads = slipping_wheel(mbs, math.radians(20.0), c_alpha, mu=1.2,
                              fz0=FZ / 2, load_exponent=n)
    assert loads["force_j"][1] == pytest.approx(
        -1.2 * 2.0 ** (n - 1.0) * FZ, rel=1e-9)


@pytest.mark.parametrize("deg", [0.2, 1.0, 2.0])
def test_brush_curve(mbs, deg):
    alpha, c_alpha, mu = math.radians(deg), 5e4, 1.0
    _, loads = slipping_wheel(mbs, alpha, c_alpha, mu, brush=True)
    s = c_alpha * math.tan(alpha)
    d = mu * FZ
    expect = -(s - s * s / (3 * d) + s ** 3 / (27 * d * d))
    assert loads["force_j"][1] == pytest.approx(expect, rel=1e-9)


def test_brush_curve_slides_past_the_sliding_angle(mbs):
    c_alpha, mu = 5e4, 1.0
    sliding = math.atan(3 * mu * FZ / c_alpha)
    _, loads = slipping_wheel(mbs, 1.01 * sliding, c_alpha, mu, brush=True)
    assert loads["force_j"][1] == pytest.approx(-mu * FZ, rel=1e-9)
    # The small-slip slope is C_alpha.
    _, loads = slipping_wheel(mbs, 1e-5, c_alpha, mu, brush=True)
    assert loads["force_j"][1] == pytest.approx(-c_alpha * math.tan(1e-5),
                                                rel=1e-4)


def test_longitudinal_demand_keeps_the_friction_circle(mbs):
    tire, loads = slipping_wheel(mbs, math.radians(15.0), 5e4, mu=1.0,
                                 demand=-0.6)
    # Braking: F_x = -0.6 F_z along the rolling direction (+x).
    assert loads["force_j"][0] == pytest.approx(-0.6 * FZ, rel=1e-9)
    assert loads["force_j"][1] == pytest.approx(-0.8 * FZ, rel=1e-9)
    state = mbs.get_tire_state(tire)
    assert state["longitudinal_force"] == pytest.approx(-0.6 * FZ, rel=1e-9)
    # A demand above the friction is capped at mu F_z.
    _, loads = slipping_wheel(mbs, math.radians(15.0), 5e4, mu=1.0,
                              demand=-1.5)
    assert loads["force_j"][0] == pytest.approx(-FZ, rel=1e-9)
    assert loads["force_j"][1] == pytest.approx(0.0, abs=1e-9)


def test_relaxation_lags_over_the_rolled_distance(mbs):
    alpha, c_alpha, sigma, speed = math.radians(1.0), 5e4, 0.3, 10.0
    tire, loads = slipping_wheel(mbs, alpha, c_alpha, mu=10.0, speed=speed,
                                 commit_first=True, relaxation_length=sigma)
    state = mbs.get_tire_state(tire)
    t = 1e-3
    x = speed * math.cos(alpha) * t
    lagged = alpha * (1.0 - math.exp(-x / sigma))
    assert state["slip_angle"] == pytest.approx(alpha, abs=1e-12)
    assert state["force_slip_angle"] == pytest.approx(lagged, rel=1e-6)
    assert loads["force_j"][1] == pytest.approx(-c_alpha * lagged, rel=1e-6)


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
