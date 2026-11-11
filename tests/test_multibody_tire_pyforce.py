# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the point-contact tire (Haug Sec. 12.5) and the Python force element.

Rules under test:

* The tire carries m g with deflection m g / k and never pulls (no tension).
* At slip angle alpha the lateral force is -C_alpha alpha, capped at mu F_z.
* The Magic Formula tire (Pacejka 2012, Sec. 4.3.2) gives D sin(C atan(B
  tan alpha)) at pure side slip: the slope B C D at zero slip, the peak D =
  mu F_z (F_z / F_z0)^(n - 1) at B tan(alpha) = tan(pi / (2 C)), and the
  sliding level D sin(C pi / 2).  The combined slip uses the theoretical
  slip: a locked wheel slides against its slip velocity.
* A longitudinal demand sets the slip ratio of the Magic Formula tire and
  locks the wheel past the peak; the linear tire keeps the rest of the
  friction circle.  The relaxation length lags the slip angle over the
  rolled distance.
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
                   commit_first=False, demand=None, spin=(0, 1, 0),
                   **options):
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
    tire = mbs.create_tire(g, w, list(spin), R, k, c_alpha=c_alpha, mu=mu,
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
#: A Magic Formula tire with the lateral peak at 7.52 deg and the sliding
#: level 0.86 of the peak (the AC data of the Golf R semislick).
SHAPE = 2.0 - 2.0 / math.pi * math.asin(0.86)
PEAK = math.radians(7.52)
B_Y = math.tan(0.5 * math.pi / SHAPE) / math.tan(PEAK)


def magic_formula(mbs, fz0=FZ, n=0.8, mu=1.2, shape=SHAPE):
    curve = dict(mu=mu, load_exponent=n, stiffness=B_Y, shape=shape)
    return mbs.MagicFormulaTire(fz0=fz0, longitudinal=dict(curve),
                                lateral=dict(curve))


def test_magic_formula_pure_side_slip(mbs):
    mf = magic_formula(mbs)
    d = 1.2 * FZ
    for deg in (0.5, 3.0, 7.52, 15.0, 60.0):
        alpha = math.radians(deg)
        expect = d * math.sin(SHAPE * math.atan(B_Y * math.tan(alpha)))
        assert mf.force(FZ, 0.0, alpha) == pytest.approx((0.0, -expect),
                                                         rel=1e-12)
    # The peak, the slope at zero slip, and the sliding level.
    assert -mf.force(FZ, 0.0, PEAK)[1] == pytest.approx(d, rel=1e-12)
    assert mf.cornering_stiffness(FZ) == pytest.approx(B_Y * SHAPE * d,
                                                       rel=1e-12)
    assert -mf.force(FZ, 0.0, math.radians(89.99))[1] == pytest.approx(
        0.86 * d, rel=1e-3)


def test_magic_formula_load_sensitivity(mbs):
    n = 0.8
    mf = magic_formula(mbs, fz0=FZ / 2, n=n)
    assert mf.friction(FZ) == pytest.approx((1.2 * 2.0 ** (n - 1.0),) * 2,
                                            rel=1e-12)
    ratio = mf.cornering_stiffness(FZ) / mf.cornering_stiffness(FZ / 2)
    assert ratio == pytest.approx(2.0 ** n, rel=1e-12)
    assert mf.force(0.0, -1.0, 0.1) == (0.0, 0.0)


def test_magic_formula_locked_wheel_slides_against_the_slip(mbs):
    mf = magic_formula(mbs)
    for alpha in (0.02, 0.1, 0.4):
        fx, fy = mf.force(FZ, -1.0, alpha)
        magnitude = math.hypot(fx, fy)
        assert magnitude == pytest.approx(1.2 * FZ * 0.86, rel=1e-9)
        assert -fy / magnitude == pytest.approx(math.sin(alpha), rel=1e-9)


def test_magic_formula_demand(mbs):
    mf = magic_formula(mbs)
    for demand in (-2000.0, -100.0, 100.0, 2000.0):
        fx, fy, kappa = mf.force_for_demand(FZ, demand, 0.03)
        assert fx == pytest.approx(demand, abs=1e-6)
        assert mf.force(FZ, kappa, 0.03) == pytest.approx((fx, fy))
    # Past the peak the wheel locks or spins.
    assert mf.force_for_demand(FZ, -1e5, 0.03)[2] == -1.0
    assert mf.force_for_demand(FZ, 1e5, 0.03)[2] == 1.0


@pytest.mark.parametrize("bad", [dict(shape=2.0), dict(curvature=1.5),
                                 dict(mu=0.0), dict(stiffness=-1.0)])
def test_magic_formula_rejects_bad_curves(mbs, bad):
    curve = dict(mu=1.0, load_exponent=1.0, stiffness=10.0, shape=1.3)
    curve.update(bad)
    with pytest.raises(ValueError):
        mbs.MagicFormulaTire(fz0=FZ, longitudinal=curve, lateral=curve)
    with pytest.raises(KeyError):
        mbs.MagicFormulaTire(fz0=FZ, longitudinal={"mux": 1.0},
                             lateral={})


def test_tire_force_uses_the_magic_formula(mbs):
    mf = magic_formula(mbs)
    alpha = math.radians(4.0)
    tire, loads = slipping_wheel(mbs, alpha, 0.0, 1.0, magic_formula=mf)
    state = mbs.get_tire_state(tire)
    assert state["normal_force"] == pytest.approx(FZ, rel=1e-9)
    assert loads["force_j"][1] == pytest.approx(
        mf.force(FZ, 0.0, alpha)[1], rel=1e-9)


def test_longitudinal_demand_of_the_magic_formula_tire(mbs):
    mf = magic_formula(mbs)
    alpha = math.radians(3.0)
    _, loads = slipping_wheel(mbs, alpha, 0.0, 1.0, magic_formula=mf,
                              demand=-0.6)
    fx, fy, _ = mf.force_for_demand(FZ, -0.6 * FZ, alpha)
    assert loads["force_j"][0] == pytest.approx(-0.6 * FZ, rel=1e-9)
    assert loads["force_j"][1] == pytest.approx(fy, rel=1e-9)
    # A demand past the peak locks the wheel.
    _, loads = slipping_wheel(mbs, alpha, 0.0, 1.0, magic_formula=mf,
                              demand=-1.5)
    locked = mf.force(FZ, -1.0, alpha)
    assert loads["force_j"][0] == pytest.approx(locked[0], rel=1e-9)
    assert loads["force_j"][1] == pytest.approx(locked[1], rel=1e-9)


def test_longitudinal_demand_keeps_the_friction_circle(mbs):
    tire, loads = slipping_wheel(mbs, math.radians(15.0), 5e4, mu=1.0,
                                 demand=-0.6)
    # Braking: F_x = -0.6 F_z along the rolling direction (+x).
    assert loads["force_j"][0] == pytest.approx(-0.6 * FZ, rel=1e-9)
    assert loads["force_j"][1] == pytest.approx(-0.8 * FZ, rel=1e-9)
    state = mbs.get_tire_state(tire)
    assert state["longitudinal_force"] == pytest.approx(-0.6 * FZ, rel=1e-9)


def test_longitudinal_demand_is_along_the_travel_on_a_mirrored_wheel(mbs):
    # The spin axis of a right wheel points the other way: s x n is then
    # backward.  A brake demand still opposes the travel (+x), and the
    # side force is the same.
    alpha = math.radians(15.0)
    _, left = slipping_wheel(mbs, alpha, 5e4, mu=1.0, demand=-0.6)
    tire, right = slipping_wheel(mbs, alpha, 5e4, mu=1.0, demand=-0.6,
                                 spin=(0, -1, 0))
    assert right["force_j"][0] == pytest.approx(-0.6 * FZ, rel=1e-9)
    np.testing.assert_allclose(right["force_j"], left["force_j"],
                               rtol=1e-9, atol=1e-9)
    _, drive = slipping_wheel(mbs, 0.0, 5e4, mu=1.0, demand=0.3,
                              spin=(0, -1, 0))
    assert drive["force_j"][0] == pytest.approx(0.3 * FZ, rel=1e-9)


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
