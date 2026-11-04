# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the axle, anti-roll bar, tire, ride modes, and road input
(roadmap P9.6, P9.7, P9.9, P9.10).

Rules under test:

* Roll: the generalized stiffness dQ/ds from driver efforts equals the
  virtual-work value 2 k MR^2 + k_bar (d theta_bar / ds)^2 at design.
* Quarter car: the static equilibrium keeps the suspension at design, and
  the tire deflection is W / k_t.
* Ride and wheel hop frequencies from modal analysis equal the exact linear
  quarter car built from K&C sensitivities (suspension.vehicle), and are
  within 0.5 % of the classic 2-DOF model.
* Road input: the steady-state sprung mass amplitude equals the 2-DOF
  frequency response.
"""

import copy
import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import axle, kc, vehicle
from solvcon.multibody.suspension.model import CornerModel


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


@pytest.fixture
def model():
    return CornerModel.load("race_front_corner")


@pytest.mark.parametrize("with_arb", [False, True])
def test_roll_stiffness_matches_virtual_work(mbs, model, with_arb):
    h = 1e-3
    r = axle.roll_sweep(mbs, model, [-h, 0.0, h], with_arb=with_arb)
    assert r["completed"]
    mr = kc.bump_sweep(mbs, model, [0.0])["motion_ratio"][0]
    expect = 2 * model.spring_k * mr ** 2
    if with_arb:
        n_l, n_r = r["axle"]["arb_nums"]
        ax = model.arb["axis"]
        dth = ((r["raw"]["domega_ds"][1, n_l] - r["raw"]["domega_ds"][1, n_r])
               @ ax)
        expect += model.arb["k"] * dth ** 2
        # the bar twists: drop link geometry is effective
        assert abs(dth) > 1.0
    # Central difference over +/- 1 mm of a smooth effort curve: rel. error
    # ~1e-5.
    assert r["dq_ds"][1] == pytest.approx(expect, rel=1e-4)


def test_arb_increases_roll_stiffness(mbs, model):
    soft = axle.roll_sweep(mbs, model, [-1e-3, 0.0, 1e-3], with_arb=False)
    stiff = axle.roll_sweep(mbs, model, [-1e-3, 0.0, 1e-3], with_arb=True)
    assert stiff["roll_stiffness"][1] > 1.2 * soft["roll_stiffness"][1]


def light_model(model):
    """Model with negligible rocker and rack mass (2-DOF reference holds)."""
    m = copy.deepcopy(model)
    m.rocker_mass, m.rocker_inertia = 1e-4, [1e-8, 1e-8, 1e-8]
    m.rack_mass = 1e-4
    return m


def test_quarter_car_static_equilibrium(mbs, model):
    m = light_model(model)
    ms = 200.0
    s, hd = vehicle.quarter_car(mbs, m, ms)
    res = s.find_equilibrium(tol=1e-8)
    assert res["converged"], res["message"]
    pos = np.asarray(s.get_body_positions())
    up = hd["corner"].nums["upright"]
    travel = pos[up][2] - pos[1][2] - m.hp["wheel_center"][2]
    assert abs(travel) < 1e-6
    # The tire free radius carries the weight at design: the wheel center
    # stays at the loaded radius and the chassis at its design height.
    assert pos[up][2] == pytest.approx(m.hp["wheel_center"][2], abs=1e-6)
    assert pos[1][2] == pytest.approx(0.0, abs=1e-6)
    w_total = (ms + m.upright_mass + m.rocker_mass + m.rack_mass) * 9.81
    state = mbs.get_tire_state(hd["tire"])
    assert state["normal_force"] == pytest.approx(w_total, rel=1e-6)
    assert state["deflection"] == pytest.approx(w_total / m.tire_k, rel=1e-6)


def test_ride_and_wheel_hop_match_two_dof_model(mbs, model):
    m = light_model(model)
    m.spring_c = 0.0
    m.tire_c = 0.0
    ms = 200.0
    s, hd = vehicle.quarter_car(mbs, m, ms)
    assert s.find_equilibrium(tol=1e-9)["converged"]
    modes = mbs.modal_analysis(s)
    f_sim = sorted(f for f in np.asarray(modes.frequencies) if f > 1e-3)
    kw = kc.bump_sweep(mbs, hd["model"], [-1e-4, 0.0, 1e-4])["wheel_rate"][1]
    w_tire = (ms + m.upright_mass + m.rocker_mass + m.rack_mass) * 9.81
    lin = vehicle.linear_quarter_car(mbs, m, ms, kw, m.tire_k,
                                     tire_load=w_tire)
    f_ref = vehicle.frequencies(lin["M"], lin["K"])
    assert len(f_sim) == 2, f_sim
    # Linearization at the same pose; wheel rate from a +/- 0.1 mm difference.
    np.testing.assert_allclose(f_sim, f_ref, rtol=1e-5)
    # The classic 2-DOF model is a close but not exact approximation here.
    mu = vehicle.effective_unsprung_mass(mbs, m)
    f_classic = vehicle.two_dof_frequencies(ms, mu, kw, m.tire_k)
    np.testing.assert_allclose(f_sim, f_classic, rtol=5e-3)


def test_road_input_steady_state_matches_frequency_response(mbs, model):
    m = light_model(model)
    ms = 200.0
    amp, freq = 1e-3, 2.0
    w = 2 * math.pi * freq
    road = mbs.DriveFunction.sine(amplitude=amp, omega=w)
    s, hd = vehicle.quarter_car(mbs, m, ms, road_func=road, dt=2e-3)
    s.find_equilibrium(tol=1e-9)
    s.set_integrator("rk45")
    z0 = np.asarray(s.get_body_positions())[1][2]
    zs = []
    n_steps = int(6.0 / 2e-3)
    for k in range(n_steps):
        s.solve()
        if k * 2e-3 > 5.0:
            zs.append(np.asarray(s.get_body_positions())[1][2] - z0)
    amp_sim = 0.5 * (max(zs) - min(zs))
    mr = hd["motion_ratio"]
    kw = kc.bump_sweep(mbs, hd["model"], [-1e-4, 0.0, 1e-4])["wheel_rate"][1]
    w_tire = (ms + m.upright_mass + m.rocker_mass + m.rack_mass) * 9.81
    lin = vehicle.linear_quarter_car(mbs, m, ms, kw, m.tire_k,
                                     tire_load=w_tire)
    a = lin["a"]
    cw = m.spring_c * mr ** 2
    cc = np.array([[m.tire_c, m.tire_c * a],
                   [m.tire_c * a, cw + m.tire_c * a * a]])
    # Road z_r enters through the tire: force k_t (z_r - z_c - a z) on the
    # patch.
    rhs = np.array([1.0, a]) * (m.tire_k + 1j * w * m.tire_c) * amp
    x = np.linalg.solve(-w * w * lin["M"] + 1j * w * cc + lin["K"], rhs)
    # 1 mm input: geometric nonlinearity and the damper rate MR^2 at design
    # (not at each pose) limit agreement to about 0.5 %.
    assert amp_sim == pytest.approx(abs(x[0]), rel=5e-3)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
