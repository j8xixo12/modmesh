# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The damper force-velocity curve (``create_curve_damper``) and the
``damper`` block of a corner model.

* The engine curve equals the table, extends it with the end slopes, and
  rejects a table that is not passive.
* A symmetric linear curve is the linear spring-damper: the quarter car
  histories agree to round-off.
* The energy a damper dissipates over a sine stroke equals the integral of
  its curve, pi omega A^2 (c_bump + c_rebound) / 2 for a two-slope damper
  below its knee.
* Asymmetric damping shifts the mean position: the time average of the
  equation of motion of a stationary oscillator gives k <x> = -<F_d>, so a
  damper stiffer in rebound than in bump (the usual 2 to 3 ratio, Balike,
  Rakheja & Stiharu, IJVD 54(3), 2010, after Ribeiro & Cunha, arXiv
  2605.05235) pulls the suspension down.
"""

import copy
import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import model as model_mod
from solvcon.multibody.suspension import road
from solvcon.multibody.suspension import vehicle
from solvcon.multibody.suspension.model import CornerModel
from solvcon.multibody.tools.arrays import reaction_arrays

Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]

pytestmark = pytest.mark.skipif(
    not getattr(multibody, "FEATURE_CURVE_DAMPER", False),
    reason="the engine has no curve damper")


def _slider(mbs, mass=1.0, z0=-0.5):
    """A body on a vertical slider below the ground origin."""
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [0, 0, z0], Z, Z, [1, 0, 0, 0], Z,
                                       Z, mass, [0.01, 0.01, 0.01], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, EZ, EZ, g, b))
    return s, g, b


def test_curve_follows_the_table_and_its_end_slopes():
    s, g, b = _slider(multibody)
    d = multibody.create_curve_damper(g, b, Z, Z, 0.0, 0.5, [-1.0, 0.0, 1.0],
                                      [-1000.0, 0.0, 3000.0])
    expect = {-2.0: -2000.0, -0.5: -500.0, 0.0: 0.0, 0.5: 1500.0,
              2.0: 6000.0}
    for v, f in expect.items():
        assert multibody.curve_damper_force(d, v) == pytest.approx(f)
    curve = ([-1.0, 0.0, 1.0], [-1000.0, 0.0, 3000.0])
    vs = np.linspace(-3.0, 3.0, 61)
    np.testing.assert_allclose(
        [multibody.curve_damper_force(d, v) for v in vs],
        model_mod.damper_force(curve, vs), rtol=1e-14, atol=1e-9)


@pytest.mark.parametrize("velocities, forces", [
    ([0.0], [0.0]),
    ([0.0, 0.0], [0.0, 1.0]),
    ([0.0, 1.0], [0.0, -5.0]),
    ([-1.0, 1.0], [10.0, 10.0]),
])
def test_curve_rejects_bad_tables(velocities, forces):
    s, g, b = _slider(multibody)
    with pytest.raises(ValueError):
        multibody.create_curve_damper(g, b, Z, Z, 0.0, 0.5, velocities,
                                      forces)


def test_model_damper_blocks():
    digressive = model_mod.damper_table(
        {"kind": "asymmetric", "bump": [1.0, 0.4], "rebound": [3.0, 1.0],
         "knee": 100.0})
    # N.s/mm and mm/s: 1 N.s/mm at 0.1 m/s is 100 N.
    assert model_mod.damper_force(digressive, -0.1) == pytest.approx(-100.0)
    assert model_mod.damper_force(digressive, 0.1) == pytest.approx(300.0)
    assert model_mod.damper_force(digressive, -0.3) == pytest.approx(
        -100.0 - 400.0 * 0.2)
    assert model_mod.damper_force(digressive, 0.3) == pytest.approx(
        300.0 + 1000.0 * 0.2)
    table = model_mod.damper_table(
        {"kind": "table", "velocity": [-500, 0, 500],
         "force": [-800, 0, 2000]})
    assert model_mod.curve_slope(table, 0.1) == pytest.approx(4000.0)
    for bad in ({"kind": "asymmetric", "bump": -1.0, "rebound": 1.0},
                {"kind": "table", "velocity": [0, 1], "force": [0, -1]},
                {"kind": "spline", "velocity": [0, 1], "force": [0, 1]}):
        with pytest.raises(ValueError):
            model_mod.damper_table(bad)


def _with_damper(stem, block):
    raw = copy.deepcopy(CornerModel.load(stem).raw)
    raw["spring"]["damper"] = block
    return CornerModel(raw)


def _rig_run(model, steps=300):
    rig = road.rig_input(0.01, 6.0, 0.5)
    s, _ = vehicle.quarter_car(multibody, model, 200.0, road=rig, dt=1e-3)
    s.set_integrator("rk45")
    s.set_tolerances(1e-9, 1e-12)
    out = []
    for _ in range(steps):
        s.solve()
        out.append(np.asarray(s.get_body_positions()).copy())
    return np.array(out)


def test_symmetric_linear_curve_is_the_linear_damper():
    base = CornerModel.load("race_front_corner")
    c = base.spring_c * 1e-3  # N.s/mm
    curve = _with_damper("race_front_corner",
                         {"kind": "asymmetric", "bump": c, "rebound": c})
    assert curve.spring_c == pytest.approx(base.spring_c)
    np.testing.assert_allclose(_rig_run(curve), _rig_run(base), atol=1e-9)


@pytest.mark.parametrize("spec, amplitude", [
    ({"kind": "asymmetric", "bump": 1.0, "rebound": 3.0, "knee": 1000.0},
     0.02),
    ({"kind": "asymmetric", "bump": [1.5, 0.5], "rebound": [4.0, 1.2],
      "knee": 80.0}, 0.03),
])
def test_dissipated_energy_per_cycle(spec, amplitude):
    mbs = multibody
    curve = model_mod.damper_table(spec)
    omega = 2.0 * math.pi * 4.0
    s, g, b = _slider(mbs, mass=1e-3)
    s.Add(mbs.create_curve_damper(g, b, Z, Z, 0.0, 0.5, curve[0],
                                  curve[1]))
    # z(t) = -0.5 + A sin(omega t): the damper stroke starts at speed
    # A omega in rebound (the damper is above the body).
    f = mbs.DriveFunction.sine(amplitude, omega, 0.0, -0.5)
    s.Add(mbs.create_absolute_driver(body=b, ground=g, point=Z,
                                     direction=EZ, func=f))
    s.init()
    period = 2.0 * math.pi / omega
    steps = int(round(period / 1e-3))
    # The work over the second and third periods (whole periods, both
    # ends sampled), per period.
    work = 0.0
    prev = None
    for step in range(1, 3 * steps + 1):
        s.solve()
        if step < steps:
            continue
        effort = reaction_arrays(s.get_joint_reactions())[1]["effort"]
        zdot = np.asarray(s.get_body_velocities())[1][2]
        power = effort * zdot
        if prev is not None:
            work += 0.5 * (power + prev) * 1e-3
        prev = power
    work *= 0.5
    # The driver puts in what the damper dissipates (the 1 g body adds a
    # negligible inertia force); the reference is the curve integral.
    t = np.linspace(0.0, period, 20001)
    v = amplitude * omega * np.cos(omega * t)
    expect = np.trapezoid(model_mod.damper_force(curve, -v) * -v, t)
    assert abs(work) == pytest.approx(expect, rel=2e-3)
    if spec["knee"] > amplitude * omega * 1e3:
        # Below the knee: the closed form of a two-slope damper.
        c_b, c_r = spec["bump"] * 1e3, spec["rebound"] * 1e3
        closed = math.pi * omega * amplitude ** 2 * 0.5 * (c_b + c_r)
        assert expect == pytest.approx(closed, rel=1e-6)


def test_rebound_biased_damping_pulls_the_mass_down():
    # A mass on a spring and an asymmetric damper to a shaking base: the
    # time average of m x'' = -k x - F_d over whole periods is zero, so the
    # mean extension is -<F_d> / k; rebound > bump makes it negative.
    mbs = multibody
    m, k = 50.0, 20000.0
    curve = model_mod.damper_table(
        {"kind": "asymmetric", "bump": 0.3, "rebound": 0.9, "knee": 1e4})
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    base = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                          1.0, [0.01] * 3, Z, Z)
    mass = mbs.create_mobilized_body_quat(2, [0, 0, 0.3], Z, Z, [1, 0, 0, 0],
                                          Z, Z, m, [0.1] * 3, Z, Z)
    for body in (g, base, mass):
        s.Add(body)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, EZ, EZ, g, base))
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, EZ, EZ, g, mass))
    omega = 2.0 * math.pi * 5.0
    s.Add(mbs.create_absolute_driver(
        body=base, ground=g, point=Z, direction=EZ,
        func=mbs.DriveFunction.sine(0.01, omega, 0.0, 0.0)))
    damper = mbs.create_curve_damper(base, mass, Z, Z, k, 0.3, curve[0],
                                     curve[1])
    s.Add(damper)
    s.set_integrator("rk45")
    s.set_tolerances(1e-9, 1e-12)
    s.init()
    period_steps = int(round(2.0 * math.pi / omega / 1e-3))
    # Settle 30 periods, then average over 20.
    for _ in range(30 * period_steps):
        s.solve()
    ext, force = [], []
    for _ in range(20 * period_steps):
        s.solve()
        st = mbs.get_curve_damper_state(damper)
        ext.append(st["length"] - 0.3)
        force.append(st["damper_force"])
    mean_ext, mean_force = np.mean(ext), np.mean(force)
    assert mean_force > 0.0
    assert mean_ext < 0.0
    assert mean_ext == pytest.approx(-mean_force / k, rel=0.02)
    # The shift is not a small numerical effect: a few percent of the
    # stroke.
    assert abs(mean_ext) > 0.02 * np.std(ext)
