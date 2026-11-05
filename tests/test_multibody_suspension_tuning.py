# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The setup of a corner (``suspension.tuning``), the coilover library
models, the anti-roll bar on a direct-acting corner, and the handling
diagram of the full vehicle.

* The readouts at any setup (spring, ride height, clicks) are the engine:
  the damped ride and wheel hop modes equal the modal analysis of the
  quarter car at its static travel, which equals the ride height setting.
* The coilover models keep the validated geometry of their parents
  (``macpherson_front``: Yang 2019 against ADAMS; ``five_link_rear``:
  Grzyb & Kowalski 2001).
* The bar roll stiffness of the axle rig equals k (d theta / d phi)^2 by
  virtual work, theta the bar twist and phi the roll.
"""

import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import axle, kc, tuning, vehicle
from solvcon.multibody.suspension.metrics import rot
from solvcon.multibody.suspension.model import CornerModel

DEG = 180.0 / math.pi
COILOVERS = {"macpherson_coilover": "macpherson_front",
             "five_link_coilover": "five_link_rear"}


def _mass(model):
    return model.raw.get("vehicle", {}).get("corner_mass") or 200.0


def test_baseline_and_bad_setups():
    model = CornerModel.load("macpherson_coilover")
    base = tuning.baseline(model)
    assert base == {"spring_rate": 60.0, "ride_height": 0.0,
                    "tire_rate": 230.0, "bump_stop_gap": 50.0,
                    "bump_clicks": 12, "rebound_clicks": 16,
                    "arb_rate": pytest.approx(1100.0 / DEG)}
    same = tuning.apply(model, base)
    assert same.spring_k == model.spring_k
    assert same.damper_curve == model.damper_curve
    for bad in ({"camber_plate": 1.0}, {"bump_clicks": 33},
                {"spring_rate": 0.0}, {"bump_stop_gap": -5.0}):
        with pytest.raises(ValueError):
            tuning.apply(model, bad)
    # A model without clicks takes the damper slopes.
    plain = CornerModel.load("macpherson_front")
    assert "bump_clicks" not in tuning.baseline(plain)
    asym = tuning.apply(plain, {"damper_bump": 2.0, "damper_rebound": 5.0})
    assert asym.spring_c == pytest.approx(3.5e3)


def test_click_damper_runs_from_soft_to_firm():
    model = CornerModel.load("macpherson_coilover")
    spec = model.damper_spec
    soft = tuning.apply(model, {"bump_clicks": 0, "rebound_clicks": 0})
    firm = tuning.apply(model, {"bump_clicks": 32, "rebound_clicks": 32})
    from solvcon.multibody.suspension.model import curve_slope
    assert curve_slope(soft.damper_curve, -1e-9) == pytest.approx(
        spec["soft"]["bump"][0] * 1e3)
    assert curve_slope(firm.damper_curve, 1e-9) == pytest.approx(
        spec["firm"]["rebound"][0] * 1e3)
    ratios = [tuning.readouts(multibody, tuning.apply(
        model, {"rebound_clicks": c}), 400.0,
        axle_roll=False)["damping_ratio_rebound"] for c in (0, 8, 16, 32)]
    assert all(b > a for a, b in zip(ratios[:-1], ratios[1:]))


@pytest.mark.parametrize("stem, setup", [
    ("macpherson_coilover", {}),
    ("macpherson_coilover", {"ride_height": -20.0, "spring_rate": 80.0}),
    ("macpherson_coilover", {"ride_height": 15.0, "bump_clicks": 30,
                             "rebound_clicks": 4}),
    ("five_link_coilover", {"ride_height": -25.0, "rebound_clicks": 28}),
])
def test_readouts_equal_the_engine_at_ride_height(stem, setup):
    model = tuning.apply(CornerModel.load(stem), setup)
    mass = _mass(model)
    out = tuning.readouts(multibody, model, mass, axle_roll=False)
    s, h = vehicle.quarter_car(multibody, model, mass)
    assert s.find_equilibrium(tol=1e-10)["converged"]
    pos = np.asarray(s.get_body_positions())
    travel = pos[h["corner"].nums["upright"]][2] - pos[vehicle.CHASSIS][2] \
        - model.hp["wheel_center"][2]
    assert travel * 1e3 == pytest.approx(-setup.get("ride_height", 0.0),
                                         abs=1e-6)
    modal = multibody.modal_analysis(s)
    freq = np.asarray(modal.frequencies)
    zeta = np.asarray(modal.damping_ratios)
    order = np.argsort(freq)
    np.testing.assert_allclose([out["ride_mode"], out["hop_mode"]],
                               freq[order], rtol=1e-3)
    np.testing.assert_allclose(
        [out["ride_mode_damping"], out["hop_mode_damping"]], zeta[order],
        rtol=1e-3)
    # The static alignment is the K&C curve at the static travel.
    sweep = kc.bump_sweep(multibody, model, [travel])
    assert out["camber"] == pytest.approx(sweep["camber"][0] * DEG,
                                          abs=1e-9)
    # Milliken's ride frequency (undamped, the corner bodies left out) is
    # near the engine ride mode.
    assert out["ride_frequency"] == pytest.approx(out["ride_mode"],
                                                  rel=0.06)


def test_lowering_moves_the_geometry():
    # Lowering a MacPherson corner drops its roll center fast and adds
    # negative camber, the known side effect of lowering springs.
    model = CornerModel.load("macpherson_coilover")
    base = tuning.readouts(multibody, model, 400.0, axle_roll=False)
    low = tuning.readouts(multibody, tuning.apply(
        model, {"ride_height": -30.0}), 400.0, axle_roll=False)
    assert low["roll_center"] < base["roll_center"] - 30.0
    assert low["camber"] < base["camber"]
    assert low["static_travel"] == pytest.approx(30.0)


@pytest.mark.parametrize("stem", sorted(COILOVERS))
def test_coilover_models_keep_the_validated_geometry(stem):
    travels = list(np.linspace(-0.06, 0.06, 13))
    got = kc.bump_sweep(multibody, CornerModel.load(stem), travels)
    ref = kc.bump_sweep(multibody, CornerModel.load(COILOVERS[stem]),
                        travels)
    for key in ("camber", "toe", "roll_center", "motion_ratio"):
        np.testing.assert_allclose(got[key], ref[key], atol=1e-12)


@pytest.mark.parametrize("stem", sorted(COILOVERS))
def test_bar_roll_stiffness_by_virtual_work(stem):
    model = CornerModel.load(stem)
    h = 2e-3
    with_bar = axle.roll_sweep(multibody, model, [-h, 0.0, h], True)
    bare = axle.roll_sweep(multibody, model, [-h, 0.0, h], False)
    raw = with_bar["raw"]
    nums = with_bar["axle"]["arb_nums"]
    axis = model.arb["axis"] / np.linalg.norm(model.arb["axis"])

    def twist(k):
        # The relative rotation of the two half-bars about the bar axis.
        r = rot(raw["quaternions"][k, nums[0]]).T \
            @ rot(raw["quaternions"][k, nums[1]])
        vec = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0],
                              r[1, 0] - r[0, 1]])
        return float(vec @ axis)

    rate = (twist(2) - twist(0)) / (with_bar["roll"][2]
                                    - with_bar["roll"][0])
    expect = model.arb["k"] * rate * rate
    got = with_bar["roll_stiffness"][1] - bare["roll_stiffness"][1]
    assert got == pytest.approx(expect, rel=1e-3)
