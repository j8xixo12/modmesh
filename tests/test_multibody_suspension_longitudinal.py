# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Braking and traction of the full vehicle (``suspension.longitudinal``).

* Statics: the axle load change is M a h / L (h the center of gravity of
  all bodies), for braking and driving, within the pitch of the body.
* Virtual work: the dive, squat, and lift of each corner equal (dF_z a_z +
  F_x a_x - m_u a_x g c_x) / k_w from the K&C rates, for outboard brakes
  (force at the contact patch), inboard brakes, and rear and front drive
  (force at the wheel center).
* The anti fractions agree with the side-view swing arm formula beta
  tan(theta) L / h of Milliken & Milliken (RCVD Ch. 17).
* Brake balance and traction limits with a constant friction equal the
  closed forms of Gillespie (Fundamentals of Vehicle Dynamics, Ch. 3): the
  front locks at d = mu (W_f/W) / (beta - mu h/L), the rear at mu (W_r/W) /
  (1 - beta + mu h/L); rear drive spins at mu (W_r/W) / (1 - mu h/L), front
  drive at mu (W_f/W) / (1 + mu h/L).
"""

import copy

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import car
from solvcon.multibody.suspension import longitudinal as lo

G = 9.81


@pytest.fixture(scope="module")
def vm():
    return car.VehicleModel.load("coilover_car")


@pytest.fixture(scope="module")
def flat(vm):
    out = copy.deepcopy(vm)
    out.mu_sensitivity = 0.0
    return out


CASES = {
    "outboard brakes": dict(a=-0.02, brake_bias=0.65),
    "inboard brakes": dict(a=-0.02, brake_bias=0.65, inboard_brakes=True),
    "rear drive": dict(a=0.02, drive_split=0.0),
    "front drive": dict(a=0.02, drive_split=1.0),
}


@pytest.fixture(scope="module")
def states(vm):
    out = {}
    for name, case in CASES.items():
        kw = dict(case)
        a = kw.pop("a")
        out[name] = lo.steady_state(multibody, vm, [0.0, a], **kw)
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_load_transfer_is_statics(vm, states, case):
    st = states[case]
    a = st["a_x"][1]
    d_front = st["loads"][1][:2].sum() - st["loads"][0][:2].sum()
    expect = -st["mass"] * a * G * st["cg_height"] / vm.wheelbase
    # The pitch moves the bodies over the contact patches: 0.3 to 0.6
    # percent more transfer, in proportion to a_x.
    assert d_front == pytest.approx(expect, rel=0.01)
    assert st["loads"][1].sum() == pytest.approx(st["loads"][0].sum(),
                                                 rel=1e-9)
    # Braking dives the nose, driving lifts it.
    assert (st["pitch"][1] > 0.0) == (a < 0.0)
    assert len(st["poses"]) == len(st["a_x"])


@pytest.mark.parametrize("case", sorted(CASES))
def test_travel_by_virtual_work(vm, states, case):
    st = states[case]
    pred, _ = lo.predicted_travel(multibody, vm, st)
    np.testing.assert_allclose(st["travel"][1], pred[1], rtol=0.015)


def test_anti_fractions_against_the_swing_arm_formula(vm, states):
    geo = lo.anti_geometry(multibody, vm)
    _, brake = lo.predicted_travel(multibody, vm, states["outboard brakes"])
    _, drive = lo.predicted_travel(multibody, vm, states["rear drive"])
    # Anti-dive and anti-lift of 65 / 35 outboard brakes, anti-squat of a
    # rear drive: the RCVD formula uses the sprung CG height and the static
    # axle loads, so a few percent apart.
    assert brake[1][0] == pytest.approx(0.65 * geo["front"]["anti_brake"],
                                        rel=0.04)
    assert brake[1][2] == pytest.approx(0.35 * geo["rear"]["anti_brake"],
                                        rel=0.04)
    assert drive[1][2] == pytest.approx(geo["rear"]["anti_drive"],
                                        rel=0.04)
    # Inboard brakes act at the wheel center: less anti than outboard.
    _, inboard = lo.predicted_travel(multibody, vm, states["inboard brakes"])
    assert inboard[1][0] < brake[1][0]


@pytest.mark.parametrize("bias", [0.55, 0.65, 0.8])
def test_brake_limit_equals_the_closed_form(flat, bias):
    decel = list(np.round(np.arange(0.0, 1.4, 0.01), 3))
    out = lo.brake_balance(multibody, flat, bias, decel)
    w_f, w_r = car.axle_loads(flat)
    w = w_f + w_r
    hl = out["state"]["cg_height"] / flat.wheelbase
    mu = flat.mu
    d_f = mu * (w_f / w) / (bias - mu * hl) if bias > mu * hl else np.inf
    d_r = mu * (w_r / w) / (1.0 - bias + mu * hl)
    assert out["limit"] == pytest.approx(min(d_f, d_r), rel=5e-3)
    assert out["locks"] == ("front" if d_f < d_r else "rear")
    # The ideal bias puts the dynamic front load share on the front.
    k = int(np.argmin(np.abs(out["decel"] - 0.8)))
    assert out["ideal_bias"][k] == pytest.approx(
        (w_f + w * 0.8 * hl) / w, rel=0.01)


@pytest.mark.parametrize("split", [0.0, 1.0])
def test_traction_limit_equals_the_closed_form(flat, split):
    accel = list(np.round(np.arange(0.0, 1.0, 0.01), 3))
    out = lo.traction_limit(multibody, flat, split, accel)
    w_f, w_r = car.axle_loads(flat)
    w = w_f + w_r
    hl = out["state"]["cg_height"] / flat.wheelbase
    mu = flat.mu
    expect = (mu * (w_r / w) / (1.0 - mu * hl) if 0.0 == split
              else mu * (w_f / w) / (1.0 + mu * hl))
    assert out["limit"] == pytest.approx(expect, rel=5e-3)
    assert out["spins"] == ("rear" if 0.0 == split else "front")


def test_load_sensitive_tires_move_the_brake_limit(vm, flat):
    # The friction falls with the load: the loaded front loses grip, the
    # unloaded rear gains it.  A front-limited bias stops sooner, a
    # rear-limited one later.
    for bias, axle in ((0.8, "front"), (0.6, "rear")):
        sensitive = lo.brake_balance(multibody, vm, bias)
        constant = lo.brake_balance(multibody, flat, bias)
        assert constant["locks"] == axle
        if "front" == axle:
            assert sensitive["limit"] < constant["limit"]
        else:
            assert sensitive["limit"] > constant["limit"]
