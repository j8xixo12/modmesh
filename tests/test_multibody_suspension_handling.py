# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The handling diagram of the full vehicle (``suspension.car``) with the
load-sensitive tire, and the vehicle setup.

* Its slope at a_y -> 0 is the understeer gradient, also for a car with
  static toe: the roll steer of the rig weighs the wheel angles by the
  cornering stiffness at the wheel loads, which holds the side force of the
  toe under load transfer.
* A stiffer front bar adds understeer at the limit and a stiff rear bar
  turns the limit to the rear axle (Milliken & Milliken, RCVD Ch. 18).
* The tire grip bounds the limit.
"""

import copy

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import car, tuning


def _race_car():
    return car.VehicleModel({
        "name": "race corners front and rear", "front": "race_front_corner",
        "rear": "race_front_corner", "wheelbase": 3000.0,
        "sprung_mass": 400.0, "front_fraction": 0.45, "cg_height": 300.0,
        "tire": {"cornering_stiffness": [500.0, 560.0], "mu": 1.6}})


def test_handling_slope_is_the_understeer_gradient():
    vm = _race_car()
    hd = car.handling_diagram(multibody, vm, [0.0, 0.02, 0.04])
    ss = car.steady_state(multibody, vm, (0.0, 0.02, 0.04))
    kus = car.understeer_gradient(vm, ss["roll_steer"])
    slope = np.polyfit(hd["a_y"], hd["understeer_angle"], 1)[0]
    assert slope == pytest.approx(kus, rel=0.02)


def test_handling_slope_with_static_toe():
    # The road car has static toe on both axles.
    vm = car.VehicleModel.load("road_car")
    hd = car.handling_diagram(multibody, vm, [0.0, 0.02, 0.04])
    ss = car.steady_state(multibody, vm, (0.0, 0.02, 0.04))
    kus = car.understeer_gradient(vm, ss["roll_steer"])
    slope = np.polyfit(hd["a_y"], hd["understeer_angle"], 1)[0]
    assert slope == pytest.approx(kus, rel=0.02)


@pytest.fixture(scope="module")
def coilover_car():
    return car.VehicleModel.load("coilover_car")


def test_bars_move_the_limit_balance(coilover_car):
    def at(vm, a):
        hd = car.handling_diagram(multibody, vm)
        k = int(np.argmin(np.abs(hd["a_y"] - a)))
        return hd, hd["understeer_angle"][k]

    # Near the limit (1.05 g) the bars move the balance through the load
    # transfer and the falling friction of the loaded tires.  Below about
    # 0.8 g the change of the roll steer is larger: the cornering
    # stiffness grows with F_z^0.88, nearly linear, so the load transfer
    # costs little stiffness there.
    base, u_base = at(coilover_car, 0.9)
    stiff_front, u_front = at(coilover_car.with_setup(
        front={"arb_rate": 40.0}), 0.9)
    stiff_rear, u_rear = at(coilover_car.with_setup(
        front={"arb_rate": 5.0}, rear={"arb_rate": 80.0}), 0.9)
    assert u_front > u_base > u_rear
    assert base["limited_by"] == "front"
    assert stiff_rear["limited_by"] == "rear"
    for hd in (base, stiff_front, stiff_rear):
        # The grip of the tires bounds the car: mu 1.1 at the static load.
        assert hd["limit"] < coilover_car.mu
        assert np.all(hd["grip_use"] <= 1.0 + 1e-9)


def test_vehicle_setup_file(coilover_car):
    front = tuning.baseline(coilover_car.front)
    assert front["spring_rate"] == 60.0
    soft = coilover_car.with_setup(front={"spring_rate": 40.0})
    assert soft.front.spring_k == pytest.approx(40e3)
    assert coilover_car.front.spring_k == pytest.approx(60e3)
    raw = copy.deepcopy(coilover_car.raw)
    raw["setup"] = {"front": {"ride_height": -10.0}, "rear": {}}
    low = car.VehicleModel(raw, base_dir=car.DATA_DIR)
    assert low.front.static_travel == pytest.approx(0.01)
