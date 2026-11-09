# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The full vehicle (``suspension.car``): steady-state cornering against
the roll center model of Milliken & Milliken (RCVD, 1995, Ch. 18) and
Gillespie (1992, Ch. 7), and a step steer against the linear bicycle
model.

* Statics: the side forces F_y = F_z a_y / g sum to the D'Alembert load and
  hold the yaw moment, so the soft chassis support carries nothing.
* With light unsprung masses (the roll center model sends their side load
  straight to the tires, an independent suspension sends it through the
  links), the engine roll gradient and LLTD equal the closed form, with and
  without anti-roll bars.
* The road car with its real unsprung masses: the LLTD and the axle load
  transfers still agree; the roll gradient is larger by the unsprung side
  load through the links.
* Step steer at speed: the steady state satisfies delta_f - delta_r = L/R +
  (W_f / C_f - W_r / C_r) a_y / g, and the yaw rate gain is V / (L + K V^2 /
  g) with the understeer gradient K of the static rig (tires and roll
  steer).
"""

import copy

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import car

G = 9.81


def _light_unsprung(vm):
    vm = copy.deepcopy(vm)
    for model in (vm.front, vm.rear):
        model.upright_mass = 1e-2
        model.rack_mass = 1e-3
        if model.has_rocker:
            model.rocker_mass = 1e-3
        for arm in model.arms.values():
            arm["mass"] = 1e-3
    return vm


def _race_car():
    # Two race front corners (the rear rack held), both with their bars:
    # the anti-roll bar path of the full vehicle.
    return car.VehicleModel({
        "name": "race corners front and rear", "front": "race_front_corner",
        "rear": "race_front_corner", "wheelbase": 3000.0,
        "sprung_mass": 400.0, "front_fraction": 0.45, "cg_height": 300.0,
        "tire": {"cornering_stiffness": [1500.0, 1600.0]}})


@pytest.fixture(scope="module")
def road_car():
    return car.VehicleModel.load("road_car")


def test_side_forces_balance_the_turn(road_car):
    ss = car.steady_state(multibody, road_car, (0.0, 0.1, 0.2))
    w_f, w_r = car.axle_loads(road_car)
    for k, a in enumerate(ss["a_y"]):
        assert ss["support"][k] < 1e-2
        np.testing.assert_allclose(ss["side_force"][k],
                                   [w_f * a, w_r * a], rtol=1e-3, atol=1e-6)
        assert np.sum(ss["loads"][k]) == pytest.approx(w_f + w_r, rel=1e-6)
    # A left turn rolls the body to the right and loads the right wheels.
    assert ss["roll"][-1] > 0.0
    assert np.all(ss["load_transfer"][1:] > 0.0)


@pytest.mark.parametrize("make", [
    lambda: car.VehicleModel.load("road_car"), _race_car])
def test_roll_center_model_with_light_unsprung_masses(make):
    vm = _light_unsprung(make())
    ss = car.steady_state(multibody, vm, (0.0, 0.02, 0.05))
    cf = car.closed_form(multibody, vm)
    assert ss["roll_gradient"] == pytest.approx(cf["roll_gradient"],
                                                rel=0.02)
    assert ss["lltd"] == pytest.approx(cf["lltd"], rel=0.02)
    np.testing.assert_allclose(ss["load_transfer_gradient"],
                               cf["load_transfer_gradient"], rtol=0.03)


def test_anti_roll_bars_carry_roll_moment():
    vm = _light_unsprung(_race_car())
    cf = car.closed_form(multibody, vm)
    bare = copy.deepcopy(vm)
    bare.front.arb = bare.rear.arb = None
    cf_bare = car.closed_form(multibody, bare)
    ss_bare = car.steady_state(multibody, bare, (0.0, 0.02, 0.05))
    assert cf["roll_stiffness"][0] > 1.2 * cf_bare["roll_stiffness"][0]
    assert ss_bare["roll_gradient"] == pytest.approx(
        cf_bare["roll_gradient"], rel=0.02)
    assert ss_bare["roll_gradient"] > cf["roll_gradient"]


def test_road_car_against_the_roll_center_model(road_car):
    # The light-unsprung test above isolates this difference.
    ss = car.steady_state(multibody, road_car, (0.0, 0.02, 0.05))
    cf = car.closed_form(multibody, road_car)
    assert ss["lltd"] == pytest.approx(cf["lltd"], rel=0.02)
    np.testing.assert_allclose(ss["load_transfer_gradient"],
                               cf["load_transfer_gradient"], rtol=0.03)
    # About 70 kg of uprights at the wheel center height load the body
    # through the links: about 5 percent more roll than the closed form.
    assert 1.0 < ss["roll_gradient"] / cf["roll_gradient"] < 1.10
    # A soft car without bars: 5 to 8 degrees per g.
    assert 5.0 < ss["roll_gradient"] < 8.0


@pytest.fixture(scope="module")
def step_runs(road_car):
    ss = car.steady_state(multibody, road_car, (0.0, 0.05, 0.1))
    kus = car.understeer_gradient(road_car, ss["roll_steer"])
    runs = {}
    for speed, rack in ((20.0, 0.002), (30.0, 0.0012)):
        runs[speed] = (rack, car.step_steer(multibody, road_car, speed, rack,
                                            duration=4.0))
    return kus, runs


@pytest.mark.parametrize("speed", [20.0, 30.0])
def test_step_steer_steady_state_relation(road_car, step_runs, speed):
    # The relation of the linear bicycle model (Gillespie Eq. 6-15).
    _, runs = step_runs
    st = runs[speed][1]["steady"]
    w_f, w_r = car.axle_loads(road_car)
    c_f, c_r = (2.0 * c for c in road_car.cornering_stiffness)
    lhs = st["steer"][0] - st["steer"][1]
    rhs = road_car.wheelbase * st["yaw_rate"] / st["speed"] \
        + (w_f / c_f - w_r / c_r) * st["lateral_acceleration"] / G
    # The axle steer weighs the wheel angles (with the static toe of the
    # spin axis) by the cornering stiffness at the wheel loads: a
    # load-sensitive tire turns the toe of an axle into a side force under
    # load transfer.  The plain mean of the upright headings missed the
    # toe: 5 percent at 20 m/s and 10 percent at 30 m/s with this tire.
    assert lhs == pytest.approx(rhs, rel=0.01)
    # The run reached a steady state.
    r = runs[speed][1]["yaw_rate"]
    assert abs(r[-1] - r[-1000]) < 1e-3 * abs(r[-1])


@pytest.mark.parametrize("speed", [20.0, 30.0])
def test_step_steer_yaw_gain(road_car, step_runs, speed):
    kus, runs = step_runs
    rack, run = runs[speed]
    st = run["steady"]
    assert abs(st["lateral_acceleration"]) < 0.2 * G
    delta = car.kinematic_steer(multibody, road_car, rack)
    v = st["speed"]
    gain = v / (road_car.wheelbase + kus * v * v / G)
    assert st["yaw_rate"] == pytest.approx(gain * delta, rel=0.01)
    # An understeering car: below the neutral steer gain V / L.
    assert abs(st["yaw_rate"]) < abs(v / road_car.wheelbase * delta)
    assert kus > 0.0


def test_vehicle_poses_for_the_view(road_car, step_runs):
    _, runs = step_runs
    run = runs[20.0][1]
    assert len(run["poses"]) == pytest.approx(4.0 / 0.02, abs=1)
    # Following the car: the chassis stays over its design position and
    # heads +x while the car turns on the road.
    last = run["poses"][-1]
    pos, quat = last["chassis"]
    np.testing.assert_allclose(pos[:2], last["cg"][:2], atol=1e-9)
    from solvcon.multibody.suspension.metrics import rot
    heading = rot(quat) @ np.array([1.0, 0.0, 0.0])
    assert abs(heading[1]) < 1e-9
    assert set(last["corners"]) == set(car.CORNERS)
    ss = car.steady_state(multibody, road_car, (0.0, 0.1))
    assert 2 == len(ss["poses"])
