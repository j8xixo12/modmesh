# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The replay of a measured acceleration history on the full vehicle
(``suspension.replay``).

* A slow ramp to a steady turn, braking, or traction gives the roll, the
  pitch, the travel, and the tire loads of the steady rigs
  (``car.steady_state``, ``longitudinal.steady_state``; the lateral one
  meets the roll center model of Milliken, RCVD Ch. 18).
* With the contact patches held sideways, a step of a_y rings at the
  natural frequency of the roll model, omega^2 = (K_phi - W_s h') / (I_xx
  + m_s h'^2), with K_phi and h' of ``car.closed_form``.
* ``compare`` recovers the gain and the delay of a scaled, shifted rate.

The replay of a measured steering and speed history (``steer_replay``):

* Without the speed control it is the step steer of ``car.step_steer``.
* The speed follows a braking and a traction ramp; the pitch and the tire
  loads meet the longitudinal rig, and the car stays straight (the
  longitudinal tire force is along the travel on both sides).
* A demand past the grip does not lock a wheel; a spin stops the run.
"""

import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import car, longitudinal, replay

G = 9.81
RAMP = np.array([0.0, 1.0, 5.0, 8.0])


@pytest.fixture(scope="module")
def road_car():
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return car.VehicleModel.load("road_car")


def test_slow_turn_meets_the_steady_rig(road_car):
    run = replay.acceleration_replay(multibody, road_car, RAMP, np.zeros(4),
                                     np.array([0.0, 0.0, 0.5, 0.5]) * G)
    ss = car.steady_state(multibody, road_car, (0.0, 0.25, 0.5))
    assert run["roll"][-1] - run["roll0"] == pytest.approx(
        ss["roll"][-1] - ss["roll"][0], rel=0.01)
    np.testing.assert_allclose(run["loads"][-1], ss["loads"][-1], rtol=0.01)
    # The support carries next to nothing of the 7.9 kN side force.
    assert run["support"].max() < 100.0


@pytest.mark.parametrize("a, share", [(-0.5, 0.65), (0.3, 0.0)])
def test_slow_braking_and_traction_meet_the_longitudinal_rig(road_car, a,
                                                             share):
    run = replay.acceleration_replay(
        multibody, road_car, RAMP, np.array([0.0, 0.0, a, a]) * G,
        np.zeros(4), brake_bias=share, drive_split=share)
    st = longitudinal.steady_state(multibody, road_car, [0.0, a],
                                   brake_bias=share, drive_split=share)
    assert run["pitch"][-1] - run["pitch0"] == pytest.approx(
        st["pitch"][-1], rel=0.01)
    np.testing.assert_allclose(run["travel"][-1], st["travel"][-1],
                               atol=1e-4)


def _roll_frequency(run):
    """The natural frequency (rad/s) of the roll after the step, from the
    half period and the half-cycle decrement of the ringing."""
    x = run["roll"] - run["roll"][-1]
    t = run["time"]
    k0 = int(np.searchsorted(t, 0.52))
    ext = [k for k in range(k0 + 1, len(x) - 1)
           if (x[k] - x[k - 1]) * (x[k + 1] - x[k]) < 0.0]
    period = 2.0 * (t[ext[1]] - t[ext[0]])
    dec = math.log(abs(x[ext[0]] / x[ext[1]]))
    zeta = dec / math.sqrt(math.pi ** 2 + dec ** 2)
    return 2.0 * math.pi / period / math.sqrt(1.0 - zeta ** 2), zeta


@pytest.mark.parametrize("stem", ["road_car", "coilover_car"])
def test_roll_step_rings_at_the_roll_model_frequency(road_car, stem):
    vm = car.VehicleModel.load(stem)
    step = np.array([0.0, 0.5, 0.51, 4.0])
    # A slow speed makes the slip damping C_alpha / V of the patches large:
    # the patches stay put, as the roll model assumes.
    run = replay.acceleration_replay(multibody, vm, step, np.zeros(4),
                                     np.array([0.0, 0.0, 0.3, 0.3]) * G,
                                     record_dt=1e-3, speed=1.0)
    omega, zeta = _roll_frequency(run)
    cf = car.closed_form(multibody, vm)
    k_phi, h = sum(cf["roll_stiffness"]), cf["h_prime"]
    m = vm.sprung_mass
    closed = math.sqrt((k_phi - m * G * h) / (vm.sprung_inertia[0]
                                              + m * h * h))
    assert omega == pytest.approx(closed, rel=0.02)
    assert 0.1 < zeta < 0.7
    # At road speed the patches move a little: more damping, a higher
    # frequency (the free car rolls about its mass center, 30 percent).
    fast = replay.acceleration_replay(multibody, vm, step, np.zeros(4),
                                      np.array([0.0, 0.0, 0.3, 0.3]) * G,
                                      record_dt=1e-3, speed=20.0)
    omega_fast, zeta_fast = _roll_frequency(fast)
    assert closed < omega_fast < 1.3 * closed
    assert zeta_fast > zeta


def test_compare_recovers_gain_and_delay():
    t = np.linspace(0.0, 10.0, 1001)
    sim = np.sin(2.0 * math.pi * 0.5 * t)
    measured = 0.8 * np.sin(2.0 * math.pi * 0.5 * (t - 0.1))
    c = replay.compare(t, sim, t, measured)
    assert c["lag"] == pytest.approx(0.1, abs=0.011)
    assert c["gain"] == pytest.approx(0.8 * math.cos(2 * math.pi * 0.05),
                                      rel=1e-3)
    flat = replay.compare(t, sim, t, np.zeros_like(t))
    assert (flat["correlation"], flat["lag"]) == (0.0, 0.0)


def test_steer_replay_without_speed_control_is_the_step_steer(road_car):
    t = np.linspace(0.0, 2.0, 201)
    rack = 0.004 * np.clip((t - 0.5) / 0.2, 0.0, 1.0)
    run = replay.steer_replay(multibody, road_car, t, rack, 30.0 + 0.0 * t,
                              speed_gain=0.0)
    ref = car.step_steer(multibody, road_car, 30.0, 0.004, duration=2.0)
    assert run["departed"] is None
    assert np.abs(run["demand"]).max() < 1e-12
    assert run["yaw_rate"][-1] == pytest.approx(ref["yaw_rate"][-1],
                                                rel=1e-9)
    # The speed along the heading: the step steer gives the whole speed.
    assert run["speed"][-1] == pytest.approx(
        ref["speed"][-1] * math.cos(run["sideslip"][-1]), rel=1e-6)
    # The lateral acceleration is V r in the settled turn.
    assert run["ay"][-20:].mean() == pytest.approx(
        (run["speed"] * run["yaw_rate"])[-20:].mean(), rel=0.02)
    assert len(run["poses"]) == len(run["pose_times"])


def test_steer_replay_follows_the_speed(road_car):
    t = np.linspace(0.0, 7.0, 351)
    speed = (30.0 - 12.0 * np.clip((t - 1.0) / 2.0, 0.0, 1.0)
             + 4.0 * np.clip((t - 4.0) / 2.0, 0.0, 1.0))
    run = replay.steer_replay(multibody, road_car, t, np.zeros_like(t),
                              speed, brake_bias=0.6, drive_split=0.0,
                              record_dt=0.02)
    assert np.abs(run["speed"] - run["target_speed"]).max() < 0.05
    # Both sides brake: the car stays straight.
    assert np.abs(run["sideslip"]).max() < 1e-6
    assert np.abs(run["yaw_rate"]).max() < 1e-6
    rig = longitudinal.steady_state(multibody, road_car,
                                    [0.0, -6.0 / G, 2.0 / G],
                                    brake_bias=0.6, drive_split=0.0)
    rest = (run["time"] > 0.5) & (run["time"] < 0.9)
    for k, (lo, hi, a) in enumerate(((2.5, 2.95, -6.0), (5.5, 5.95, 2.0))):
        part = (run["time"] > lo) & (run["time"] < hi)
        assert run["ax"][part].mean() == pytest.approx(a, abs=0.02)
        np.testing.assert_allclose(run["loads"][part].mean(axis=0),
                                   rig["loads"][k + 1], rtol=0.015)
        # The pitch from rest: the brake at the patch, the drive through
        # the shaft (the anti-squat of the rig).
        pitch = run["pitch"][part].mean() - run["pitch"][rest].mean()
        assert pitch == pytest.approx(rig["pitch"][k + 1], rel=0.05)


def test_steer_replay_keeps_the_demand_inside_the_grip(road_car):
    # 2 g of braking asked for: the tires give what they have, no wheel
    # locks (a locked axle would drop to the sliding level and the car
    # would turn), and every tire brakes at the same F_x / F_z.
    t = np.linspace(0.0, 2.0, 101)
    speed = 40.0 - 2.0 * G * np.clip(t - 0.5, 0.0, 1.0)
    run = replay.steer_replay(multibody, road_car, t, np.zeros_like(t),
                              speed, brake_bias=None, record_dt=0.02)
    part = (run["time"] > 1.0) & (run["time"] < 1.4)
    mu = road_car.tire(True).friction(4000.0)[0]
    assert -mu * G < run["ax"][part].mean() < -0.7 * mu * G
    np.testing.assert_allclose(run["demand"][part, 0],
                               run["demand"][part, 1], rtol=0.1)
    assert np.abs(run["sideslip"]).max() < 1e-6


def test_steer_replay_stops_at_a_spin(road_car):
    t = np.linspace(0.0, 4.0, 201)
    rack = 0.03 * np.clip((t - 0.5) / 0.2, 0.0, 1.0)
    run = replay.steer_replay(multibody, road_car, t, rack, 35.0 + 0.0 * t,
                              record_dt=0.02,
                              stop_sideslip=math.radians(15.0))
    assert run["departed"] is not None and run["departed"] < 4.0
    assert run["time"][-1] <= run["departed"]
    assert abs(run["sideslip"][-1]) < math.radians(16.0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
