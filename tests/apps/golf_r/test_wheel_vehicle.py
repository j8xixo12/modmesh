# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Wheel model checks for brakes, load transfer and driveline."""

import math
import unittest
import numpy as np
from solvcon.apps.golf_r import wheel_vehicle as wv
from solvcon.apps.golf_r.bicycle import TIRE, Vehicle, gear_ratio
from solvcon.apps.golf_r.tire import GRAVITY

CURVE = np.array([[800, 100], [8000, 100]], dtype='float64')
SPEED = 120 / 3.6


class WheelVehicleTest(unittest.TestCase):
    def test_full_brake_from_120_kmh_stops_and_stays_stopped(self):
        # The old brake torque had no sign: the wheels spun backward to
        # about -720 m/s, and the model raised below 1 m/s.
        time, states = wv.simulate(
            wv.initial_state(SPEED),
            lambda t: (0, 1, 0, 7),
            5.0,
            dt=0.001,
            curve=CURVE,
        )
        self.assertGreaterEqual(states[:, 7:9].min(), 0)
        stop = int(np.argmax(states[:, 3] <= 0))
        self.assertGreater(stop, 0)
        self.assertLess(time[stop], 4.0)
        np.testing.assert_array_equal(states[stop:, 3], 0)
        np.testing.assert_array_equal(states[stop:, 7:9], 0)
        self.assertEqual(states[stop, 0], states[-1, 0])
        # Locked tires at mu about 1: distance near V^2 / (2 g).
        distance = SPEED**2 / (2 * GRAVITY)
        self.assertAlmostEqual(states[stop, 0] / distance, 1, delta=0.05)

    def test_brake_holds_a_wheel_at_rest(self):
        inertia = np.diag([3.0, 3.0])
        capacity = np.array([1000.0, 500.0])
        accel, brake = wv.wheel_accelerations(
            inertia, np.array([400.0, -200.0]), capacity, np.zeros(2)
        )
        np.testing.assert_array_equal(accel, [0, 0])
        np.testing.assert_allclose(brake, [-400, 200])
        # The torque beyond the capacity turns the wheel; the brake
        # resists with its full capacity.
        accel, brake = wv.wheel_accelerations(
            inertia, np.array([1600.0, 0.0]), capacity, np.zeros(2)
        )
        np.testing.assert_allclose(accel, [200, 0])
        np.testing.assert_allclose(brake, [-1000, 0])
        # A turning wheel gets the full brake torque against rotation.
        accel, brake = wv.wheel_accelerations(
            inertia, np.zeros(2), capacity, np.array([10.0, 10.0])
        )
        np.testing.assert_allclose(brake, -capacity)

    def test_braking_moves_load_to_front(self):
        p = Vehicle()
        state = wv.initial_state(SPEED)
        state[7:11] = [0, 0, -1, -1]
        f = wv.forces(p, TIRE, state, (0, 1, 0, 7), CURVE)
        weight = p.mass * GRAVITY
        self.assertLess(f['ax'], -0.9 * GRAVITY)
        expected = weight * p.front_fraction - p.mass * f['ax'] * (
            p.cg_height / p.wheelbase
        )
        self.assertAlmostEqual(f['front_load'], expected, delta=1e-2)
        self.assertGreater(f['front_load'], 1.25 * weight * p.front_fraction)

    def test_engine_rpm_follows_driven_wheel_speed(self):
        p = Vehicle()
        omega = SPEED / p.radius
        for gear in range(1, len(p.gears) + 1):
            rpm = wv.engine_rpm(p, omega, omega, gear)
            expected = (
                p.gears[gear - 1] * p.final_drive * omega * 60 / (2 * math.pi)
            )
            self.assertAlmostEqual(rpm, expected, places=9)
            # AC drivetrain.ini ratios: gears 1 to 3 cannot reach 120 km/h
            # below the 8000 rpm limiter; 3rd tops out at 113 km/h.
            if gear >= 4:
                self.assertTrue(p.idle_rpm < rpm < p.limiter_rpm)
            else:
                self.assertGreater(rpm, p.limiter_rpm)
        # A spinning wheel sets the rpm, not the vehicle speed.
        state = wv.initial_state(SPEED)
        state[7:9] = 1.2 * omega
        f = wv.forces(p, TIRE, state, (1, 0, 0, 5), CURVE)
        self.assertAlmostEqual(
            f['rpm'], wv.engine_rpm(p, 1.2 * omega, 1.2 * omega, 5)
        )
        # Past the limiter the fuel cut leaves no drive torque.
        idle = wv.forces(p, TIRE, wv.initial_state(SPEED), (0, 0, 0, 2), CURVE)
        full = wv.forces(p, TIRE, wv.initial_state(SPEED), (1, 0, 0, 2), CURVE)
        np.testing.assert_array_equal(idle['wheel_accel'], full['wheel_accel'])

    def test_reflected_driveline_inertia(self):
        p = Vehicle()
        omega = np.array([90.0, 80.0])
        for gear in (1, 7):
            inertia = wv.inertia_matrix(p, gear)
            engine = wv.engine_rpm(p, *omega, gear) * 2 * math.pi / 60
            energy = p.wheel_inertia * omega @ omega + 0.5 * (
                p.engine_inertia + p.gearbox_inertia
            ) * engine**2
            self.assertAlmostEqual(0.5 * omega @ inertia @ omega, energy, 6)
        # In 1st gear the reflected inertia dominates the wheels.
        reflected = (p.engine_inertia + p.gearbox_inertia) * (
            gear_ratio(p, 1) ** 2
        )
        self.assertGreater(reflected, 10 * p.wheel_inertia)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
