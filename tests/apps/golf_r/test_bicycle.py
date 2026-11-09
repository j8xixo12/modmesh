# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Planar bicycle model checks with synthetic data only."""

import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from solvcon.apps.golf_r.bicycle import (
    TIRE,
    Vehicle,
    derivative,
    drivetrain,
    forces,
    simulate,
    vehicle_from_data,
)
from solvcon.apps.golf_r.tire import GRAVITY, Tire, force

CURVE = np.array([[800, 100], [8000, 100]], dtype='float64')


class BicycleTest(unittest.TestCase):
    def test_full_braking_rear_load_uses_delivered_force(self):
        # With mu = 1 and no load sensitivity, the locked axles deliver
        # -m g; W_r = m g a / L + m a_x h / L with that a_x.  The old
        # requested-force transfer gave W_r = 28 N instead of 3070 N.
        p, tire = Vehicle(), Tire(friction=1.0, load_sensitivity=0)
        state = [0, 0, 0, 120 / 3.6, 0, 0, 0]
        f = forces(p, state, 0, 1, 0, 7, CURVE, tire)
        weight = p.mass * GRAVITY
        ax = (f['front_x'] + f['rear_x'] - f['drag']) / p.mass
        self.assertAlmostEqual(ax, f['ax'], places=5)
        self.assertAlmostEqual(f['front_x'] + f['rear_x'], -weight, 3)
        a = p.wheelbase * (1 - p.front_fraction)
        expected = weight * a / p.wheelbase + p.mass * ax * p.cg_height / (
            p.wheelbase
        )
        self.assertAlmostEqual(f['rear_load'] / expected, 1, delta=0.01)
        self.assertGreater(f['rear_load'], 0.15 * weight)

    def test_locked_braking_with_steer_does_not_spin(self):
        # Locked tires slide along the contact patch velocity and give no
        # yaw moment.  The old model yawed to -0.17 rad in 1.5 s.
        _, states = simulate(
            Vehicle(),
            [0, 0, 0, 25, 0, 0, 0],
            lambda t: (0, 1, 0.02, 4),
            1.5,
            0.002,
            CURVE,
        )
        self.assertLess(np.abs(states[:, 2]).max(), 0.01)
        self.assertLess(states[-1, 3], 11)

    def test_axle_forces_come_from_golf_r_tire(self):
        p = Vehicle()
        state = [0, 0, 0, 20, 0.1, 0.05, 0]
        f = forces(p, state, 0, 0, 0.01, 3, CURVE)
        a = p.wheelbase * (1 - p.front_fraction)
        alpha = math.atan2(0.1 + a * 0.05, 20) - 0.01
        expected = 2 * force(TIRE, f['front_load'] / 2, 0, alpha)
        np.testing.assert_allclose(
            [f['front_x'], f['front_y']], expected, atol=1e-6
        )

    def test_gillespie_steady_state_yaw_velocity_gain(self):
        # Gillespie, Fundamentals of Vehicle Dynamics (1992), ch. 6:
        # r / delta = (V / L) / (1 + K V^2 / (57.3 L g)), K in deg/g.
        p, speed, steer = Vehicle(), 30.0, 0.002
        weight = p.mass * GRAVITY
        stiffness = 2 * TIRE.cornering_stiffness * math.pi / 180
        gradient = (
            weight * p.front_fraction / stiffness
            - weight * (1 - p.front_fraction) / stiffness
        )
        expected = (speed / p.wheelbase) / (
            1 + gradient * speed**2 / (57.3 * p.wheelbase * GRAVITY)
        )
        # Integrate the lateral states at a fixed forward speed.
        state = np.array([0, 0, 0, speed, 0, 0, 0], dtype='float64')
        for _ in range(3000):
            rate = derivative(p, state, 0, 0, steer, 7, CURVE)
            state[4:6] += 0.001 * rate[4:6]
        rate = derivative(p, state, 0, 0, steer, 7, CURVE)
        self.assertLess(np.abs(rate[4:6]).max(), 1e-6)
        self.assertAlmostEqual(state[5] / steer / expected, 1, delta=0.005)

    def test_drivetrain_reads_ac_gear_ratios(self):
        fixture = (
            '[HEADER]\nVERSION=3\n[GEARS]\nCOUNT=3\t; forward gears\n'
            'GEAR_R=-3.0\nGEAR_1=3.5\nGEAR_2=2.0 \nGEAR_3=1.2\n'
            'FINAL=4.1\t\t; final gear ratio\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'drivetrain.ini').write_text(fixture)
            self.assertEqual(
                drivetrain(tmp), dict(gears=(3.5, 2.0, 1.2), final_drive=4.1)
            )
            p = vehicle_from_data(tmp)
            self.assertEqual(p.gears, (3.5, 2.0, 1.2))
            self.assertEqual(p.mass, Vehicle().mass)
            (Path(tmp) / 'drivetrain.ini').write_text(
                fixture.replace('COUNT=3', 'COUNT=4')
            )
            with self.assertRaises(ValueError):
                drivetrain(tmp)

    def test_drive_and_brake_split_from_ac_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'drivetrain.ini').write_text(
                '[GEARS]\nCOUNT=1\nGEAR_1=3.0\nFINAL=4.0\n'
                '[AWD]\nFRONT_SHARE=98\t; percent\n')
            (Path(tmp) / 'brakes.ini').write_text(
                '[DATA]\nMAX_TORQUE=2000\nFRONT_SHARE=0.7\n')
            p = vehicle_from_data(tmp)
            self.assertAlmostEqual(p.front_drive_share, 0.98)
            self.assertAlmostEqual(p.front_brake_share, 0.7)
            self.assertAlmostEqual(p.max_brake_torque, 2000.0)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
