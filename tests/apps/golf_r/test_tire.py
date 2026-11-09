# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import unittest
import numpy as np
from solvcon.apps.golf_r.tire import (
    Tire,
    force,
    force_for_demand,
    friction_limit,
    relax,
)


class TireTest(unittest.TestCase):
    def test_zero_load_and_zero_slip(self):
        tire = Tire()
        np.testing.assert_array_equal(force(tire, 0, 1, 1), [0, 0])
        np.testing.assert_array_equal(force(tire, 4000, 0, 0), [0, 0])

    def test_small_slip_slopes_and_signs(self):
        # Pure slip, one input at a time: the theoretical slip divides by
        # 1 + kappa, so a combined 1e-5 input changes the slope by 1e-5.
        tire = Tire()
        fx, fy = force(tire, 4000, 1e-7, 0)
        self.assertAlmostEqual(fx / 1e-7, tire.longitudinal_stiffness, 1)
        self.assertEqual(fy, 0)
        fx, fy = force(tire, 4000, 0, 1e-7)
        self.assertAlmostEqual(-fy / 1e-7, tire.cornering_stiffness, 1)
        self.assertEqual(fx, 0)

    def test_combined_slip_is_bounded_and_antisymmetric(self):
        # The theoretical slip is not odd in kappa (kappa = -1 is a locked
        # wheel, kappa = 1 is not), so only the slip angle is mirrored.
        tire = Tire()
        for kappa in (-1, -0.5, 0, 0.5, 2):
            f = force(tire, 4000, kappa, 0.5)
            self.assertLessEqual(
                np.linalg.norm(f), friction_limit(tire, 4000) + 1e-9
            )
            np.testing.assert_allclose(
                force(tire, 4000, kappa, -0.5), f * [1, -1]
            )

    def test_locked_wheel_force_follows_slip_velocity(self):
        # A locked wheel slides; the force points along the sliding
        # velocity (Pacejka 2012, ch. 3), so |Fy| / |F| = sin(alpha).
        # The friction-circle clip gave 0.571 at alpha = 0.05 rad.
        tire = Tire()
        for alpha in (0.02, 0.05, 0.1, 0.3):
            fx, fy = force(tire, 4000, -1, alpha)
            magnitude = math.hypot(fx, fy)
            self.assertAlmostEqual(
                magnitude, friction_limit(tire, 4000), places=6
            )
            self.assertAlmostEqual(-fy / magnitude, math.sin(alpha), 12)
            self.assertLess(fx, 0)

    def test_force_for_demand_inverts_force(self):
        tire = Tire()
        for demand in (-3000, -100, 100, 3000):
            fx, fy, kappa = force_for_demand(tire, 4000, demand, 0.03)
            self.assertAlmostEqual(fx, demand, delta=1e-3)
            np.testing.assert_allclose(
                force(tire, 4000, kappa, 0.03), [fx, fy]
            )
        fx, fy, kappa = force_for_demand(tire, 4000, -1e5, 0.03)
        self.assertEqual(kappa, -1)
        np.testing.assert_allclose(force(tire, 4000, -1, 0.03), [fx, fy])

    def test_relaxation_length_is_passenger_car_value(self):
        # Pacejka 2012, ch. 7: about 0.3 m to 0.6 m for passenger tires.
        self.assertTrue(0.3 <= Tire().relaxation_length <= 0.6)

    def test_relaxation_has_exact_timestep_invariance(self):
        tire = Tire()
        whole = relax(tire, 0, 0.1, 10, 0.01)
        half = relax(tire, 0, 0.1, 10, 0.005)
        half = relax(tire, half, 0.1, 10, 0.005)
        self.assertAlmostEqual(whole, half, places=14)
        self.assertAlmostEqual(relax(tire, 0, 1, 0, 1), 0)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
