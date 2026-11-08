# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest
import numpy as np
from solvcon.apps.golf_r.tire import Tire, force, relax


class TireTest(unittest.TestCase):
    def test_zero_load_and_zero_slip(self):
        tire = Tire()
        np.testing.assert_array_equal(force(tire, 0, 1, 1), [0, 0])
        np.testing.assert_array_equal(force(tire, 4000, 0, 0), [0, 0])

    def test_small_slip_slopes_and_signs(self):
        tire = Tire()
        fx, fy = force(tire, 4000, 1e-5, 1e-5)
        self.assertAlmostEqual(
            fx / 1e-5, tire.longitudinal_stiffness, delta=0.1
        )
        self.assertAlmostEqual(-fy / 1e-5, tire.cornering_stiffness, delta=0.1)

    def test_combined_slip_is_bounded_and_antisymmetric(self):
        tire = Tire()
        f = force(tire, 4000, 0.5, 0.5)
        self.assertLessEqual(np.linalg.norm(f), 4000 + 1e-9)
        np.testing.assert_allclose(force(tire, 4000, -0.5, -0.5), -f)

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
