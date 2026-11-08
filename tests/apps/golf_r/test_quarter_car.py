# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest
import numpy as np
from solvcon.apps.golf_r.quarter_car import (
    Parameters,
    derivative,
    energy,
    simulate,
)


class QuarterCarTest(unittest.TestCase):
    def setUp(self):
        # Assumed 45 kg unsprung per front wheel; total static front mass 992.4
        # kg.
        self.p = Parameters(451.2, 45.0, 55000.0, 6200.0, 298926.0)

    def test_static_equilibrium_for_constant_road(self):
        np.testing.assert_allclose(
            derivative(self.p, [0.01, 0.01, 0, 0], 0.01), 0, atol=1e-12
        )

    def test_undamped_frequencies_match_polynomial(self):
        p = self.p
        coefficients = [
            p.sprung_mass * p.unsprung_mass,
            -(
                p.sprung_mass * (p.wheel_rate + p.tire_rate)
                + p.unsprung_mass * p.wheel_rate
            ),
            p.wheel_rate * p.tire_rate,
        ]
        expected = np.sqrt(np.sort(np.roots(coefficients))) / (2 * np.pi)
        np.testing.assert_allclose(
            p.natural_frequencies(), expected, rtol=1e-12
        )

    def test_free_response_loses_energy_and_converges(self):
        initial = [0.01, 0, 0, 0]
        _, coarse = simulate(self.p, lambda t: 0, 1, 0.002, initial)
        _, fine = simulate(self.p, lambda t: 0, 1, 0.001, initial)
        _, reference = simulate(self.p, lambda t: 0, 1, 0.0005, initial)
        self.assertLess(
            np.linalg.norm(fine[-1] - reference[-1]),
            np.linalg.norm(coarse[-1] - reference[-1]),
        )
        e = energy(self.p, fine)
        self.assertTrue(np.all(np.diff(e) <= 1e-9))
        self.assertLess(e[-1], e[0])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Parameters(0, 1, 1, 0, 1)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
