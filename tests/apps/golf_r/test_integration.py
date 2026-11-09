# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""The experimental application needs no proprietary fixtures in tests."""

import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
from solvcon.apps.golf_r.bicycle import Vehicle, simulate, torque_curve
from solvcon.apps.golf_r.multibody_pose import transforms
from solvcon.multibody.suspension.metrics import rot


class IntegrationTest(unittest.TestCase):
    def test_external_curve_drives_bicycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'power.lut'
            path.write_text('800|100\n8000|100\n')
            curve = torque_curve(tmp)
            _, states = simulate(
                Vehicle(),
                [0, 0, 0, 15, 0, 0, 0],
                lambda t: (0.25, 0, 0, 3),
                1,
                0.01,
                curve,
            )
            self.assertGreater(states[-1, 3], 15)
            self.assertAlmostEqual(states[-1, 1], 0)

    def test_wheel_pivot_tracks_solved_upright(self):
        corners = ('front_left', 'front_right', 'rear_left', 'rear_right')
        names = ('WHEEL_LF', 'WHEEL_RF', 'WHEEL_LR', 'WHEEL_RR')
        pivots = {
            name: np.array([1, 1, 0.3], dtype='float64') for name in names
        }
        q = np.array([1, 0, 0, 0], dtype='float64')
        corners_0 = {
            name: {'upright': (np.zeros(3, dtype='float64'), q)}
            for name in corners
        }
        corners_1 = {
            name: {'upright': (np.array([0, 0, 0.02], dtype='float64'), q)}
            for name in corners
        }
        initial = {
            'cg': np.array([0, 0, 0.5], dtype='float64'),
            'chassis': (np.array([0, 0, 0.5], dtype='float64'), q),
            'corners': corners_0,
        }
        pose = {
            'chassis': (np.array([0, 0, 0.51], dtype='float64'), q),
            'corners': corners_1,
        }
        items = [
            types.SimpleNamespace(key=name, node=i)
            for i, name in enumerate(names)
        ]
        actual = transforms(
            items, dict(enumerate(names)), pivots, pose, initial, 0.7
        )
        for name in names:
            shift, rotation = actual[name]
            np.testing.assert_allclose(
                rot(rotation) @ pivots[name] + shift,
                pivots[name] + [0, 0, 0.02],
                atol=1e-12,
            )


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
