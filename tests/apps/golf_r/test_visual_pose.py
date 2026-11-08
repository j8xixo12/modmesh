# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import types
import unittest
import numpy as np
from solvcon.apps.golf_r.visual_pose import transforms, wheel_groups


class VisualPoseTest(unittest.TestCase):
    def test_wheel_rotation_preserves_pivot_and_body_is_rigid(self):
        items = [
            types.SimpleNamespace(key='body', node=0),
            types.SimpleNamespace(key='wheel', node=1),
        ]
        pivot = np.array([2, 1, 0.3], dtype='float64')
        pose = transforms(
            items, {1: 'WHEEL_LF'}, {'WHEEL_LF': pivot}, 0, 0, np.pi / 2
        )
        shift, q = pose['wheel']
        w, x, y, z = q
        r = np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype='float64',
        )
        np.testing.assert_allclose(r @ pivot + shift, pivot, atol=1e-12)
        np.testing.assert_allclose(pose['body'][0], [0, 0, 0])

    def test_missing_wheels_are_rejected(self):
        model = types.SimpleNamespace(nodes=[])
        with self.assertRaises(ValueError):
            wheel_groups(model, {})


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
