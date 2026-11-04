# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The 3D navigation math of the suspension app, free of Qt: the vehicle
views (ISO 8855, x forward, y left, z up), the view transitions, the view
history, the orientation cube, the hardpoint pick, and the ground grid."""

import unittest

import numpy as np

from solvcon.apps.suspension import _navigation as nav


def _basis(state):
    forward = state.target - state.position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, state.up)
    right /= np.linalg.norm(right)
    return forward, right, np.cross(right, forward)


class VehicleViewTC(unittest.TestCase):

    TARGET = np.array([10.0, 600.0, 300.0])

    def test_views_look_along_the_vehicle_axes(self):
        # Each entry: the view direction (eye toward target) and screen up.
        want = {
            'front': ([-1, 0, 0], [0, 0, 1]),
            'rear': ([1, 0, 0], [0, 0, 1]),
            'left': ([0, -1, 0], [0, 0, 1]),
            'right': ([0, 1, 0], [0, 0, 1]),
            'top': ([0, 0, -1], [1, 0, 0]),
            'bottom': ([0, 0, 1], [1, 0, 0]),
        }
        for name, (forward, up) in want.items():
            state = nav.view_state(name, self.TARGET, 1500.0)
            np.testing.assert_allclose(self.TARGET, state.target)
            got_forward, _, got_up = _basis(state)
            np.testing.assert_allclose(forward, got_forward, atol=1.e-12,
                                       err_msg=name)
            np.testing.assert_allclose(up, got_up, atol=1.e-12, err_msg=name)
            self.assertAlmostEqual(
                1500.0, np.linalg.norm(state.position - state.target))
            self.assertEqual(name, nav.aligned_view(state))

    def test_iso_is_from_front_left_above(self):
        state = nav.view_state('iso', self.TARGET, 1000.0)
        eye = state.position - state.target
        self.assertTrue(eye[0] > 0 and eye[1] > 0 and eye[2] > 0)
        _, right, up = _basis(state)
        self.assertAlmostEqual(0.0, right[2], places=12)
        self.assertGreater(up[2], 0.0)
        self.assertIsNone(nav.aligned_view(state))

    def test_aligned_view_tolerance(self):
        state = nav.view_state('front', self.TARGET, 1000.0)
        tilt = nav.CameraState(state.position + [0.0, 0.0, 1.0],
                               state.target, state.up)
        self.assertEqual('front', nav.aligned_view(tilt))  # 0.06 deg off
        tilt = nav.CameraState(state.position + [0.0, 0.0, 50.0],
                               state.target, state.up)
        self.assertIsNone(nav.aligned_view(tilt))  # 2.9 deg off

    def test_unknown_view_raises(self):
        with self.assertRaisesRegex(ValueError, "no view 'side'"):
            nav.view_state('side', self.TARGET, 1.0)


class TransitionTC(unittest.TestCase):

    def test_ends_and_midpoint(self):
        a = nav.view_state('front', np.zeros(3), 1000.0)
        b = nav.view_state('left', np.array([0.0, 0.0, 100.0]), 500.0)
        for t, state in ((0.0, a), (1.0, b)):
            got = nav.interpolate(a, b, t)
            np.testing.assert_allclose(state.position, got.position,
                                       atol=1.e-9)
            np.testing.assert_allclose(state.up, got.up, atol=1.e-12)
        mid = nav.interpolate(a, b, 0.5)
        eye = mid.position - mid.target
        # Halfway around the turn from +x to +y, at the mean distance.
        np.testing.assert_allclose([np.sqrt(0.5), np.sqrt(0.5), 0.0],
                                   eye / np.linalg.norm(eye), atol=1.e-12)
        self.assertAlmostEqual(750.0, np.linalg.norm(eye))
        np.testing.assert_allclose([0.0, 0.0, 50.0], mid.target)

    def test_up_stays_orthogonal_to_the_view(self):
        a = nav.view_state('iso', np.zeros(3), 1000.0)
        b = nav.view_state('top', np.zeros(3), 1000.0)
        for t in np.linspace(0.0, 1.0, 11):
            state = nav.interpolate(a, b, t)
            forward, _, up = _basis(state)
            self.assertAlmostEqual(0.0, float(forward @ state.up), places=9)
            self.assertAlmostEqual(1.0, float(np.linalg.norm(state.up)))

    def test_opposite_views_turn_about_the_up_axis(self):
        # Front to rear is a half turn: it must not pass through the target.
        a = nav.view_state('front', np.zeros(3), 1000.0)
        b = nav.view_state('rear', np.zeros(3), 1000.0)
        for t in np.linspace(0.0, 1.0, 9):
            state = nav.interpolate(a, b, t)
            self.assertAlmostEqual(1000.0, np.linalg.norm(state.position))
            self.assertAlmostEqual(0.0, state.position[2], places=9)

    def test_easing(self):
        self.assertEqual(0.0, nav.ease(0.0))
        self.assertEqual(1.0, nav.ease(1.0))
        self.assertAlmostEqual(0.5, nav.ease(0.5))
        self.assertEqual(1.0, nav.ease(3.0))


class HistoryTC(unittest.TestCase):

    def test_back_and_forward(self):
        hist = nav.ViewHistory()
        a = nav.view_state('front', np.zeros(3), 1.0)
        b = nav.view_state('left', np.zeros(3), 1.0)
        c = nav.view_state('top', np.zeros(3), 1.0)
        self.assertIsNone(hist.back(a))
        hist.push(a)
        hist.push(a)  # the same view twice is one entry
        hist.push(b)
        self.assertTrue(nav.same_state(b, hist.back(c)))
        self.assertTrue(nav.same_state(a, hist.back(b)))
        self.assertIsNone(hist.back(a))
        self.assertTrue(nav.same_state(b, hist.forward(a)))
        self.assertTrue(nav.same_state(c, hist.forward(b)))
        self.assertIsNone(hist.forward(c))
        # A new view drops the forward branch.
        hist.back(c)
        hist.push(a)
        self.assertIsNone(hist.forward(a))

    def test_named_views(self):
        hist = nav.ViewHistory()
        a = nav.view_state('front', np.zeros(3), 1.0)
        hist.save('pickups', a)
        self.assertEqual(['pickups'], hist.names())
        self.assertTrue(nav.same_state(a, hist.named('pickups')))
        with self.assertRaisesRegex(KeyError, "no saved view 'x'"):
            hist.named('x')


class OrientationCubeTC(unittest.TestCase):

    CENTER = np.array([60.0, 60.0])
    SIZE = 40.0

    def _hit(self, view, point):
        state = nav.view_state(view, np.zeros(3), 10.0)
        return nav.cube_hit(state, self.CENTER, self.SIZE, point)

    def test_front_view_shows_the_front_face(self):
        state = nav.view_state('front', np.zeros(3), 10.0)
        faces = nav.cube_faces(state, self.CENTER, self.SIZE)
        self.assertEqual('front', faces[-1][0])  # drawn last, nearest
        self.assertEqual(['front'], [name for name, _, facing in faces
                                     if facing > 0.99])

    def test_face_edge_and_corner_clicks(self):
        # In the front view, facing the car, screen right is +y (the left
        # side of the car) and screen up is +z (the pixel y axis points down).
        hit = self._hit('front', self.CENTER)
        np.testing.assert_allclose([1.0, 0.0, 0.0], hit, atol=1.e-12)
        hit = self._hit('front', self.CENTER + [0.0, -0.9 * self.SIZE])
        np.testing.assert_allclose(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0),
                                   hit, atol=1.e-12)
        hit = self._hit('front', self.CENTER + [0.9 * self.SIZE,
                                                0.9 * self.SIZE])
        np.testing.assert_allclose(np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0),
                                   hit, atol=1.e-12)
        self.assertIsNone(self._hit('front', self.CENTER + [3 * self.SIZE,
                                                            0.0]))

    def test_iso_view_hits_the_visible_faces(self):
        state = nav.view_state('iso', np.zeros(3), 10.0)
        for name, polygon, facing in nav.cube_faces(state, self.CENTER,
                                                    self.SIZE):
            if facing <= 0.0:
                continue
            hit = nav.cube_hit(state, self.CENTER, self.SIZE,
                               polygon.mean(axis=0))
            np.testing.assert_allclose(nav.FACES[name], hit, atol=1.e-9,
                                       err_msg=name)

    def test_state_for_a_cube_direction(self):
        state = nav.state_for_direction([1.0, 1.0, 1.0], np.zeros(3), 10.0)
        _, right, up = _basis(state)
        self.assertAlmostEqual(0.0, right[2], places=12)
        state = nav.state_for_direction([0.0, 0.0, 1.0], np.zeros(3), 10.0)
        self.assertEqual('top', nav.aligned_view(state))


class PickTC(unittest.TestCase):

    def test_nearest_point_within_the_radius(self):
        pixels = {'a': (10.0, 10.0, 0.5), 'b': (14.0, 10.0, 0.2),
                  'c': (200.0, 200.0, 0.1)}
        self.assertEqual('b', nav.nearest(pixels, (13.0, 10.0), 8.0))
        self.assertIsNone(nav.nearest(pixels, (100.0, 100.0), 8.0))
        # A tie in screen distance goes to the point nearer the eye.
        pixels = {'far': (10.0, 10.0, 0.9), 'near': (10.0, 10.0, 0.1)}
        self.assertEqual('near', nav.nearest(pixels, (10.0, 10.0), 8.0))
        # A point behind the eye (empty projection) is never picked.
        self.assertIsNone(nav.nearest({'x': ()}, (0.0, 0.0), 8.0))


class GridTC(unittest.TestCase):

    def test_spacing_is_a_round_number(self):
        self.assertEqual(50.0, nav.grid_spacing(700.0))
        self.assertEqual(100.0, nav.grid_spacing(1500.0))
        self.assertEqual(10.0, nav.grid_spacing(120.0))

    def test_grid_lines_cover_the_box(self):
        ends, spacing = nav.grid_segments([-120.0, 230.0], [180.0, 760.0],
                                          5.0, 50.0)
        self.assertEqual(50.0, spacing)
        ends = np.array(ends).reshape(-1, 2, 3)
        np.testing.assert_allclose(5.0, ends[:, :, 2])
        xs = sorted({e[0][0] for e in ends if e[0][0] == e[1][0]})
        ys = sorted({e[0][1] for e in ends if e[0][1] == e[1][1]})
        self.assertEqual(list(np.arange(-150.0, 201.0, 50.0)), xs)
        self.assertEqual(list(np.arange(200.0, 801.0, 50.0)), ys)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
