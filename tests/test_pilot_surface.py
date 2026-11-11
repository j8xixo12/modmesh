# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Imported surfaces must own geometry and share named-object semantics."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from solvcon import core, pilot, testing


def arrays(offset=0):
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype='float32')
    positions += offset
    normals = np.tile(np.array([0, -1, 1], dtype='float32'), (3, 1))
    indices = np.array([[0, 1, 2]], dtype='uint32')
    return (core.SimpleArrayFloat32(array=positions),
            core.SimpleArrayFloat32(array=normals),
            core.SimpleArrayUint32(array=indices))


@testing.requires_pilot
class SurfaceObjectTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.widget = pilot.RDomainWidget()

    def test_background_rejects_invalid_colors_without_changing_state(self):
        self.widget.setBackgroundColor(0.1, 0.2, 0.3)
        before = self.widget.backgroundColor
        np.testing.assert_allclose(before, (0.1, 0.2, 0.3), atol=2e-5)
        for value in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.widget.setBackgroundColor(value, 0.2, 0.3)
            self.assertEqual(before, self.widget.backgroundColor)

    def test_surface_update_preserves_state_and_rejects_invalid_input(self):
        self.widget.addSurfaceObject('part', *arrays())
        self.widget.setObjectTransform('part', 2, 3, 4)
        before = self.widget.objectModel('part')
        self.widget.updateSurfaceObject('part', *arrays(10))
        self.assertEqual(before, self.widget.objectModel('part'))
        self.widget.fitCameraToScene()
        np.testing.assert_allclose(self.widget.cameraTarget,
                                   [12.5, 13.5, 14.5])
        pos, normal, indices = arrays()
        indices.ndarray[0, 0] = 99
        with self.assertRaises(ValueError):
            self.widget.updateSurfaceObject('part', pos, normal, indices)
        self.assertEqual(before, self.widget.objectModel('part'))

    def test_material_parameters_reject_nonfinite_and_out_of_range(self):
        self.widget.addSurfaceObject('part', *arrays())
        self.widget.setObjectMaterial('part', 0.2, 0.8)
        for value in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.widget.setObjectMaterial('part', value, 0.8)
            with self.assertRaises(ValueError):
                self.widget.setObjectMaterial('part', 0.2, value)
        self.assertEqual(['part'], self.widget.objectNames())

    def test_nonplanar_surface_and_replacement(self):
        self.widget.addSurfaceObject('part', *arrays())
        self.widget.fitCameraToScene()
        np.testing.assert_allclose(self.widget.cameraTarget, [0.5, 0.5, 0.5])
        self.widget.addSurfaceObject('part', *arrays(10))
        self.assertEqual(self.widget.objectNames(), ['part'])
        self.widget.fitCameraToScene()
        np.testing.assert_allclose(self.widget.cameraTarget,
                                   [10.5, 10.5, 10.5])

    def test_removal_rebuilds_bounds_and_preserves_other_objects(self):
        self.widget.addSurfaceObject('near', *arrays())
        self.widget.addSurfaceObject('far', *arrays(100))
        self.widget.removeObject('far')
        self.widget.fitCameraToScene()
        self.assertEqual(self.widget.objectNames(), ['near'])
        np.testing.assert_allclose(self.widget.cameraTarget, [0.5, 0.5, 0.5])
        self.widget.clearObjects()
        self.assertEqual(self.widget.objectNames(), [])
        self.widget.addSurfaceObject('small', *arrays(-1))
        self.widget.fitCameraToScene()
        np.testing.assert_allclose(self.widget.cameraTarget, [-0.5] * 3)

    def test_invalid_replacement_does_not_destroy_previous_surface(self):
        self.widget.addSurfaceObject('part', *arrays())
        positions, normals, indices = arrays()
        indices.ndarray[0, 2] = 3
        with self.assertRaisesRegex(ValueError, 'index out of range'):
            self.widget.addSurfaceObject('part', positions, normals, indices)
        self.assertEqual(self.widget.objectNames(), ['part'])
        self.widget.fitCameraToScene()
        np.testing.assert_allclose(self.widget.cameraTarget, [0.5] * 3)

    def test_nonfinite_and_zero_normals_rejected(self):
        for invalid in (float('nan'), float('inf')):
            pos, normals, ind = arrays()
            pos.ndarray[0, 0] = invalid
            with self.assertRaisesRegex(ValueError, 'finite'):
                self.widget.addSurfaceObject('part', pos, normals, ind)
        pos, normals, ind = arrays()
        normals.ndarray[0] = 0
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            self.widget.addSurfaceObject('part', pos, normals, ind)

    def test_shapes_and_empty_index_buffer(self):
        pos, normals, ind = arrays()
        bad = core.SimpleArrayFloat32(array=np.zeros((3, 2), dtype='float32'))
        with self.assertRaises(ValueError):
            self.widget.addSurfaceObject('part', bad, normals, ind)
        with self.assertRaises(ValueError):
            self.widget.addSurfaceObject('part', pos, bad, ind)
        self.widget.addSurfaceObject('part', pos, normals, ind)
        empty = core.SimpleArrayUint32(
            array=np.empty((0, 3), dtype='uint32'))
        self.widget.addSurfaceObject('part', pos, normals, empty)
        self.assertEqual(self.widget.objectNames(), [])

    def test_transform_remains_available_for_surface_objects(self):
        self.widget.addSurfaceObject('part', *arrays())
        self.widget.setObjectTransform('part', 2, 3, 4, 1, 1, 1)
        matrix = np.array(self.widget.objectModel('part'), dtype='float64')
        np.testing.assert_allclose(matrix.reshape(4, 4)[:3, 3], [2, 3, 4])

    def test_hidden_surface_disappears_from_capture_when_available(self):
        from PySide6.QtGui import QImage
        self.widget.addSurfaceObject('part', *arrays())
        self.widget.resize(320, 240)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'surface.png')
            if not self.widget.renderToImage(path, 320, 240, False):
                self.skipTest('QRhi offscreen capture is unavailable')
            shown = QImage(path)
            hidden = pilot.RDomainWidget()
            hidden.addSurfaceObject('part', *arrays())
            hidden.setObjectVisible('part', False)
            hidden.resize(320, 240)
            if not hidden.renderToImage(path, 320, 240, False):
                self.skipTest('QRhi offscreen capture is unavailable')
            self.assertNotEqual(shown, QImage(path))

    @staticmethod
    def _texture():
        from PySide6.QtCore import QBuffer, QIODevice
        from PySide6.QtGui import QColor, QImage
        image = QImage(2, 2, QImage.Format_RGBA8888)
        for y in range(2):
            image.setPixelColor(0, y, QColor(255, 0, 0))
            image.setPixelColor(1, y, QColor(0, 0, 255))
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        image.save(buffer, 'PNG')
        return bytes(buffer.data())

    def test_textured_surface_validates_uv_image_and_sampler(self):
        pos, normal, idx = arrays()
        uv = core.SimpleArrayFloat32(
            array=np.array([[0, 0], [1, 0], [0, 1]], dtype='float32'))
        self.widget.addTexturedSurfaceObject(
            'part', pos, normal, uv, idx, self._texture(), 'BLEND')
        self.assertEqual(self.widget.objectNames(), ['part'])
        for kwargs in ({'alpha_mode': 'unknown'}, {'alpha_cutoff': -1},
                       {'alpha_factor': 2}, {'wrap_s': 0}):
            with self.assertRaises(ValueError):
                self.widget.addTexturedSurfaceObject(
                    'part', pos, normal, uv, idx, self._texture(), **kwargs)
        with self.assertRaises(ValueError):
            self.widget.addTexturedSurfaceObject(
                'part', pos, normal, uv, idx, b'not an image')
        uv.ndarray[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.widget.addTexturedSurfaceObject(
                'part', pos, normal, uv, idx, self._texture())
        self.assertEqual(self.widget.objectNames(), ['part'])

    def test_texture_colors_reach_pixels_when_graphics_available(self):
        from PySide6.QtGui import QImage
        pos, normal, idx = arrays()
        uv = core.SimpleArrayFloat32(
            array=np.array([[0, 0], [1, 0], [0, 1]], dtype='float32'))
        self.widget.addTexturedSurfaceObject(
            'part', pos, normal, uv, idx, self._texture())
        self.widget.resize(320, 240)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'texture.png')
            if not self.widget.renderToImage(path, 320, 240, False):
                self.skipTest('QRhi offscreen capture is unavailable')
            image = QImage(path).convertToFormat(QImage.Format_RGBA8888)
            rgba = np.frombuffer(image.constBits(), dtype='uint8').reshape(
                image.height(), image.width(), 4).astype('int16')
            self.assertTrue(np.any(rgba[:, :, 0] > rgba[:, :, 2] + 40))
            self.assertTrue(np.any(rgba[:, :, 2] > rgba[:, :, 0] + 40))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
