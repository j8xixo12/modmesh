# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""A complete Model Viewer window retains its model after a failed load."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from solvcon import pilot, testing
from solvcon.apps.modelview import ModelViewerApp


def write_model(path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype='<f4')
    binary = vertices.tobytes()
    doc = dict(asset=dict(version='2.0'), scene=0, scenes=[dict(nodes=[0])],
               nodes=[dict(name='Root', children=[1]),
                      dict(name='Wheel', mesh=0)],
               meshes=[dict(primitives=[dict(attributes=dict(POSITION=0))])],
               buffers=[dict(byteLength=len(binary))],
               bufferViews=[dict(buffer=0, byteLength=len(binary))],
               accessors=[dict(bufferView=0, type='VEC3', componentType=5126,
                               count=3)])
    js = json.dumps(doc).encode()
    js += b' ' * (-len(js) % 4)
    path.write_bytes(struct.pack('<III', 0x46546C67, 2,
                                 28 + len(js) + len(binary))
                     + struct.pack('<II', len(js), 0x4E4F534A) + js
                     + struct.pack('<II', len(binary), 0x004E4942) + binary)


@testing.requires_live_window
class ModelViewerWindowTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.app = ModelViewerApp(mgr=self.mgr)
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'sample.glb'
        write_model(self.path)

    def tearDown(self):
        if self.app.subwin is not None:
            self.app.subwin.close()
        self.folder.cleanup()

    def test_load_toggle_reload_and_close(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QPushButton
        self.app.load_path(self.path)
        self.assertEqual(self.app.widget.objectNames(), ['node:1/primitive:0'])
        item = self.app.node_items[1]
        item.setCheckState(0, Qt.Unchecked)
        item.setCheckState(0, Qt.Checked)
        self.app.tree.setCurrentItem(item)
        self.assertIn('3 vertices', self.app.selection.text())
        buttons = self.app.subwin.findChildren(QPushButton)
        for button in buttons:
            if button.text() != 'Open GLB':
                button.click()
        self.app.load_path(self.path)
        self.assertEqual(len(self.app.widget.objectNames()), 1)
        self.app.subwin.close()
        self.assertIsNone(self.app.widget)

    def test_failed_load_keeps_the_previous_scene(self):
        self.app.load_path(self.path)
        old_model = self.app.model
        bad = self.path.with_name('bad.glb')
        bad.write_bytes(b'not a GLB')
        with self.assertRaises(ValueError):
            self.app.load_path(bad)
        self.assertIs(self.app.model, old_model)
        self.assertEqual(len(self.app.widget.objectNames()), 1)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
