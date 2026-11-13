# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""The KN5 reader on a model written here (no game asset): a root, four
wheel nodes with a tire mesh each, and a body mesh under a moved node.

* The node tree, the transforms (row vectors, the parent after the
  child), and the meshes come back.
* ``body_surface`` gives the body in the vehicle axes (x forward, y left,
  z up) about the middle of the wheels, without the wheel meshes.
* A cut or foreign file is rejected.
"""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solvcon.apps.golf_r import kn5


def text(value):
    data = value.encode()
    return struct.pack('<i', len(data)) + data


def dummy(name, children, offset=(0.0, 0.0, 0.0)):
    matrix = np.eye(4, dtype='<f4')
    matrix[3, :3] = offset
    return (struct.pack('<i', 1) + text(name)
            + struct.pack('<i', len(children)) + b'\x01' + matrix.tobytes()
            + b''.join(children))


def mesh(name, positions, normals, indices):
    vertices = b''.join(
        struct.pack('<11f', *p, *n, 0.0, 0.0, 1.0, 0.0, 0.0)
        for p, n in zip(positions, normals))
    return (struct.pack('<i', 2) + text(name) + struct.pack('<i', 0)
            + b'\x01' + b'\x01\x01\x00'
            + struct.pack('<i', len(positions)) + vertices
            + struct.pack('<i', len(indices))
            + struct.pack(f'<{len(indices)}H', *indices)
            + struct.pack('<ii', 0, 0) + struct.pack('<2f', 0.0, 0.0)
            + struct.pack('<4f', 0.0, 0.0, 0.0, 1.0) + b'\x01')


TRIANGLE = ([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
            [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)], [0, 1, 2])
#: KN5 axes: x left, y up, z forward.  A 2.4 m wheelbase, a 1.5 m track.
WHEELS = {'WHEEL_LF': (0.75, 0.3, 1.5), 'WHEEL_RF': (-0.75, 0.3, 1.5),
          'WHEEL_LR': (0.75, 0.3, -0.9), 'WHEEL_RR': (-0.75, 0.3, -0.9)}


def model():
    material = (text('paint') + text('ksPerPixel') + b'\x00\x00'
                + struct.pack('<i', 0)
                + struct.pack('<i', 1) + text('ksDiffuse') + b'\x00' * 40
                + struct.pack('<i', 1) + text('txDiffuse')
                + struct.pack('<i', 0) + text('paint.dds'))
    texture = struct.pack('<i', 1) + text('paint.dds') \
        + struct.pack('<i', 3) + b'abc'
    nodes = [dummy(name, [mesh('tyre', *TRIANGLE)], offset)
             for name, offset in WHEELS.items()]
    nodes.append(dummy('BODY', [mesh('shell', *TRIANGLE)],
                       (0.0, 0.5, 0.3)))
    return (b'sc6969' + struct.pack('<ii', 6, 0)
            + struct.pack('<i', 1) + texture
            + struct.pack('<i', 1) + material + dummy('root', nodes))


class Kn5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / 'car.kn5'
        cls.path.write_bytes(model())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_tree_and_the_meshes(self):
        nodes = kn5.read(self.path)
        self.assertEqual(11, len(nodes))
        self.assertEqual('root', nodes[0]['name'])
        self.assertEqual(-1, nodes[0]['parent'])
        body = next(n for n in nodes if 'BODY' == n['name'])
        np.testing.assert_allclose(body['world'][3, :3], [0.0, 0.5, 0.3])
        shell = next(n for n in nodes if 'shell' == n['name'])
        self.assertEqual('BODY', nodes[shell['parent']]['name'])
        np.testing.assert_allclose(shell['positions'], TRIANGLE[0])
        self.assertEqual([0, 1, 2], shell['indices'].tolist())
        self.assertEqual(5, sum('indices' in n for n in nodes))

    def test_the_body_in_the_vehicle_axes(self):
        body = kn5.body_surface(self.path)
        # Only the shell: the wheel meshes are left out.
        self.assertEqual((3, 3), body['positions'].shape)
        self.assertEqual([0, 1, 2], body['indices'].tolist())
        # The wheels: x forward, y left, z up, about their middle.
        np.testing.assert_allclose(body['wheels']['WHEEL_LF'],
                                   [1.2, 0.75, 0.3], atol=1e-6)
        np.testing.assert_allclose(body['wheels']['WHEEL_RR'],
                                   [-1.2, -0.75, 0.3], atol=1e-6)
        # The shell node sits 0.3 m ahead of the KN5 origin (the middle of
        # the wheels is 0.3 m ahead too) and 0.5 m up: its first vertex is
        # over the origin; the second is 1 m forward, the third 1 m up.
        np.testing.assert_allclose(
            body['positions'], [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5],
                                [0.0, 0.0, 1.5]], atol=1e-6)
        # Unit normals; a zero one becomes the up axis.
        np.testing.assert_allclose(
            body['normals'], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]], atol=1e-6)

    def test_with_the_wheels(self):
        body = kn5.body_surface(self.path, without=())
        self.assertEqual((15, 3), body['positions'].shape)
        self.assertEqual(14, int(body['indices'].max()))

    def test_the_body_file_of_the_car_data(self):
        # <car>/data/lods.ini names the models; they are in <car>.
        car = Path(self.tmp.name) / 'car'
        data = car / 'data'
        data.mkdir(parents=True)
        self.assertIsNone(kn5.find_body(data))
        (data / 'lods.ini').write_text(
            '[LOD_0]\nFILE=full.kn5\n[LOD_1]\nFILE=second.kn5 ; far\n')
        self.assertIsNone(kn5.find_body(data))
        (car / 'full.kn5').write_bytes(model())
        self.assertEqual(car / 'full.kn5', kn5.find_body(data))
        (car / 'second.kn5').write_bytes(model())
        self.assertEqual(car / 'second.kn5', kn5.find_body(data))
        # Unpacked data two folders below the car.
        deep = car / 'unpacked' / 'physics'
        deep.mkdir(parents=True)
        (deep / 'lods.ini').write_text('[LOD_1]\nFILE=second.kn5\n')
        self.assertEqual(car / 'second.kn5', kn5.find_body(deep))

    def test_a_cut_or_foreign_file(self):
        data = model()
        cut = Path(self.tmp.name) / 'cut.kn5'
        cut.write_bytes(data[:-10])
        with self.assertRaisesRegex(ValueError, 'KN5'):
            kn5.read(cut)
        other = Path(self.tmp.name) / 'other.kn5'
        other.write_bytes(b'glTF' + data)
        with self.assertRaisesRegex(ValueError, 'not a KN5 file'):
            kn5.read(other)
        more = Path(self.tmp.name) / 'more.kn5'
        more.write_bytes(data + b'\x00\x00')
        with self.assertRaisesRegex(ValueError, 'after the node tree'):
            kn5.read(more)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
