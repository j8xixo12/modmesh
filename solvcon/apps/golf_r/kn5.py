# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Read the geometry of an Assetto Corsa KN5 model for a body outline.

The layout (as the community documents it; Kunos publishes none): the
magic ``sc6969``, a version, the textures (skipped), the materials
(skipped), then the node tree.  A node is a dummy (class 1, a 4 x 4
transform of row vectors, the translation in the last row), a mesh (class
2: vertices of position, normal, texture coordinate, and tangent, 44
bytes; 16-bit triangle indices), or a skinned mesh (class 3: bones, 76
byte vertices).  The vertices of a mesh are in the frame of its parent.

The KN5 axes are x to the left (seen from the driver: ``WHEEL_LF`` has a
positive x), y up, z forward; :func:`body_surface` gives the meshes in the
vehicle axes of :mod:`solvcon.multibody.suspension.car` (x forward, y left,
z up), in metres, with the origin under the middle of the four wheel
nodes, at the KN5 ground height.
"""

import configparser
import struct
from pathlib import Path

import numpy as np

__all__ = [
    'read',
    'body_surface',
    'find_body',
    'WHEEL_NODES',
]

#: The wheel nodes of an AC car, in the order of ``car.CORNERS``.
WHEEL_NODES = ('WHEEL_LF', 'WHEEL_RF', 'WHEEL_LR', 'WHEEL_RR')
#: KN5 axes (x left, y up, z forward) to vehicle axes (x forward, y left,
#: z up): a cyclic permutation, so the triangle winding stays.
AXES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class _Reader(object):
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise ValueError("KN5: the file ends inside a record")
        out = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return out

    def skip(self, size):
        if size < 0 or self.pos + size > len(self.data):
            raise ValueError("KN5: the file ends inside a record")
        self.pos += size

    def int(self):
        return self.take('<i')[0]

    def text(self):
        size = self.int()
        if size < 0 or self.pos + size > len(self.data):
            raise ValueError("KN5: a string runs past the end of the file")
        out = self.data[self.pos:self.pos + size].decode('utf-8', 'replace')
        self.pos += size
        return out

    def array(self, dtype, count):
        dtype = np.dtype(dtype)
        size = dtype.itemsize * count
        if count < 0 or self.pos + size > len(self.data):
            raise ValueError("KN5: an array runs past the end of the file")
        out = np.frombuffer(self.data, dtype, count, self.pos)
        self.pos += size
        return out


_VERTEX = np.dtype([('position', '<f4', 3), ('normal', '<f4', 3),
                    ('rest', '<f4', 5)])
_SKINNED = np.dtype([('position', '<f4', 3), ('normal', '<f4', 3),
                     ('rest', '<f4', 13)])


def _node(reader, parent, world, nodes):
    kind = reader.int()
    name = reader.text()
    children = reader.int()
    reader.skip(1)                               # active
    node = {'name': name, 'class': kind, 'parent': parent, 'world': world}
    if 1 == kind:
        local = np.array(reader.take('<16f')).reshape(4, 4)
        # Row vectors: the local transform first, then the parent.
        node['world'] = world = local @ world
    elif kind in (2, 3):
        reader.skip(3)                # cast shadows, visible, transparent
        if 3 == kind:
            for _ in range(reader.int()):
                reader.text()
                reader.skip(64)
        vertices = reader.array(_VERTEX if 2 == kind else _SKINNED,
                                reader.int())
        node['positions'] = vertices['position']
        node['normals'] = vertices['normal']
        node['indices'] = reader.array('<u2', reader.int())
        node['material'] = reader.int()
        reader.skip(4)                           # layer
        # LOD range and the bounding sphere and the renderable flag of a
        # mesh; a skinned mesh has 8 bytes here.
        reader.skip(25 if 2 == kind else 8)
    else:
        raise ValueError(f"KN5: node '{name}' has the unknown class {kind}")
    index = len(nodes)
    nodes.append(node)
    for _ in range(children):
        _node(reader, index, world, nodes)


def read(path):
    """Read the nodes of a KN5 file.

    :return: list of nodes in tree order (parents first), each a dict with
        ``name``, ``class`` (1 dummy, 2 mesh, 3 skinned mesh), ``parent``
        (index, -1 for the root), ``world`` (4 x 4, row vectors, KN5
        axes), and for a mesh ``positions``, ``normals`` ((n, 3), in the
        frame of the parent), ``indices`` (triangles), and ``material``
        (index)
    :raises ValueError: when the file is not a KN5 file or is cut short
    """
    with open(path, 'rb') as fobj:
        reader = _Reader(fobj.read())
    if b'sc6969' != reader.data[:6]:
        raise ValueError(f"{path}: not a KN5 file")
    reader.pos = 6
    version = reader.int()
    if version > 5:
        reader.skip(4)
    for _ in range(reader.int()):                # textures
        reader.skip(4)
        reader.text()
        reader.skip(reader.int())
    for _ in range(reader.int()):                # materials
        reader.text()
        reader.text()
        reader.skip(2)
        if version > 4:
            reader.skip(4)
        for _ in range(reader.int()):
            reader.text()
            reader.skip(40)
        for _ in range(reader.int()):
            reader.text()
            reader.skip(4)
            reader.text()
    nodes = []
    _node(reader, -1, np.eye(4), nodes)
    if reader.pos != len(reader.data):
        raise ValueError(f"{path}: {len(reader.data) - reader.pos} bytes "
                         "after the node tree")
    return nodes


def find_body(data_dir):
    """The KN5 file of the body of an AC car for an outline, or None.

    ``lods.ini`` of the car data lists the models by level of detail; the
    second one (``LOD_1``, some ten thousand triangles) is enough for an
    outline, the first one is the full model.  The file is beside the data
    folder (``<car>/data``) or up to two folders above it (unpacked data).
    """
    data_dir = Path(data_dir)
    lods = configparser.ConfigParser(inline_comment_prefixes=(';',),
                                     strict=False)
    lods.read(data_dir / 'lods.ini')
    for section in ('LOD_1', 'LOD_0'):
        name = lods.get(section, 'FILE', fallback='').strip()
        if not name:
            continue
        for folder in (data_dir, *list(data_dir.resolve().parents)[:2]):
            if (folder / name).is_file():
                return folder / name
    return None


def _under(nodes, index, names):
    while index >= 0:
        if nodes[index]['name'] in names:
            return True
        index = nodes[index]['parent']
    return False


def body_surface(path, without=None):
    """The meshes of a KN5 car as one surface in the vehicle axes.

    :param without: names of nodes to leave out with their children; by
        default the wheel and suspension nodes (``WHEEL_*``, ``SUSP_*``),
        which the suspension view draws itself
    :return: dict with ``positions`` and ``normals`` ((n, 3) float32, m),
        ``indices`` ((m,) uint32, triangles), and ``wheels`` (``{node:
        position}`` of :data:`WHEEL_NODES`, m); the origin is under the
        middle of the wheel nodes, at the KN5 ground height, when the file
        has the four, else the KN5 origin
    """
    nodes = read(path)
    if without is None:
        without = {f'{part}_{corner}' for part in ('WHEEL', 'SUSP')
                   for corner in ('LF', 'RF', 'LR', 'RR')}
    positions, normals, indices = [], [], []
    base = 0
    for index, node in enumerate(nodes):
        if 'positions' not in node or _under(nodes, index, without):
            continue
        world = node['world']
        positions.append((node['positions'] @ world[:3, :3] + world[3, :3])
                         @ AXES.T)
        normals.append(node['normals'] @ world[:3, :3] @ AXES.T)
        indices.append(node['indices'].astype('uint32') + base)
        base += len(node['positions'])
    if not positions:
        raise ValueError(f"{path}: no mesh")
    positions = np.concatenate(positions)
    normals = np.concatenate(normals)
    # Unit normals; a vertex without one (zero length) gets the up axis.
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(length > 1e-12, normals / np.where(
        length > 1e-12, length, 1.0), [0.0, 0.0, 1.0])
    wheels = {node['name']: AXES @ node['world'][3, :3] for node in nodes
              if node['name'] in WHEEL_NODES}
    if len(wheels) == len(WHEEL_NODES):
        centers = np.array([wheels[name] for name in WHEEL_NODES])
        origin = centers.mean(axis=0)
        # The KN5 ground stays the ground.
        origin[2] = 0.0
        positions = positions - origin
        wheels = {name: p - origin for name, p in wheels.items()}
    return {'positions': positions.astype('float32'),
            'normals': normals.astype('float32'),
            'indices': np.concatenate(indices), 'wheels': wheels}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
