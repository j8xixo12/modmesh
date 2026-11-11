# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Read a bounded static glTF 2.0 subset from self-contained GLB files."""

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np


@dataclass
class Primitive:
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    material: int | None
    texcoords: np.ndarray | None = None


@dataclass
class Texture:
    data: bytes
    wrap_s: int
    wrap_t: int


@dataclass
class Node:
    name: str
    children: tuple
    matrix: np.ndarray
    mesh: int | None


@dataclass
class Model:
    nodes: list
    meshes: list
    materials: list
    roots: tuple
    notices: list
    textures: list


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label}: expected integer >= {minimum}')
    return value


def reference(sequence, value, label):
    index = integer(value, label)
    if index >= len(sequence):
        raise ValueError(f'{label}: index out of range')
    return sequence[index]


def _chunks(data):
    if len(data) < 20:
        raise ValueError('Truncated GLB header')
    magic, version, size = struct.unpack_from('<III', data)
    if magic != 0x46546C67 or version != 2 or size != len(data):
        raise ValueError('Expected GLB version 2 with matching file length')
    chunks = []
    offset = 12
    while offset < size:
        if size - offset < 8:
            raise ValueError('Truncated GLB chunk header')
        length, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        if length % 4 or length > size - offset:
            raise ValueError('Invalid GLB chunk length')
        chunks.append((kind, memoryview(data)[offset:offset + length]))
        offset += length
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise ValueError('GLB JSON must be the first chunk')
    if sum(k == 0x4E4F534A for k, _ in chunks) != 1:
        raise ValueError('Duplicate GLB JSON chunk')
    bins = [b for k, b in chunks if k == 0x004E4942]
    if len(bins) > 1 or (bins and chunks[1][0] != 0x004E4942):
        raise ValueError('Invalid GLB binary chunk order')
    try:
        doc = json.loads(bytes(chunks[0][1]))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid GLB JSON: {exc}') from exc
    if not isinstance(doc, dict):
        raise ValueError('GLB JSON must be an object')
    return doc, bins[0] if bins else memoryview(b'')


class Accessors:
    TYPES = {5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}

    def __init__(self, doc, binary):
        self.doc = doc
        buffers = doc.get('buffers', [])
        if len(buffers) != 1 or 'uri' in buffers[0]:
            raise ValueError('Only one embedded GLB geometry buffer supported')
        size = integer(buffers[0].get('byteLength'), 'buffer.byteLength', 1)
        if not size <= len(binary) <= size + 3:
            raise ValueError('GLB buffer length does not match binary chunk')
        self.binary = binary[:size]

    def read(self, index, kind, components):
        acc = reference(self.doc.get('accessors', []), index, 'accessor')
        if 'sparse' in acc or acc.get('normalized', False):
            raise ValueError('Sparse or normalized accessors unsupported')
        component = acc.get('componentType')
        if component not in components or acc.get('type') != kind:
            raise ValueError(f'Expected {kind} accessor of type {components}')
        dtype = np.dtype(self.TYPES[component])
        width = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[kind]
        item_size = dtype.itemsize * width
        view = reference(self.doc.get('bufferViews', []),
                         acc.get('bufferView'), 'bufferView')
        if view.get('buffer') != 0 or view.get('extensions'):
            raise ValueError('Unsupported bufferView buffer or extension')
        start = integer(view.get('byteOffset', 0), 'bufferView.byteOffset')
        size = integer(view.get('byteLength'), 'bufferView.byteLength', 1)
        offset = integer(acc.get('byteOffset', 0), 'accessor.byteOffset')
        count = integer(acc.get('count'), 'accessor.count', 1)
        stride = integer(view.get('byteStride', item_size), 'byteStride', 1)
        if (stride < item_size or stride % dtype.itemsize
                or start % dtype.itemsize or offset % dtype.itemsize):
            raise ValueError('Invalid accessor alignment or stride')
        if 'byteStride' in view and (stride > 252 or stride % 4):
            raise ValueError('Invalid interleaved byteStride')
        span = (count - 1) * stride + item_size
        if (start + size > len(self.binary) or offset + span > size):
            raise ValueError('Accessor exceeds bufferView or buffer bounds')
        out = np.ndarray((count, width), dtype=dtype, buffer=self.binary,
                         offset=start + offset,
                         strides=(stride, dtype.itemsize)).copy()
        if component == 5126 and not np.isfinite(out).all():
            raise ValueError('Nonfinite vertex attribute')
        return out


def _matrix(node):
    if 'matrix' in node:
        if any(k in node for k in ('translation', 'rotation', 'scale')):
            raise ValueError('A node cannot contain both matrix and TRS')
        values = np.asarray(node['matrix'], dtype='float64')
        if values.shape != (16,):
            raise ValueError('Node matrix must have 16 components')
        matrix = values.reshape(4, 4).T.copy()
    else:
        t = np.asarray(node.get('translation', [0, 0, 0]), dtype='float64')
        s = np.asarray(node.get('scale', [1, 1, 1]), dtype='float64')
        q = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype='float64')
        if t.shape != (3,) or s.shape != (3,) or q.shape != (4,):
            raise ValueError('Invalid node translation, rotation, or scale')
        length = np.linalg.norm(q)
        if not np.isfinite(length) or length < 1e-12:
            raise ValueError('Invalid node quaternion')
        x, y, z, w = q / length
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rotation = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]],
            dtype='float64')
        matrix = np.eye(4, dtype='float64')
        matrix[:3, :3] = rotation @ np.diag(s)
        matrix[:3, 3] = t
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError('Expected a finite affine node matrix')
    return matrix


def _normals(positions, indices):
    normals = np.zeros(positions.shape, dtype='float64')
    p = positions[indices].astype('float64')
    faces = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    for column in range(3):
        np.add.at(normals, indices[:, column], faces)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths == 0] = [0, 0, 1]
    lengths[lengths == 0] = 1
    return np.asarray(normals / lengths[:, None], dtype='float32')


def _primitive(spec, accessors, materials, notices):
    if (spec.get('mode', 4) != 4 or spec.get('extensions')
            or spec.get('targets')):
        raise ValueError('Only uncompressed, unmorphed TRIANGLES supported')
    attrs = spec.get('attributes', {})
    pos = accessors.read(attrs.get('POSITION'), 'VEC3', (5126,))
    if 'indices' in spec:
        idx = accessors.read(spec['indices'], 'SCALAR', (5121, 5123, 5125))
        idx = idx.reshape(-1).astype('uint32')
    else:
        idx = np.arange(len(pos), dtype='uint32')
    if len(idx) % 3 or (len(idx) and int(idx.max()) >= len(pos)):
        raise ValueError('Invalid triangle indices')
    idx = idx.reshape(-1, 3)
    if 'NORMAL' in attrs:
        normals = accessors.read(attrs['NORMAL'], 'VEC3', (5126,))
        if normals.shape != pos.shape:
            raise ValueError('Normal count does not match positions')
        lengths = np.linalg.norm(normals.astype('float64'), axis=1)
        if np.any(lengths < 1e-12):
            raise ValueError('Zero vertex normals are invalid')
        normals = np.asarray(normals / lengths[:, None], dtype='float32')
    else:
        normals = _normals(pos, idx)
        notices.append('Missing normals: area-weighted vertex normals used')
    p = pos[idx].astype('float64')
    degenerate = int(np.count_nonzero(np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) < 1e-15))
    if degenerate:
        notices.append(f'{degenerate} degenerate triangles retained')
    material = spec.get('material')
    if material is not None:
        reference(materials, material, 'material')
    uv = None
    if 'TEXCOORD_0' in attrs:
        uv = accessors.read(attrs['TEXCOORD_0'], 'VEC2', (5126,))
        if len(uv) != len(pos):
            raise ValueError('UV count does not match positions')
    return Primitive(pos, normals, idx, material, uv)


def _textures(doc, accessors):
    images = []
    for spec in doc.get('images', []):
        if 'uri' in spec:
            raise ValueError('Images must be embedded GLB bufferViews')
        mime = spec.get('mimeType')
        if mime not in ('image/png', 'image/jpeg'):
            raise ValueError('Only PNG/JPEG base-color images supported')
        view = reference(doc.get('bufferViews', []), spec.get('bufferView'),
                         'image bufferView')
        start = integer(view.get('byteOffset', 0), 'image byteOffset')
        size = integer(view.get('byteLength'), 'image byteLength', 1)
        if view.get('buffer') != 0 or start + size > len(accessors.binary):
            raise ValueError('Image exceeds embedded buffer bounds')
        data = bytes(accessors.binary[start:start + size])
        magic = b'\x89PNG\r\n\x1a\n' if mime == 'image/png' else b'\xff\xd8'
        if not data.startswith(magic):
            raise ValueError('Image signature does not match MIME type')
        images.append(data)
    result = []
    for spec in doc.get('textures', []):
        data = reference(images, spec.get('source'), 'texture image')
        sampler = (reference(doc.get('samplers', []),
                             spec['sampler'], 'sampler')
                   if 'sampler' in spec else {})
        wraps = (sampler.get('wrapS', 10497), sampler.get('wrapT', 10497))
        if any(w not in (33071, 33648, 10497) for w in wraps):
            raise ValueError('Unsupported texture wrap mode')
        result.append(Texture(data, *wraps))
    return result


def _materials(materials, textures):
    for spec in materials:
        pbr = spec.get('pbrMetallicRoughness', {})
        color = np.asarray(pbr.get('baseColorFactor', [1, 1, 1, 1]),
                           dtype='float64')
        if (color.shape != (4,) or not np.isfinite(color).all()
                or np.any(color < 0) or np.any(color > 1)):
            raise ValueError('baseColorFactor must be four values in [0, 1]')
        if spec.get('alphaMode', 'OPAQUE') not in ('OPAQUE', 'MASK', 'BLEND'):
            raise ValueError('Invalid material alphaMode')
        cutoff = float(spec.get('alphaCutoff', 0.5))
        if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError('Invalid material alphaCutoff')
        for name in ('roughnessFactor', 'metallicFactor'):
            factor = float(pbr.get(name, 1.0))
            if not np.isfinite(factor) or not 0 <= factor <= 1:
                raise ValueError(f'{name} must be a finite value in [0, 1]')
        info = pbr.get('baseColorTexture')
        if info is not None:
            reference(textures, info.get('index'), 'baseColorTexture')
            if info.get('texCoord', 0) != 0 or info.get('extensions'):
                raise ValueError('Only untransformed TEXCOORD_0 supported')


def _graph(nodes, roots):
    parents = {}
    for index, node in enumerate(nodes):
        for child in node.children:
            reference(nodes, child, 'child node')
            if child in parents:
                raise ValueError('A node cannot have multiple parents')
            parents[child] = index
    for root in roots:
        reference(nodes, root, 'scene root')
        if root in parents:
            raise ValueError('Scene root cannot have a parent')
    for index in range(len(nodes)):
        visited = set()
        while index in parents:
            if index in visited or len(visited) > 256:
                raise ValueError('Cyclic or excessively deep node hierarchy')
            visited.add(index)
            index = parents[index]


def load_glb(path):
    """Read static GLB geometry; unsupported required features raise errors."""
    try:
        doc, binary = _chunks(Path(path).read_bytes())
        if doc.get('asset', {}).get('version') != '2.0':
            raise ValueError('Expected glTF asset version 2.0')
        if doc.get('asset', {}).get('minVersion', '2.0') != '2.0':
            raise ValueError('Unsupported minimum glTF version')
        if doc.get('extensionsRequired'):
            raise ValueError('Required glTF extensions unsupported: '
                             + str(doc['extensionsRequired']))
        accessors = Accessors(doc, binary)
        notices = []
        if doc.get('animations'):
            notices.append('Static pose only; animation playback unavailable')
        materials = doc.get('materials', [])
        textures = _textures(doc, accessors)
        _materials(materials, textures)
        meshes = []
        for mesh in doc.get('meshes', []):
            if mesh.get('weights'):
                raise ValueError('Morph targets unsupported')
            specs = mesh.get('primitives', [])
            if not specs:
                raise ValueError('Mesh must contain primitives')
            meshes.append([_primitive(s, accessors, materials, notices)
                           for s in specs])
        for mesh in meshes:
            for primitive in mesh:
                if primitive.material is not None:
                    pbr = materials[primitive.material].get(
                        'pbrMetallicRoughness', {})
                    if ('baseColorTexture' in pbr
                            and primitive.texcoords is None):
                        raise ValueError('Textured primitive needs TEXCOORD_0')
        nodes = []
        for index, spec in enumerate(doc.get('nodes', [])):
            if 'skin' in spec or spec.get('weights'):
                raise ValueError('Skinning and morph targets unsupported')
            mesh = spec.get('mesh')
            if mesh is not None:
                reference(meshes, mesh, 'mesh')
            nodes.append(Node(str(spec.get('name', f'Node {index}')),
                              tuple(spec.get('children', [])),
                              _matrix(spec), mesh))
        scene = reference(doc.get('scenes', []), doc.get('scene', 0), 'scene')
        roots = tuple(scene.get('nodes', []))
        if not roots or len(set(roots)) != len(roots):
            raise ValueError('Expected distinct scene roots')
        _graph(nodes, roots)
        return Model(nodes, meshes, materials, roots,
                     list(dict.fromkeys(notices)), textures)
    except (KeyError, TypeError, IndexError, AttributeError,
            OverflowError) as exc:
        raise ValueError(f'Malformed GLB structure: {exc}') from exc

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
