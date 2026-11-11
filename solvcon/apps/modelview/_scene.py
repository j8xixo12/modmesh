# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Adapt glTF scene instances to static, Z-up display surfaces."""

from dataclasses import dataclass

import numpy as np

# A proper basis rotation: glTF +Z becomes display +X, +Y becomes +Z.
GLTF_TO_DISPLAY = np.array([[0, 0, 1, 0], [1, 0, 0, 0],
                            [0, 1, 0, 0], [0, 0, 0, 1]], dtype='float64')


@dataclass
class Surface:
    key: str
    node: int
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    color: tuple
    material: str
    texcoords: np.ndarray | None
    texture: bytes | None
    alpha_mode: str
    alpha_cutoff: float
    wrap_s: int
    wrap_t: int
    roughness: float = 1.0
    metallic: float = 1.0


def world_matrices(model):
    """World transforms of nodes in the selected scene, in display axes."""
    result = {}
    stack = [(i, GLTF_TO_DISPLAY) for i in reversed(model.roots)]
    while stack:
        index, parent = stack.pop()
        node = model.nodes[index]
        world = parent @ node.matrix
        result[index] = world
        stack.extend((c, world) for c in reversed(node.children))
    return result


def surfaces(model):
    """Bake instance transforms without losing the model's source hierarchy."""
    output = []
    for index, world in world_matrices(model).items():
        node = model.nodes[index]
        if node.mesh is None:
            continue
        linear = world[:3, :3]
        if not np.isfinite(world).all():
            raise ValueError(f'Nonfinite world transform: {node.name}')
        try:
            normal_matrix = np.linalg.inv(linear).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(f'Singular transform: {node.name}') from exc
        for number, primitive in enumerate(model.meshes[node.mesh]):
            positions = (primitive.positions.astype('float64') @ linear.T
                         + world[:3, 3])
            normals = primitive.normals.astype('float64') @ normal_matrix.T
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            indices = primitive.indices.copy()
            if np.linalg.det(linear) < 0:
                indices = indices[:, [0, 2, 1]]
            if not (np.isfinite(positions).all()
                    and np.max(np.abs(positions)) <= np.finfo('float32').max):
                raise ValueError(f'Display positions overflow: {node.name}')
            mat = primitive.material
            spec = {} if mat is None else model.materials[mat]
            pbr = spec.get('pbrMetallicRoughness', {})
            color = tuple(pbr.get('baseColorFactor', [1, 1, 1, 1]))
            info = pbr.get('baseColorTexture')
            texture = None if info is None else model.textures[info['index']]
            name = ('Default' if mat is None else
                    str(model.materials[mat].get('name', f'Material {mat}')))
            output.append(Surface(
                f'node:{index}/primitive:{number}', index,
                np.asarray(positions, dtype='float32'),
                np.asarray(normals, dtype='float32'),
                np.ascontiguousarray(indices, dtype='uint32'), color, name,
                primitive.texcoords, None if texture is None else texture.data,
                spec.get('alphaMode', 'OPAQUE'),
                float(spec.get('alphaCutoff', 0.5)),
                10497 if texture is None else texture.wrap_s,
                10497 if texture is None else texture.wrap_t,
                float(pbr.get('roughnessFactor', 1.0)),
                float(pbr.get('metallicFactor', 1.0))))
    return output


def bounds(items):
    if not items:
        raise ValueError('Selected scene has no triangle surfaces')
    lo = np.min(np.array([s.positions.min(axis=0) for s in items],
                         dtype='float64'), axis=0)
    hi = np.max(np.array([s.positions.max(axis=0) for s in items],
                         dtype='float64'), axis=0)
    return lo, hi

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
