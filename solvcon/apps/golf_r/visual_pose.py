# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Map planar vehicle and wheel poses to the existing GLB viewer surfaces."""

import math
import numpy as np

WHEELS = ('WHEEL_LF', 'WHEEL_RF', 'WHEEL_LR', 'WHEEL_RR')


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype='float64')


def quaternion_product(a, b):
    w, x, y, z = a
    q, i, j, k = b
    return np.array(
        [
            w * q - x * i - y * j - z * k,
            w * i + x * q + y * k - z * j,
            w * j - x * k + y * q + z * i,
            w * k + x * j - y * i + z * q,
        ],
        dtype='float64',
    )


def wheel_groups(model, world_matrices):
    names = {}
    for index, node in enumerate(model.nodes):
        if node.name in WHEELS and index in world_matrices:
            if node.name in names:
                raise ValueError(f'Duplicate wheel node: {node.name}')
            names[node.name] = index
    if set(names) != set(WHEELS):
        raise ValueError(
            'GLB must contain all four uniquely named wheel nodes'
        )
    groups, pivots = {}, {}
    for name, index in names.items():
        pivots[name] = world_matrices[index][:3, 3].copy()
        stack = [index]
        while stack:
            current = stack.pop()
            if current in groups:
                raise ValueError('Wheel hierarchies overlap')
            groups[current] = name
            stack.extend(model.nodes[current].children)
    return groups, pivots


def transforms(
    items, groups, pivots, yaw, steer, rotation, translation=(0, 0, 0)
):
    """Rigid transforms in the viewer's x-forward, y-left, z-up basis.

    A zero translation follows the vehicle; supply world x/y to show its
    trajectory. Positive wheel rotation is about local +Y.
    """
    chassis_q = np.array(
        [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], dtype='float64'
    )
    chassis_r = rotation_z(yaw)
    base = np.asarray(translation, dtype='float64')
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError('translation must be three finite values')
    output = {}
    for item in items:
        wheel = groups.get(item.node)
        if wheel is None:
            output[item.key] = (base.copy(), chassis_q.copy())
            continue
        angle = steer if wheel.endswith(('LF', 'RF')) else 0.0
        steer_q = np.array(
            [math.cos(angle / 2), 0, 0, math.sin(angle / 2)], dtype='float64'
        )
        roll_q = np.array(
            [math.cos(rotation / 2), 0, math.sin(rotation / 2), 0],
            dtype='float64',
        )
        q = quaternion_product(chassis_q, quaternion_product(steer_q, roll_q))
        c, s = math.cos(rotation), math.sin(rotation)
        roll_r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype='float64')
        wheel_r = rotation_z(angle) @ roll_r
        pivot = pivots[wheel]
        shift = base + chassis_r @ (pivot - wheel_r @ pivot)
        output[item.key] = (shift, q)
    return output


def apply(widget, poses):
    for key, (translation, q) in poses.items():
        widget.setObjectTransform(
            key, *map(float, translation), 1.0, 1.0, 1.0, *map(float, q)
        )


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
