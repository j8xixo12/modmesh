# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Map solvcon full-vehicle chassis and upright poses onto the GLB."""

import math
import numpy as np
from .visual_pose import quaternion_product

WHEEL_TO_CORNER = {
    'WHEEL_LF': 'front_left',
    'WHEEL_RF': 'front_right',
    'WHEEL_LR': 'rear_left',
    'WHEEL_RR': 'rear_right',
}


def inverse(q):
    q = np.asarray(q, dtype=float)
    return q * [1, -1, -1, -1] / np.dot(q, q)


def transforms(items, groups, pivots, pose, initial, wheel_spin=0.0):
    from solvcon.multibody.suspension.metrics import rot

    p, q = pose['chassis']
    p0, q0 = initial['chassis']
    delta_q = quaternion_product(q, inverse(q0))
    # The physical front axle is x=0; the GLB front wheels locate that axle.
    front_x = 0.5 * (pivots['WHEEL_LF'][0] + pivots['WHEEL_RF'][0])
    cg = np.asarray(initial['cg']) + [front_x, 0, 0]
    chassis_shift = cg + (p - p0) - rot(delta_q) @ cg
    spin_q = np.array(
        [math.cos(wheel_spin / 2), 0, math.sin(wheel_spin / 2), 0],
        dtype='float64',
    )
    result = {}
    for item in items:
        wheel = groups.get(item.node)
        if wheel is None:
            result[item.key] = (chassis_shift.copy(), delta_q.copy())
            continue
        corner = WHEEL_TO_CORNER[wheel]
        position, quat = pose['corners'][corner]['upright']
        position0, quat0 = initial['corners'][corner]['upright']
        wheel_q = quaternion_product(
            quaternion_product(quat, inverse(quat0)), spin_q
        )
        pivot = pivots[wheel]
        shift = pivot + (position - position0) - rot(wheel_q) @ pivot
        result[item.key] = (shift, wheel_q)
    return result


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
