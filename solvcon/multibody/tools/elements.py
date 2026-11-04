# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Stiffness matrices for force elements of the compiled solver.

All matrices act on the bushing deformation [dx, dy, dz, rx, ry, rz] of
``solvcon.multibody.create_bushing`` (frame B on body i, unloaded at the
construction pose).
"""

import numpy as np


def bushing_matrix(trans, rot):
    """Diagonal 6x6 matrix from translational and rotational entries.

    :param trans: [kx, ky, kz] (N/m) or [cx, cy, cz] (N s/m)
    :type trans: list
    :param rot: [krx, kry, krz] (N m/rad) or damping (N m s/rad)
    :type rot: list
    :return: 6x6 matrix as nested lists
    :rtype: list
    """
    return np.diag(list(trans) + list(rot)).tolist()


def beam_stiffness(e_mod, g_mod, area, i_y, i_z, j_t, length):
    """Euler-Bernoulli cantilever stiffness of a straight massless beam.

    The beam runs from point i (clamped, frame B origin) along +x of frame B
    to point j. The matrix maps the tip deformation [u, v, w, rx, ry, rz] to
    the restoring tip load [Fx, Fy, Fz, Mx, My, Mz] = -K q. It is the inverse
    of the cantilever compliance, so the coupling terms are -6EIz/L^2 (v, rz)
    and +6EIy/L^2 (w, ry).

    :param e_mod: Young's modulus (Pa)
    :type e_mod: float
    :param g_mod: shear modulus (Pa)
    :type g_mod: float
    :param area: cross-section area (m^2)
    :type area: float
    :param i_y: second moment of area about local y (m^4)
    :type i_y: float
    :param i_z: second moment of area about local z (m^4)
    :type i_z: float
    :param j_t: torsion constant (m^4)
    :type j_t: float
    :param length: beam length (m)
    :type length: float
    :return: 6x6 stiffness as nested lists
    :rtype: list
    """
    if length <= 0.0:
        raise ValueError(f"beam length must be positive, got {length}")
    ln = length
    k = np.zeros((6, 6))
    k[0, 0] = e_mod * area / ln
    k[1, 1] = 12 * e_mod * i_z / ln ** 3
    k[1, 5] = k[5, 1] = -6 * e_mod * i_z / ln ** 2
    k[5, 5] = 4 * e_mod * i_z / ln
    k[2, 2] = 12 * e_mod * i_y / ln ** 3
    k[2, 4] = k[4, 2] = 6 * e_mod * i_y / ln ** 2
    k[4, 4] = 4 * e_mod * i_y / ln
    k[3, 3] = g_mod * j_t / ln
    return k.tolist()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
