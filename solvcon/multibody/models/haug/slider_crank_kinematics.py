# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Spatial slider-crank of Haug Section 10.2 (kinematic analysis).

The builder drives the crank at a constant rate about the global x axis,
either through the legacy type-3 joint or through the relative rotational
driver of Haug Eq. 9.5.4.
"""

import math

from .... import multibody as mbs

Z = [0, 0, 0]
AX = [1, 0, 0]
AY = [0, 1, 0]


def build_slider_crank(use_new_driver=False, crank_func=None):
    """Build the Haug Section 10.2 spatial slider-crank mechanism.

    The same mechanism as ``models.gallery.build_slider_crank``.

    :param use_new_driver: drive the crank with the relative rotational
        driver of Haug Eq. 9.5.4 instead of the legacy type-3 joint
    :type use_new_driver: bool
    :param crank_func: drive function for the new driver; default is
        theta_0 + omega_0 t
    :type crank_func: solvcon.multibody.DriveFunction or None
    :return: (system, ground, crank, rod, slider, dt) tuple
    :rtype: tuple
    """
    dt = 0.0001
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    omega_0 = 2.0 * math.pi
    e0_c = math.sqrt(1.0 - 0.71**2)
    crank = mbs.create_mobilized_body_quat(
        1, [0.0, 0.1, 0.12], Z, Z,
        [e0_c, 0.71, 0.0, 0.0], [omega_0, 0, 0], Z,
        1.0, [0.01, 0.01, 0.01], Z, Z)
    s.Add(crank)
    e0_r = math.sqrt(1.0 - 0.21**2 - 0.40**2 - 0.1**2)
    rod = mbs.create_mobilized_body_quat(
        2, [0.1, 0.05, 0.1], Z, Z,
        [e0_r, -0.21, 0.40, -0.1], Z, Z,
        1.0, [0.01, 0.01, 0.01], Z, Z)
    s.Add(rod)
    slider = mbs.create_mobilized_body_quat(
        3, [0.2, 0.0, 0.0], Z, Z,
        [1.0, 0.0, 0.0, 0.0], Z, Z,
        1.0, [0.01, 0.01, 0.01], Z, Z)
    s.Add(slider)
    s.Add(mbs.create_joint(
        1, Z, [0, 0.1, 0.12], Z, Z, AX, AX, crank, g))
    s.Add(mbs.create_joint(
        0, [0, 0.08, 0], [-0.15, 0, 0], Z, Z, AX, AX, crank, rod))
    s.Add(mbs.create_joint(
        7, [0.15, 0, 0], [0.2, 0, 0], Z, Z, AY, AX, rod, slider))
    s.Add(mbs.create_joint(
        2, Z, [0.2, 0, 0], Z, Z, AX, AX, slider, g))
    s.Add(mbs.create_joint_distance(
        6, [0.15, 0, 0], [1.0, 0, 0], Z, Z,
        AX, AX, rod, slider, 1.0))
    if use_new_driver:
        theta_0 = 2.0 * math.atan2(0.71, e0_c)  # initial crank angle about x
        s.Add(mbs.create_rotational_driver(
            body_i=g, body_j=crank, axis_i=AX, axis_j=AX,
            func=crank_func or mbs.DriveFunction.polynomial([theta_0,
                                                             omega_0])))
    else:
        s.Add(mbs.create_joint_driving(
            3, Z, Z, Z, Z, AX, AX, g, crank, omega_0))
    return s, g, crank, rod, slider, dt

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
