# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Section 10.3: kinematic analysis of a spatial four-bar mechanism.

Reference: E. J. Haug, "Computer-Aided Kinematics and Dynamics of
Mechanical Systems, Vol. I", Section 10.3, pp. 402-408. The model data are
in Tables 10.3.1 to 10.3.5. The results are Table 10.3.6 (assembled
configuration) and Figs. 10.3.3 to 10.3.5. The transcribed book data are in
``tests/data/multibody/haug_10_3.json``.

Body numbers. Haug numbers the links 1, 2, 3 and the ground 4. The solver
needs the ground as body 0, so the ground (Haug 4) is body 0 here. Links 1,
2, and 3 keep their numbers.

Model 1 (Fig. 10.3.1): revolute A (ground-link 1), universal B
(link 1-link 2), spherical C (link 2-link 3), revolute D (link 3-ground).
Model 2 (Fig. 10.3.2): link 2 is a spherical-spherical (distance)
constraint between link 1 and the rocker. The book numbers the rocker 2 in
model 2; this module keeps it as body 2 of a three-body model.

Driver (Sec. 10.3.3): the relative angle of revolute A is theta = pi t,
measured about the global x axis.

Run this file to print the assembled configuration and a short time
history::

    python solvcon/multibody/models/haug/spatial_four_bar.py
"""

import math

from .... import multibody as mbs
from ...tools.arrays import as_list

Z = [0.0, 0.0, 0.0]
AX = [1.0, 0.0, 0.0]

#: Crank speed of the driver theta = pi t (Sec. 10.3.3) [rad/s].
CRANK_SPEED = math.pi

#: Coupler length. Model 1 fixes |BC| = 6.1 + 6.1 (Table 10.3.2, 10.3.3).
#: The book prints 12.19 m for model 2 (p. 406) and states that model 2
#: gives identical results, so 12.19 is a rounded value of 12.2.
COUPLER_LENGTH = 12.2


def _quat(e1, e2, e3):
    """Return a unit Euler parameter vector from its vector part.

    Table 10.3.5 lists only e1, e2, e3. The value e0 >= 0 follows from the
    normalization constraint (Haug Eq. 9.6.3).

    :param e1: Euler parameter e1
    :type e1: float
    :param e2: Euler parameter e2
    :type e2: float
    :param e3: Euler parameter e3
    :type e3: float
    :return: Euler parameters [e0, e1, e2, e3]
    :rtype: list
    """
    e0 = math.sqrt(max(0.0, 1.0 - e1 * e1 - e2 * e2 - e3 * e3))
    return [e0, e1, e2, e3]


def _body(num, pos, quat, mass, inertia):
    """Create a mobilized body at rest without applied forces.

    :param num: body number (insertion order)
    :type num: int
    :param pos: initial estimate of the origin position (global)
    :type pos: list
    :param quat: initial estimate of the Euler parameters
    :type quat: list
    :param mass: mass [kg]
    :type mass: float
    :param inertia: principal moments of inertia [kg m^2]
    :type inertia: list
    :return: solver body
    :rtype: solvcon.multibody.MobilizedBody
    """
    return mbs.create_mobilized_body_quat(num, pos, Z, Z, quat, Z, Z, mass,
                                          inertia, Z, Z)


def build_model_1(dt=0.01, crank_speed=CRANK_SPEED):
    """Build model 1 of the spatial four-bar (Fig. 10.3.1).

    The bodies start at the initial estimates of Table 10.3.5. ``init()``
    assembles the mechanism (Sec. 10.3.2). The mass data do not enter the
    kinematic results; unit values are used.

    :param dt: kinematic output step [s]
    :type dt: float
    :param crank_speed: driver rate of theta = crank_speed * t [rad/s]
    :type crank_speed: float
    :return: (system, bodies) where bodies maps a name to a solver body
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    link1 = _body(1, [0.0, 0.0, 0.0], _quat(0.0, 0.0, 0.0), 1.0,
                  [1.0, 1.0, 1.0])
    link2 = _body(2, [-3.75, -4.25, 4.25], _quat(-0.29, -0.27, -0.26), 1.0,
                  [1.0, 1.0, 1.0])
    link3 = _body(3, [-5.75, -8.5, 3.25], _quat(-0.36, 0.36, -0.61), 1.0,
                  [1.0, 1.0, 1.0])
    for b in (link1, link2, link3):
        s.Add(b)

    # Revolute A (Table 10.3.1): P at the origin of both frames, axis PQ = x.
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AX, AX, ground, link1))
    # Revolute D (Table 10.3.1): link 3 P = (0, 3.7, 0), axis x'; ground
    # P = (-4, -8.5, 0), axis PQ = (0, -1, 0).
    s.Add(mbs.create_joint(1, [0.0, 3.7, 0.0], [-4.0, -8.5, 0.0], Z, Z,
                           AX, [0.0, -1.0, 0.0], link3, ground))
    # Universal B (Table 10.3.2): link 1 P = (0, 0, 2), axis PQ =
    # (0.750, -0.662, 0); link 2 P = (0, 6.1, 0), axis PQ = (0, 0, 1).
    s.Add(mbs.create_joint(4, [0.0, 0.0, 2.0], [0.0, 6.1, 0.0], Z, Z,
                           [0.750, -0.662, 0.0], [0.0, 0.0, 1.0], link1,
                           link2))
    # Spherical C (Table 10.3.3): link 2 P = (0, -6.1, 0), link 3 P = (0, -3.7,
    # 0).
    s.Add(mbs.create_joint(0, [0.0, -6.1, 0.0], [0.0, -3.7, 0.0], Z, Z,
                           AX, AX, link2, link3))
    driver = mbs.create_rotational_driver(
        ground, link1, AX, AX,
        mbs.DriveFunction.polynomial([0.0, crank_speed]))
    s.Add(driver)
    return s, {"ground": ground, "link1": link1, "coupler": link2,
               "rocker": link3, "driver": driver}


def build_model_2(dt=0.01, crank_speed=CRANK_SPEED,
                  coupler_length=COUPLER_LENGTH):
    """Build model 2 of the spatial four-bar (Fig. 10.3.2).

    Link BC is a spherical-spherical composite joint (Table 10.3.4), that
    is a distance constraint between P = (0, 0, 2) on link 1 and
    P = (0, -3.7, 0) on the rocker. The rocker is body 2.

    :param dt: kinematic output step [s]
    :type dt: float
    :param crank_speed: driver rate of theta = crank_speed * t [rad/s]
    :type crank_speed: float
    :param coupler_length: length of the spherical-spherical link [m]
    :type coupler_length: float
    :return: (system, bodies) where bodies maps a name to a solver body
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    link1 = _body(1, [0.0, 0.0, 0.0], _quat(0.0, 0.0, 0.0), 1.0,
                  [1.0, 1.0, 1.0])
    rocker = _body(2, [-5.75, -8.5, 3.25], _quat(-0.36, 0.36, -0.61), 1.0,
                   [1.0, 1.0, 1.0])
    s.Add(link1)
    s.Add(rocker)
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AX, AX, ground, link1))
    s.Add(mbs.create_joint(1, [0.0, 3.7, 0.0], [-4.0, -8.5, 0.0], Z, Z,
                           AX, [0.0, -1.0, 0.0], rocker, ground))
    s.Add(mbs.create_joint_distance(6, [0.0, 0.0, 2.0], [0.0, -3.7, 0.0], Z, Z,
                                    AX, AX, link1, rocker, coupler_length))
    driver = mbs.create_rotational_driver(
        ground, link1, AX, AX,
        mbs.DriveFunction.polynomial([0.0, crank_speed]))
    s.Add(driver)
    return (s,
            {"ground": ground, "link1": link1, "rocker": rocker,
             "driver": driver})


def rotation_matrix(e):
    """Return Haug's A(p), body-to-global (Eq. 6.1.13).

    :param e: Euler parameters [e0, e1, e2, e3]
    :type e: list
    :return: 3x3 rotation matrix as nested lists
    :rtype: list
    """
    e0, e1, e2, e3 = e
    return [
        [2 * (e0 * e0 + e1 * e1) - 1, 2 * (e1 * e2 - e0 * e3),
         2 * (e1 * e3 + e0 * e2)],
        [2 * (e1 * e2 + e0 * e3), 2 * (e0 * e0 + e2 * e2) - 1,
         2 * (e2 * e3 - e0 * e1)],
        [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1),
         2 * (e0 * e0 + e3 * e3) - 1],
    ]


def global_point(pos, e, local):
    """Return the global position of a body-fixed point.

    :param pos: body origin (global)
    :type pos: list
    :param e: body Euler parameters [e0, e1, e2, e3]
    :type e: list
    :param local: point in body coordinates
    :type local: list
    :return: global point
    :rtype: list
    """
    a = rotation_matrix(e)
    return [pos[i] + sum(a[i][k] * local[k] for k in range(3)) for i
            in range(3)]


def simulate(system, t_end, rocker=3):
    """Run kinematic analysis and record the book output quantities.

    :param system: initialized solver system
    :type system: solvcon.multibody.DynamicsSystem
    :param t_end: end time [s]
    :type t_end: float
    :param rocker: body number of the rocker (link 3)
    :type rocker: int
    :return: dict of lists: t, z_B (Fig. 10.3.3), z_rocker (Fig. 10.3.4),
        zdot_rocker (Fig. 10.3.5)
    :rtype: dict
    """
    out = {"t": [], "z_B": [], "z_rocker": [], "zdot_rocker": []}

    def record():
        pos = as_list(system.get_body_positions())
        quat = as_list(system.get_body_quaternions())
        vel = as_list(system.get_body_velocities())
        out["t"].append(system.get_sim_time())
        out["z_B"].append(global_point(list(pos[1]), list(quat[1]),
                                       [0.0, 0.0, 2.0])[2])
        out["z_rocker"].append(float(pos[rocker][2]))
        out["zdot_rocker"].append(float(vel[rocker][2]))

    record()
    while system.get_sim_time() < t_end - 1e-9:
        system.solve()
        record()
    return out


def main():
    """Print the assembled configuration and the extreme rocker values."""
    s, _ = build_model_1(dt=0.005)
    s.init()
    names = ["Ground (4)", "Link 1", "Link 2", "Link 3"]
    for name, p, e in zip(names, as_list(s.get_body_positions()),
                          as_list(s.get_body_quaternions())):
        print(f"{name:10s} r = ({p[0]:+.5f}, {p[1]:+.5f}, {p[2]:+.5f})  "
              f"e = ({e[0]:+.5f}, {e[1]:+.5f}, {e[2]:+.5f}, {e[3]:+.5f})")
    res = simulate(s, 2.5)
    zr = res["z_rocker"]
    print(f"rocker z: min {min(zr):.4f}, max {max(zr):.4f}")
    print(f"rocker zdot: min {min(res['zdot_rocker']):.4f}, max "
          f"{max(res['zdot_rocker']):.4f}")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
