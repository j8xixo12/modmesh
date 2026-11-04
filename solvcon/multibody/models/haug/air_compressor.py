# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Sections 10.4 and 12.4: six-piston air compressor (wobble plate).

Reference: E. J. Haug, "Computer-Aided Kinematics and Dynamics of
Mechanical Systems, Vol. I". Section 10.4 (pp. 408-414) gives the model
(Fig. 10.4.1, Tables 10.4.1 to 10.4.4), the assembled configuration
(Table 10.4.5), and the piston 1 kinematics (Figs. 10.4.2 to 10.4.4).
Section 12.4 (pp. 462-468) adds the inertia data (Table 12.4.1), the air
pressure force (Fig. 12.4.1, p. 464), inverse dynamics (Sec. 12.4.2,
Figs. 12.4.2 to 12.4.5), and dynamics (Sec. 12.4.3, Figs. 12.4.6 to
12.4.8). The transcribed book data are in
``tests/data/multibody/haug_10_4.json`` and
``tests/data/multibody/haug_12_4.json``.

Body numbers. Haug numbers the ground 1, the rotor 2, the disk 3, and
pistons 1 to 6 as bodies 4 to 9. The solver needs the ground as body 0, so
every solver body number is the Haug number minus 1.

Model (Sec. 10.4.1):

* revolute A (ground-rotor), axis global y;
* revolute B (rotor-disk), rotor axis 30 degrees from y (the offset angle);
* translational joints C to H (ground-pistons), axis global y;
* spherical-spherical joints C-I to H-N (disk-pistons), length 0.5 m,
  modeled as distance constraints;
* x position constraint at point O of the disk (the slot that stops the
  disk from spinning);
* driver: relative angle of revolute A, theta = 62.832 t (600 rpm).

Point O. The book gives no coordinates for O. Fig. 10.4.1 shows O on the
local -z axis of the disk, next to the joint point N (radius 0.2 m, Table
10.4.3). The kinematics do not depend on the radius of O, because the
constraint is x_O = 0 and the disk origin stays on the rotor axis. The x
constraint force is proportional to 1 / radius. ``O_RADIUS`` = 0.2 m is an
assumption (the radius of N). The 0 psi curve of Fig. 12.4.3 agrees with it;
a radius of 0.25 m gives a force amplitude 20 % below the book.

Run this file to print the assembled configuration::

    python solvcon/multibody/models/haug/air_compressor.py
"""

import math

from .... import multibody as mbs
from ...tools.arrays import as_list, reaction_arrays

Z = [0.0, 0.0, 0.0]
AX = [1.0, 0.0, 0.0]
AY = [0.0, 1.0, 0.0]

#: Rotor speed, theta = 62.832 t (Sec. 10.4.3, 600 rpm) [rad/s].
ROTOR_SPEED = 62.832

#: Piston attachment points (x', z') on the disk, Table 10.4.3. The same
#: (x, z) locate the cylinder axes on the ground, Table 10.4.2.
PISTON_XZ = [
    (0.0, 0.2),
    (0.1732, 0.1),
    (-0.1732, 0.1),
    (0.1732, -0.1),
    (-0.1732, -0.1),
    (0.0, -0.2),
]

#: Cylinder reference point y on the ground (Table 10.4.2) [m].
CYLINDER_Y = -1.0

#: Rotor point of revolute B (Table 10.4.1) [m].
ROTOR_B_Y = -0.5

#: Connecting rod length (Table 10.4.3) [m].
ROD_LENGTH = 0.5

#: Radius of point O on the disk (assumption, see the module docstring) [m].
O_RADIUS = 0.2

#: Inertia data, Table 12.4.1 (mass [kg], [Ixx, Iyy, Izz] [kg m^2]).
ROTOR_MASS, ROTOR_INERTIA = 10.0, [0.1, 0.01, 0.1]
DISK_MASS, DISK_INERTIA = 2.0, [0.04, 0.08, 0.04]
PISTON_MASS, PISTON_INERTIA = 1.5, [0.002, 0.002, 0.002]


def build(offset_deg=30.0, rotor_speed=ROTOR_SPEED, theta0=0.0, dt=0.001,
          driven=True, piston_forces=None, initial=None, rotor_omega=0.0,
          rotor_torque=0.0):
    """Build the air compressor model.

    The initial estimates follow Table 10.4.4. For an offset angle other
    than 30 degrees, the disk estimate is e1 = -sin(offset / 2).

    :param offset_deg: angle between the rotor axis and the revolute B
        axis [deg] (Sec. 10.4.4: 20, 30, 40)
    :type offset_deg: float
    :param rotor_speed: driver rate [rad/s]
    :type rotor_speed: float
    :param theta0: driver angle at t = 0 [rad]
    :type theta0: float
    :param dt: output step [s]
    :type dt: float
    :param driven: if True, add the rotor driver (kinematics, inverse
        dynamics); if False, the rotor is free (dynamics)
    :type driven: bool
    :param piston_forces: global y force on each piston [N] (6 values), or
        None for no force
    :type piston_forces: list or None
    :param initial: optional (positions, quaternions) per solver body that
        replace the Table 10.4.4 estimates
    :type initial: tuple or None
    :param rotor_omega: initial rotor angular velocity about +y [rad/s]
        (dynamics only)
    :type rotor_omega: float
    :param rotor_torque: constant torque on the rotor about +y [N m]
    :type rotor_torque: float
    :return: (system, parts) where parts maps a name to a solver object
    :rtype: tuple
    """
    a = math.radians(offset_deg)
    forces = piston_forces if piston_forces is not None else [0.0] * 6

    est_pos = [[0.0, 0.0, 0.0], [0.0, ROTOR_B_Y, 0.0]]
    est_quat = [[1.0, 0.0, 0.0, 0.0],
                [math.cos(a / 2), -math.sin(a / 2), 0.0, 0.0]]
    for x, z in PISTON_XZ:
        est_pos.append([x, CYLINDER_Y, z])
        est_quat.append([1.0, 0.0, 0.0, 0.0])
    if initial is not None:
        est_pos = [list(p) for p in initial[0][1:]]
        est_quat = [list(q) for q in initial[1][1:]]

    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    rotor = mbs.create_mobilized_body_quat(
        1, est_pos[0], Z, Z, est_quat[0], [0.0, rotor_omega, 0.0], Z,
        ROTOR_MASS, ROTOR_INERTIA, Z, [0.0, rotor_torque, 0.0])
    disk = mbs.create_mobilized_body_quat(
        2, est_pos[1], Z, Z, est_quat[1], Z, Z, DISK_MASS, DISK_INERTIA, Z, Z)
    s.Add(rotor)
    s.Add(disk)
    pistons = []
    for k in range(6):
        p = mbs.create_mobilized_body_quat(
            3 + k, est_pos[2 + k], Z, Z, est_quat[2 + k], Z, Z,
            PISTON_MASS, PISTON_INERTIA, [0.0, forces[k], 0.0], Z)
        s.Add(p)
        pistons.append(p)

    # Revolute A (Table 10.4.1): P at the origin, axis PQ = y on both bodies.
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AY, AY, ground, rotor))
    # Revolute B (Table 10.4.1): rotor P = (0, -0.5, 0), axis PQ =
    # (0, cos a, -sin a); disk P = origin, axis y'.
    s.Add(mbs.create_joint(1, [0.0, ROTOR_B_Y, 0.0], Z, Z, Z,
                           [0.0, math.cos(a), -math.sin(a)], AY, rotor, disk))
    # Translational C to H (Table 10.4.2): ground P = (x, -1, z), axis y.
    for (x, z), p in zip(PISTON_XZ, pistons):
        s.Add(mbs.create_joint(2, Z, [x, CYLINDER_Y, z], Z, Z, AY, AY, p,
                               ground))
    # Spherical-spherical C-I to H-N (Table 10.4.3): length 0.5 m.
    for (x, z), p in zip(PISTON_XZ, pistons):
        s.Add(mbs.create_joint_distance(6, [x, 0.0, z], Z, Z, Z, AX, AX, disk,
                                        p, ROD_LENGTH))
    # x position constraint at O (Sec. 10.4.1): x of the disk point O = 0.
    slot = mbs.create_absolute_driver(disk, ground, [0.0, 0.0, -O_RADIUS], AX,
                                      mbs.DriveFunction.polynomial([0.0]))
    s.Add(slot)
    parts = {"ground": ground, "rotor": rotor, "disk": disk,
             "pistons": pistons, "slot": slot, "driver": None}
    if driven:
        driver = mbs.create_rotational_driver(
            ground, rotor, AY, AY,
            mbs.DriveFunction.polynomial([theta0, rotor_speed]))
        s.Add(driver)
        parts["driver"] = driver
    return s, parts


def reaction(system, joint):
    """Return the reaction record of one joint.

    :param system: solver system after ``init()`` or ``solve()``
    :type system: solvcon.multibody.DynamicsSystem
    :param joint: joint object that was added to the system
    :type joint: solvcon.multibody.Joint
    :return: reaction dict of ``get_joint_reactions()``
    :rtype: dict
    """
    joints = system.get_joints()
    for j, r in zip(joints, reaction_arrays(system.get_joint_reactions())):
        if j is joint:
            return r
    raise ValueError("joint not in system")


def kinematics(offset_deg=30.0, t_end=0.2, dt=0.001, rotor_speed=ROTOR_SPEED):
    """Run kinematic analysis and return the piston 1 history.

    :param offset_deg: offset angle [deg]
    :type offset_deg: float
    :param t_end: end time [s]
    :type t_end: float
    :param dt: step [s]
    :type dt: float
    :param rotor_speed: driver rate [rad/s]
    :type rotor_speed: float
    :return: dict with lists t, y, ydot, yddot (piston 1, solver body 3)
    :rtype: dict
    """
    s, _ = build(offset_deg=offset_deg, rotor_speed=rotor_speed, dt=dt)
    s.init()
    out = {"t": [], "y": [], "ydot": [], "yddot": []}

    def record():
        out["t"].append(s.get_sim_time())
        out["y"].append(float(as_list(s.get_body_positions())[3][1]))
        out["ydot"].append(float(as_list(s.get_body_velocities())[3][1]))
        out["yddot"].append(
            float(as_list(s.get_body_accelerations())[3][1]))

    record()
    while s.get_sim_time() < t_end - 1e-9:
        s.solve()
        record()
    return out


#: Air pressure force data (Sec. 12.4.1, p. 464).
PSI = 6894.757293168
BORE = 0.1
Y_A = -1.1
Y_B = -0.9
#: AC = 0.04 m for the 5:1 compression ratio over the 0.2 m stroke.
Y_C = Y_A + 0.04


def air_force(y, ydot, p_max_psi):
    """Return the air pressure force on one piston (Fig. 12.4.1).

    The book prints F_DB = F_max / (y - y_A) - F_min. That form is not
    dimensionally consistent (N/m). This function uses the isothermal form
    that matches the stated 5:1 compression ratio and zero gauge force at
    B::

        F_DB = F_max (y_C - y_A) / (y - y_A) - F_min,  F_min = F_max / 5
        F_AD = -F_C / 2 cos(pi (y - y_A) / (y_C - y_A)) + F_C / 2
        F_C = F_DB(y_C) = F_max - F_min

    with F_max = (piston area) x (maximum pressure). The force acts only on
    the compression stroke (piston moving from B toward A, ydot < 0). The
    book does not state this, but a force that depends on position only is
    conservative and could not give the nonzero mean driving torque of
    Fig. 12.4.2. The force pushes the piston toward the disk (+y).

    :param y: piston y position [m]
    :type y: float
    :param ydot: piston y velocity [m/s]
    :type ydot: float
    :param p_max_psi: maximum air pressure [psi] (0, 250, 500)
    :type p_max_psi: float
    :return: global y force on the piston [N]
    :rtype: float
    """
    if ydot >= 0.0 or p_max_psi == 0.0:
        return 0.0
    f_max = p_max_psi * PSI * math.pi * BORE * BORE / 4.0
    u_c = Y_C - Y_A
    u_b = Y_B - Y_A
    f_min = f_max * u_c / u_b
    f_c = f_max - f_min
    u = y - Y_A
    if u <= 0.0 or u >= u_b:
        return 0.0
    if u < u_c:
        return 0.5 * f_c * (1.0 - math.cos(math.pi * u / u_c))
    return f_max * u_c / u - f_min


def cycle_work(p_max_psi, n=20000):
    """Return the air pressure work on one piston per compression stroke.

    :param p_max_psi: maximum air pressure [psi]
    :type p_max_psi: float
    :param n: number of integration intervals
    :type n: int
    :return: work integral of Fig. 12.4.1 from A to B [J]
    :rtype: float
    """
    h = (Y_B - Y_A) / n
    total = 0.0
    for i in range(n):
        y = Y_A + (i + 0.5) * h
        total += air_force(y, -1.0, p_max_psi) * h
    return total


def inverse_dynamics(times, offset_deg=30.0, pressures=(0.0,),
                     rotor_speed=20.0 * math.pi):
    """Inverse dynamics with the rotor driven at constant speed (Sec. 12.4.2).

    The solver has no state-dependent force element, so each output time
    gets its own model. Step 1 builds the force-free model at the driver
    angle theta = rotor_speed * t (estimates: the previous output state) and
    calls ``init()``, which assembles the mechanism and solves the velocity
    equations. Step 2 builds the model again at that state with the piston
    forces of ``air_force`` as constant body forces and calls ``init()``,
    which solves Cq^T lambda = Q - M q'' (Haug Sec. 6.4). The reactions are
    exact for the prescribed motion; no time integration is involved.

    :param times: output times, increasing, starting near 0 [s]
    :type times: list
    :param offset_deg: offset angle [deg]
    :type offset_deg: float
    :param pressures: maximum air pressures [psi]
    :type pressures: tuple
    :param rotor_speed: rotor speed, 20 pi rad/s (600 rpm)
    :type rotor_speed: float
    :return: dict keyed by pressure; each value is a dict of lists: t,
        torque (driver effort, actuator torque on the rotor about +y [N m]),
        slot_force (global x reaction force on the disk at O [N]), and
        piston_y, piston_ydot (6 values per time)
    :rtype: dict
    """
    out = {p: {"t": [], "torque": [], "slot_force": [], "piston_y": [],
               "piston_ydot": []}
           for p in pressures}
    prev = None
    for t in times:
        theta0 = rotor_speed * t
        s0, parts0 = build(offset_deg=offset_deg, rotor_speed=rotor_speed,
                           theta0=theta0, initial=prev)
        s0.init()
        pos = as_list(s0.get_body_positions())
        quat = as_list(s0.get_body_quaternions())
        vel = as_list(s0.get_body_velocities())
        prev = (pos, quat)
        ys = [float(pos[3 + j][1]) for j in range(6)]
        yds = [float(vel[3 + j][1]) for j in range(6)]
        for p in pressures:
            forces = [air_force(y, yd, p) for y, yd in zip(ys, yds)]
            if any(f != 0.0 for f in forces):
                si, parts = build(offset_deg=offset_deg,
                                  rotor_speed=rotor_speed, theta0=theta0,
                                  piston_forces=forces, initial=(pos, quat))
                si.init()
            else:
                si, parts = s0, parts0
            rec = out[p]
            rec["t"].append(t)
            rec["torque"].append(float(reaction(si,
                                                parts["driver"])["effort"]))
            slot = reaction(si, parts["slot"])
            rec["slot_force"].append(float(slot["force_i"][0]))
            rec["piston_y"].append(ys)
            rec["piston_ydot"].append(yds)
    return out


def main():
    """Print the assembled configuration and the piston 1 stroke."""
    s, _ = build()
    s.init()
    names = ["Ground", "Rotor", "Disk"] + [f"Piston {k}" for k in range(1, 7)]
    for name, p, e in zip(names, as_list(s.get_body_positions()),
                          as_list(s.get_body_quaternions())):
        print(f"{name:9s} r = ({p[0]:+.5f}, {p[1]:+.5f}, {p[2]:+.5f})  "
              f"e = ({e[0]:+.5f}, {e[1]:+.5f}, {e[2]:+.5f}, {e[3]:+.5f})")
    for offset in (20.0, 30.0, 40.0):
        k = kinematics(offset_deg=offset, t_end=0.1)
        print(f"offset {offset:4.1f}: y in [{min(k['y']):.4f}, "
              f"{max(k['y']):.4f}], "
              f"ydot in [{min(k['ydot']):.3f}, {max(k['ydot']):.3f}], "
              f"yddot in [{min(k['yddot']):.1f}, {max(k['yddot']):.1f}]")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
