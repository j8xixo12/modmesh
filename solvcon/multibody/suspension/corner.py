# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Build one suspension corner in the Haug engine (``solvcon.multibody``).

Topology (roadmap P9.4, P15.1):

* upright (with wheel): held by 5 constraints, so it keeps 1 DOF (wheel
  travel) when the steer is held: distance links (1 each), a strut (2: the
  top mount stays on the strut axis of the upright), or a trailing arm (5:
  a revolute to the chassis);
* rack (when a link ends on it): prismatic to the chassis along the rack
  axis, positioned by a translational driver (steer input);
* rocker (when the model has one): revolute to the chassis, connected to
  the upright by the pushrod or pullrod (distance constraint), spring-damper
  from the rocker to the chassis; without a rocker the spring-damper acts
  from the upright to the chassis;
* wheel travel: absolute z driver on the wheel center (K&C rig input).

Upright and rocker frames are parallel to the global frame at design, with
their origins at the wheel center and the rocker pivot.  The bodies are
numbered from ``first_num`` in the order upright, rocker, rack, skipping
the parts the model does not have.
"""

import numpy as np

Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]


class Corner:
    """Handles of one built corner.

    :ivar upright: upright body
    :ivar rocker: rocker body, or None
    :ivar rack: rack body, or None
    :ivar travel: wheel-center z driver (None if the corner has no rig driver)
    :ivar steer: rack driver, or None without a rack
    :ivar spring: spring-damper element
    :ivar spring_body: ``'rocker'`` or ``'upright'``, the body of the spring
    :ivar spring_points: spring point on its body (body frame) and on the
        chassis (chassis frame)
    :ivar arms: arm name -> arm body (rigid wishbones)
    :ivar nums: body numbers of the present bodies {"upright", "rocker",
        "rack", "arm_<name>"}
    :ivar joints: ``(label, joint)`` of every joint, in the order added
    """

    def __init__(self):
        """Create empty handles; ``add_corner`` fills them."""
        self.upright = self.rocker = self.rack = None
        self.travel = self.steer = self.spring = None
        self.strut = self.arm = None
        self.arms = {}
        self.joints = []
        self.nums = {}
        self.link_joints = {}
        self.spring_body = "rocker"
        self.spring_points = None
        self.z0 = 0.0

    def body(self, name):
        """The engine body of a corner body name (``arm_<name>`` for an
        arm)."""
        if name.startswith("arm_"):
            return self.arms[name[4:]]
        return {"upright": self.upright, "rocker": self.rocker,
                "rack": self.rack}[name]

    def add_joint(self, sys_obj, label, joint):
        """Add ``joint`` to the system and keep it under ``label``."""
        sys_obj.Add(joint)
        self.joints.append((label, joint))
        return joint

    @property
    def nbody(self):
        """Number of engine bodies of the corner."""
        return len(self.nums)


def body_numbers(model, first_num):
    """Engine body numbers of the corner bodies of ``model``.

    :return: ``{"upright": n, "rocker": n + 1, "rack": ...}`` for the parts
        the model has
    :rtype: dict
    """
    nums = {"upright": first_num}
    if model.has_rocker:
        nums["rocker"] = first_num + len(nums)
    if model.has_rack:
        nums["rack"] = first_num + len(nums)
    for arm in sorted(model.arms):
        nums["arm_" + arm] = first_num + len(nums)
    return nums


def arm_frame(model, arm):
    """Origin (the midpoint of the inner pivots), unit pivot axis, and ball
    joint of an arm at design (global, m)."""
    front, rear, ball = model.arm_points(arm)
    axis = rear - front
    return 0.5 * (front + rear), axis / np.linalg.norm(axis), ball


def strut_frame(model):
    """Strut axis (unit, from the upright point to the top mount) at design
    with the adjusters applied, and the upright point (global, m)."""
    low = model.upright_points()["strut_outer"]
    top = model.chassis_point("strut_inner")
    axis = top - low
    return axis / np.linalg.norm(axis), low


def _load(gravity, mass):
    """The constant body force (N) of ``mass`` in the acceleration field
    ``gravity``: a scalar is g along -z, a 3-vector is the field."""
    if np.ndim(gravity) == 0:
        return [0.0, 0.0, -float(gravity) * mass]
    return [float(a) * mass for a in gravity]


def add_corner(mbs, sys_obj, model, ground, chassis, chassis_origin, first_num,
               travel_func=None, steer_func=None, gravity=0.0, velocity=None):
    """Add the bodies, joints, drivers, and spring of one corner.

    :param mbs: compiled solver module
    :type mbs: module
    :param sys_obj: system that already contains ground and chassis
    :type sys_obj: solvcon.multibody.DynamicsSystem
    :param model: corner model (SI)
    :type model: suspension.model.CornerModel
    :param ground: ground body
    :param chassis: chassis body (the ground for a K&C rig)
    :param chassis_origin: global position of the chassis body origin (m);
        the chassis frame is parallel to the global frame at design
    :type chassis_origin: numpy.ndarray
    :param first_num: body number of the upright; the rocker and the rack
        follow when the model has them
    :type first_num: int
    :param travel_func: drive function of the wheel-center z coordinate, or
        None for no travel driver (for example with a tire)
    :param steer_func: drive function of the rack travel (m); None fixes the
        rack at design; ignored without a rack
    :param gravity: gravitational acceleration along -z (m/s^2), or the
        acceleration field as a global 3-vector (gravity plus a D'Alembert
        load, for example (0, -a_y, -g) in a steady turn)
    :type gravity: float or sequence
    :param velocity: initial velocity (m/s, global) of every corner body,
        the velocity of a chassis in pure translation; None is at rest
    :return: handles
    :rtype: Corner
    """
    hp = model.hp
    up_pts = model.upright_points()
    wc = up_pts["wheel_center"]
    co = np.asarray(chassis_origin, float)
    vel = Z if velocity is None else [float(v) for v in velocity]

    def chassis_local(name):
        return list(model.chassis_point(name) - co)

    def upright_local(name):
        return list(up_pts[name] - wc)

    c = Corner()
    c.nums = body_numbers(model, first_num)
    c.upright = mbs.create_mobilized_body_quat(
        c.nums["upright"], list(wc), vel, Z, [1, 0, 0, 0], Z, Z,
        model.upright_mass, model.upright_inertia,
        _load(gravity, model.upright_mass), Z)
    sys_obj.Add(c.upright)
    if model.has_rocker:
        piv = hp["rocker_pivot"]
        c.rocker = mbs.create_mobilized_body_quat(
            c.nums["rocker"], list(piv), vel, Z, [1, 0, 0, 0], Z, Z,
            model.rocker_mass, model.rocker_inertia,
            _load(gravity, model.rocker_mass), Z)
        sys_obj.Add(c.rocker)
    rack_pos = None
    if model.has_rack:
        rack_pos = model.chassis_point("tie_rod_inner") \
            if "tie_rod_inner" in hp else _rack_origin(model)
        c.rack = mbs.create_mobilized_body_quat(
            c.nums["rack"], list(rack_pos), vel, Z, [1, 0, 0, 0], Z, Z,
            model.rack_mass, [1e-3, 1e-3, 1e-3],
            _load(gravity, model.rack_mass), Z)
        sys_obj.Add(c.rack)
        axis = list(model.rack_axis)
        c.add_joint(sys_obj, "rack prismatic",
                    mbs.create_joint(2, list(rack_pos - co), Z, Z, Z, axis,
                                     axis, chassis, c.rack))
        c.steer = c.add_joint(sys_obj, "rack driver (steer)",
                              mbs.create_translational_driver(
                                  body_i=chassis, body_j=c.rack,
                                  point_i=list(rack_pos - co), point_j=Z,
                                  axis_i=axis,
                                  func=steer_func
                                  or mbs.DriveFunction.parameter(0.0)))

    c.link_joints = {}
    in_arm = set()
    for arm in sorted(model.arms):
        # A rigid wishbone: revolute about its inner pivots, ball joint at
        # the upright.
        origin, axis, ball = arm_frame(model, arm)
        spec = model.arms[arm]
        body = mbs.create_mobilized_body_quat(
            c.nums["arm_" + arm], list(origin), vel, Z, [1, 0, 0, 0], Z, Z,
            spec["mass"], spec["inertia"],
            _load(gravity, spec["mass"]), Z)
        sys_obj.Add(body)
        c.add_joint(sys_obj, f"{arm} arm pivot",
                    mbs.create_joint(1, list(origin - co), Z, Z, Z,
                                     list(axis), list(axis), chassis, body))
        outer = model.links[spec["links"][0]][2]
        c.add_joint(sys_obj, f"{arm} ball joint",
                    mbs.create_joint(0, list(ball - origin),
                                     upright_local(outer), Z, Z, EZ, EZ,
                                     body, c.upright))
        c.arms[arm] = body
        in_arm.update(spec["links"])
    for name, (kind, inner, outer) in model.links.items():
        if name in in_arm:
            continue
        if kind == "rack":
            body_i, p_i = c.rack, list(model.chassis_point(inner) - rack_pos)
        else:
            body_i, p_i = chassis, chassis_local(inner)
        j = mbs.create_joint_distance(6, p_i, upright_local(outer), Z, Z, EZ,
                                      EZ, body_i, c.upright,
                                      model.link_length(name))
        c.add_joint(sys_obj, name, j)
        c.link_joints[name] = j

    if model.has_strut:
        # The top mount (on the chassis) stays on the strut axis of the
        # upright.
        axis, _ = strut_frame(model)
        c.strut = c.add_joint(sys_obj, "strut",
                              mbs.create_strut_joint(
                                  c.upright, chassis,
                                  upright_local("strut_outer"),
                                  chassis_local("strut_inner"), list(axis)))
    if model.has_arm:
        p_in = model.chassis_point("arm_inner")
        axis = model.chassis_point("arm_outer") - p_in
        axis = list(axis / np.linalg.norm(axis))
        c.arm = c.add_joint(sys_obj, "trailing arm pivot",
                            mbs.create_joint(1, list(p_in - co),
                                             list(p_in - wc), Z, Z, axis,
                                             axis, chassis, c.upright))

    spring_chassis = chassis_local("spring_chassis")
    if model.has_rocker:
        piv = hp["rocker_pivot"]
        ax = list(model.rocker_axis)
        c.add_joint(sys_obj, "rocker revolute",
                    mbs.create_joint(1, list(piv - co), Z, Z, Z, ax, ax,
                                     chassis, c.rocker))
        pr_len = float(np.linalg.norm(hp["pushrod_outer"]
                                      - hp["pushrod_inner"]))
        rod_body, rod_local = _on_body(model, c, "pushrod_outer", up_pts)
        c.add_joint(sys_obj, model.rod_kind,
                    mbs.create_joint_distance(
                        6, list(rod_local), list(hp["pushrod_inner"] - piv),
                        Z, Z, EZ, EZ, rod_body, c.rocker, pr_len))
        c.spring_body = "rocker"
        local = hp["spring_rocker"] - piv
        spring_on = c.rocker
        origin = piv
    else:
        c.spring_body = model.spring_body()
        spring_on, local = _on_body(model, c, "spring_upright", up_pts)
        origin = wc if "upright" == model.spring_on else \
            arm_frame(model, model.spring_on)[0]
    spring_len = float(np.linalg.norm(
        origin + local - model.chassis_point("spring_chassis")))
    free_length = spring_len + model.spring_free_offset
    if model.damper_curve is None:
        c.spring = mbs.create_spring_damper(spring_on, chassis, list(local),
                                            spring_chassis, model.spring_k,
                                            model.spring_c, free_length)
    else:
        # The spring and the damper act on one line: the length rate > 0
        # (rebound) is the curve's positive velocity.
        c.spring = mbs.create_curve_damper(
            spring_on, chassis, list(local), spring_chassis, model.spring_k,
            free_length, list(model.damper_curve[0]),
            list(model.damper_curve[1]))
    sys_obj.Add(c.spring)
    c.spring_points = (np.array(local),
                       np.array(model.chassis_point("spring_chassis") - co))

    c.travel = None
    if travel_func is not None:
        c.travel = c.add_joint(sys_obj, "wheel travel driver",
                               mbs.create_absolute_driver(
                                   body=c.upright, ground=ground, point=Z,
                                   direction=EZ, func=travel_func))
    return c


def _on_body(model, c, point, up_pts):
    """Body and body-frame position of ``point``: on its arm or on the
    upright."""
    arm = model.arm_of(point)
    if arm is None:
        return c.upright, up_pts[point] - up_pts["wheel_center"]
    return c.arms[arm], model.hp[point] - arm_frame(model, arm)[0]


def _rack_origin(model):
    """Rack body origin: the inner point of the first link on the rack."""
    for kind, inner, _ in model.links.values():
        if "rack" == kind:
            return model.chassis_point(inner)
    raise ValueError("the model has no link on the rack")


def kc_rig(mbs, model, travel=0.0, steer=0.0):
    """Build a K&C rig system: chassis = ground, parameter drivers for travel
    and steer.

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model
    :type model: suspension.model.CornerModel
    :param travel: initial wheel travel from design (m, + = bump)
    :type travel: float
    :param steer: initial rack travel (m)
    :type steer: float
    :return: (system, corner handles)
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    s.Add(g)
    z0 = float(model.hp["wheel_center"][2])
    c = add_corner(mbs, s, model, g, g, np.zeros(3), 1,
                   travel_func=mbs.DriveFunction.parameter(z0 + travel),
                   steer_func=mbs.DriveFunction.parameter(steer))
    c.z0 = z0
    s.init()
    return s, c

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
