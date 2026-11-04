# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Compliance K&C: links as bodies with inboard bushings (roadmap P9.8).

Each link is a body with a spherical joint at the upright and a linear 6x6
bushing to the chassis (or rack) at the inboard pickup.  A trailing arm sits
on two bushings at its pivots.  The rocker, pushrod, strut, and spring are
rigid as in suspension.corner. Wheel-center height and rack
travel are held by parameter drivers; loads are applied at the contact patch
and the static equilibrium gives the compliance response.
"""

import numpy as np

from ..tools.arrays import as_list
from ..tools.elements import bushing_matrix

from . import corner as corner_mod
from . import metrics as mt

Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]


def compliant_rig(mbs, model, k_trans=None, k_rot=None, link_mass=0.3):
    """Build the compliant K&C rig and solve the unloaded equilibrium.

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model
    :type model: suspension.model.CornerModel
    :param k_trans: bushing translational rate (N/m); default from the model
    :type k_trans: float or None
    :param k_rot: bushing rotational rate (N m/rad); default from the model
    :type k_rot: float or None
    :param link_mass: mass of each link body (kg)
    :type link_mass: float
    :return: (system, handles dict)
    :rtype: tuple
    """
    kt = model.bushing_k if k_trans is None else k_trans
    kr = model.bushing_k_rot if k_rot is None else k_rot
    hp = model.hp
    up_pts = model.upright_points()
    wc = up_pts["wheel_center"]
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    s.Add(g)
    nums = corner_mod.body_numbers(model, 1)
    upright = mbs.create_mobilized_body_quat(nums["upright"], list(wc), Z, Z,
                                             [1, 0, 0, 0], Z, Z,
                                             model.upright_mass,
                                             model.upright_inertia, Z, Z)
    s.Add(upright)
    rocker = rack = steer = None
    if model.has_rocker:
        piv = hp["rocker_pivot"]
        rocker = mbs.create_mobilized_body_quat(nums["rocker"], list(piv), Z,
                                                Z, [1, 0, 0, 0], Z, Z,
                                                model.rocker_mass,
                                                model.rocker_inertia, Z, Z)
        s.Add(rocker)
    if model.has_rack:
        rack_pos = corner_mod._rack_origin(model) \
            if "tie_rod_inner" not in hp else model.chassis_point(
                "tie_rod_inner")
        rack = mbs.create_mobilized_body_quat(nums["rack"], list(rack_pos), Z,
                                              Z, [1, 0, 0, 0], Z, Z,
                                              model.rack_mass,
                                              [1e-3, 1e-3, 1e-3], Z, Z)
        s.Add(rack)
        axis = list(model.rack_axis)
        s.Add(mbs.create_joint(2, list(rack_pos), Z, Z, Z, axis, axis, g,
                               rack))
        steer = mbs.create_translational_driver(
            body_i=g, body_j=rack, point_i=list(rack_pos), point_j=Z,
            axis_i=axis, func=mbs.DriveFunction.parameter(0.0))
        s.Add(steer)
    num = len(nums) + 1
    k6 = bushing_matrix([kt] * 3, [kr] * 3)
    c6 = bushing_matrix(Z, Z)

    def on_body(point):
        arm = model.arm_of(point)
        if arm is None:
            return upright, up_pts[point] - wc, wc
        body, origin = arms[arm]
        return body, hp[point] - origin, origin

    links = {}
    arms = {}
    in_arm = set()
    for arm in sorted(model.arms):
        # A rigid wishbone on two inner bushings, ball joint at the upright.
        origin, _, ball = corner_mod.arm_frame(model, arm)
        spec = model.arms[arm]
        body = mbs.create_mobilized_body_quat(nums["arm_" + arm], list(origin),
                                              Z, Z,
                                              [1, 0, 0, 0], Z, Z,
                                              spec["mass"], spec["inertia"],
                                              Z, Z)
        s.Add(body)
        for link_name in spec["links"]:
            p = model.chassis_point(model.links[link_name][1])
            s.Add(mbs.create_bushing(g, body, list(p), list(p - origin), k6,
                                     c6))
        outer = model.links[spec["links"][0]][2]
        s.Add(mbs.create_joint(0, list(ball - origin), list(up_pts[outer]
                                                            - wc),
                               Z, Z, EZ, EZ, body, upright))
        arms[arm] = (body, origin)
        in_arm.update(spec["links"])
    for name, (kind, inner, outer) in model.links.items():
        if name in in_arm:
            continue
        p_in = model.chassis_point(inner)
        # The link body has the (adjusted) link length. Its outer end may miss
        # the upright point at construction; assembly then moves the upright.
        p_up = up_pts[outer]
        p_out = (p_in + (p_up - p_in) * (model.link_length(name)
                                         / np.linalg.norm(p_up - p_in)))
        mid = 0.5 * (p_in + p_out)
        link = mbs.create_mobilized_body_quat(num, list(mid), Z, Z,
                                              [1, 0, 0, 0], Z, Z, link_mass,
                                              [1e-4, 1e-4, 1e-4], Z, Z)
        s.Add(link)
        s.Add(mbs.create_joint(0, list(p_out - mid), list(p_up - wc), Z, Z, EZ,
                               EZ, link, upright))
        if kind == "rack":
            s.Add(mbs.create_bushing(rack, link, list(p_in - rack_pos),
                                     list(p_in - mid), k6, c6))
        else:
            s.Add(mbs.create_bushing(g, link, list(p_in),
                                     list(p_in - mid), k6, c6))
        links[name] = (num, link)
        num += 1
    if model.has_strut:
        # The strut slides in its guide and the top mount is a rigid pivot
        # here; the strut tube carries the side load of the wheel.
        axis, _ = corner_mod.strut_frame(model)
        s.Add(mbs.create_strut_joint(upright, g,
                                     list(up_pts["strut_outer"] - wc),
                                     list(model.chassis_point("strut_inner")),
                                     list(axis)))
    if model.has_arm:
        # The arm sits on its two pivot bushings: they let it turn about the
        # pivot axis (soft in torsion) and give under the wheel loads.
        for name in ("arm_inner", "arm_outer"):
            p = model.chassis_point(name)
            s.Add(mbs.create_bushing(g, upright, list(p), list(p - wc), k6,
                                     c6))
    if model.has_rocker:
        piv = hp["rocker_pivot"]
        ax = list(model.rocker_axis)
        s.Add(mbs.create_joint(1, list(piv), Z, Z, Z, ax, ax, g, rocker))
        pr_len = float(np.linalg.norm(hp["pushrod_outer"]
                                      - hp["pushrod_inner"]))
        rod_body, rod_local, _ = on_body("pushrod_outer")
        s.Add(mbs.create_joint_distance(6, list(rod_local),
                                        list(hp["pushrod_inner"] - piv), Z, Z,
                                        EZ, EZ, rod_body, rocker, pr_len))
        spring_on, local, origin = rocker, hp["spring_rocker"] - piv, piv
    else:
        spring_on, local, origin = on_body("spring_upright")
    p_chassis = model.chassis_point("spring_chassis")
    spring_len = float(np.linalg.norm(origin + local - p_chassis))
    s.Add(mbs.create_spring_damper(spring_on, g, list(local), list(p_chassis),
                                   model.spring_k, 0.0,
                                   spring_len + model.spring_free_offset))
    travel = mbs.create_absolute_driver(body=upright, ground=g, point=Z,
                                        direction=EZ,
                                        func=mbs.DriveFunction.parameter(
                                            float(wc[2])))
    s.Add(travel)
    s.init()
    res = s.find_equilibrium(tol=1e-9)
    if not res["converged"]:
        raise RuntimeError("compliant rig: unloaded equilibrium failed: "
                           f"{res['message']}")
    return s, {"upright": 1, "links": links, "travel": travel, "steer": steer}


def apply_patch_load(mbs, s, model, force):
    """Apply a force at the contact patch, solve equilibrium, return alignment.

    :param force: global force at the contact patch (N)
    :type force: list
    :return: dict with toe, camber (rad) and contact_patch (m)
    :rtype: dict
    """
    up = 1
    r = np.array(as_list(s.get_body_positions())[up])
    a = mt.rot(as_list(s.get_body_quaternions())[up])
    cp = mt.wheel_geometry(r, a, model.spin_axis, model.wheel_radius,
                           model.side)["contact_patch"]
    f = np.asarray(force, float)
    s.set_applied_load(up, list(f), list(np.cross(cp - r, f)))
    res = s.find_equilibrium(tol=1e-9)
    if not res["converged"]:
        raise RuntimeError(f"compliance load case failed: {res['message']}")
    r = np.array(as_list(s.get_body_positions())[up])
    a = mt.rot(as_list(s.get_body_quaternions())[up])
    wg = mt.wheel_geometry(r, a, model.spin_axis, model.wheel_radius,
                           model.side)
    # Material contact point (fixed on the upright) for displacement measures.
    return {"toe": wg["toe"], "camber": wg["camber"],
            "contact_patch": wg["contact_patch"], "r": r, "a": a}


CASES = ("lateral", "braking", "traction", "aligning")


def _load_of(case, value, cp, r):
    """Global force and torque (about the upright origin) of one load case."""
    if case == "lateral":
        f = np.array([0.0, value, 0.0])
        return f, np.cross(cp - r, f)
    if case == "braking":
        f = np.array([value, 0.0, 0.0])
        return f, np.cross(cp - r, f)
    if case == "traction":  # at the wheel center, the upright origin
        return np.array([value, 0.0, 0.0]), np.zeros(3)
    # aligning: a pure couple about z is the same about any point, so also
    # about the vertical through the contact patch.
    return np.zeros(3), np.array([0.0, 0.0, value])


def compliance_sweep(mbs, model, case, fmax, n=11, k_trans=None, k_rot=None):
    """Ramp one load from -fmax to +fmax on the compliant rig (roadmap P11.7).

    The load points are fixed at the unloaded pose: the contact patch for
    "lateral" (Fy) and "braking" (Fx), the wheel center for "traction" (Fx);
    "aligning" is a torque Mz (fmax in N m). The ramp runs from zero load
    down to -fmax, then from zero load up to +fmax, each step starting from
    the previous equilibrium. It stops at the first failed equilibrium.

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model
    :type model: suspension.model.CornerModel
    :param case: "lateral", "braking", "traction", or "aligning"
    :type case: str
    :param fmax: largest load magnitude (N, or N m for "aligning")
    :type fmax: float
    :param n: number of load points, odd so that zero load is a point
    :type n: int
    :param k_trans: bushing translational rate (N/m); default from the model
    :type k_trans: float or None
    :param k_rot: bushing rotational rate (N m/rad); default from the model
    :type k_rot: float or None
    :return: dict with "load" (N or N m), "toe" and "camber" (rad, change
        from zero load), "patch" ((n, 3) displacement of the contact patch
        material point, m), "rotation" ((n, 3) rotation vector of the
        upright from zero load, global, rad), "gradient" ({toe, camber,
        patch_x, patch_y, patch_z, rot_x, rot_y, rot_z} at zero load by
        central difference of the two neighbor points, per N or per N m;
        NaN if a neighbor is missing), "case",
        "completed", and "message". After a failure the arrays hold the
        solved points in ascending load order.
    :rtype: dict
    :raises ValueError: for an unknown case, an even or too small n, or a
        non-positive fmax
    """
    if case not in CASES:
        raise ValueError(f"unknown compliance case '{case}'; known: "
                         f"{', '.join(CASES)}")
    if n < 3 or n % 2 == 0:
        raise ValueError("n must be odd and at least 3 (zero load is a "
                         f"point), got {n}")
    if not fmax > 0:
        raise ValueError(f"fmax must be positive, got {fmax}")
    loads = np.linspace(-fmax, fmax, n)
    mid = n // 2
    loads[mid] = 0.0
    up = 1
    points = {}
    completed, message = True, ""
    cp_local = None
    for branch in (range(mid, -1, -1), range(mid, n)):
        try:
            s, _ = compliant_rig(mbs, model, k_trans, k_rot)
        except RuntimeError as e:  # the unloaded rig has no equilibrium
            completed, message = False, str(e)
            break
        r0 = np.array(as_list(s.get_body_positions())[up])
        a0 = mt.rot(as_list(s.get_body_quaternions())[up])
        wg0 = mt.wheel_geometry(r0, a0, model.spin_axis, model.wheel_radius,
                                model.side)
        cp0 = wg0["contact_patch"]
        if cp_local is None:
            cp_local = a0.T @ (cp0 - r0)
            base = (wg0["toe"], wg0["camber"], cp0, a0)
        for k in branch:
            if k == mid:
                points[k] = (0.0, 0.0, np.zeros(3), np.zeros(3))
                continue
            f, t = _load_of(case, loads[k], cp0, r0)
            s.set_applied_load(up, list(f), list(t))
            res = s.find_equilibrium(tol=1e-9)
            if not res["converged"]:
                completed = False
                message = (f"equilibrium failed at load {loads[k]:.6g}: "
                           f"{res['message']}")
                break
            r = np.array(as_list(s.get_body_positions())[up])
            a = mt.rot(as_list(s.get_body_quaternions())[up])
            wg = mt.wheel_geometry(r, a, model.spin_axis, model.wheel_radius,
                                   model.side)
            points[k] = (wg["toe"] - base[0], wg["camber"] - base[1],
                         r + a @ cp_local - base[2],
                         mt.rotation_vector(a @ base[3].T))
        if not completed:
            break
    ks = sorted(points)
    out = {"case": case, "load": loads[ks],
           "toe": np.array([points[k][0] for k in ks]),
           "camber": np.array([points[k][1] for k in ks]),
           "patch": np.array([points[k][2] for k in ks]).reshape(-1, 3),
           "rotation": np.array([points[k][3] for k in ks]).reshape(-1, 3)}
    grad = {key: float("nan") for key in ("toe", "camber", "patch_x",
                                          "patch_y", "patch_z", "rot_x",
                                          "rot_y", "rot_z")}
    if mid - 1 in points and mid + 1 in points:
        lo, hi = points[mid - 1], points[mid + 1]
        dl = loads[mid + 1] - loads[mid - 1]
        grad["toe"] = (hi[0] - lo[0]) / dl
        grad["camber"] = (hi[1] - lo[1]) / dl
        for i, axis in enumerate("xyz"):
            grad["patch_" + axis] = (hi[2][i] - lo[2][i]) / dl
            grad["rot_" + axis] = (hi[3][i] - lo[3][i]) / dl
    out["gradient"] = grad
    out["completed"] = completed
    out["message"] = message
    return out

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
