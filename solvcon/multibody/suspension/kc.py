# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Kinematics and compliance sweeps of one corner on a K&C rig (roadmap P9.2,
P9.5)."""

import numpy as np

from ..tools.arrays import result_arrays
from . import corner as corner_mod
from . import metrics as mt


def _driver_col(res, sys_obj, driver):
    joints = sys_obj.get_joints()
    for col, jidx in enumerate(res["driver_joints"]):
        if joints[jidx] is driver:
            return col
    raise ValueError("driver not found in sweep result")


def bump_sweep(mbs, model, travels, steer=0.0):
    """Wheel travel sweep with all K&C metrics per point.

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model
    :type model: suspension.model.CornerModel
    :param travels: wheel travel values from design (m, + = bump)
    :type travels: list
    :param steer: rack travel (m)
    :type steer: float
    :return: dict of numpy arrays (SI) plus "completed" and "message".
        Besides the scalar metrics, (n, 3) arrays in m (global):
        "contact_patch"; "fvic", the front-view instant center in the plane
        x = contact patch x; "svic", the side-view instant center in the
        plane y = contact patch y; "roll_center_point", the contact patch to
        FVIC line at y = 0; "kingpin_point", the steering axis ground point;
        "kingpin_dir", its upward unit direction; "patch_rate" and
        "center_rate", the velocity per unit travel of the contact patch and
        of the wheel center. An instant center at infinity is a NaN row.
    :rtype: dict
    """
    s, c = corner_mod.kc_rig(mbs, model, travel=float(travels[0]), steer=steer)
    up, rk = c.nums["upright"], c.nums[c.spring_body]
    res = result_arrays(s.sweep(c.travel, [c.z0 + t for t in travels]))
    n = len(res["values"])
    out = {k: np.zeros(n) for k in ("travel", "camber", "toe", "caster", "kpi",
                                    "scrub", "trail", "motion_ratio",
                                    "roll_center", "anti_dive", "wheel_rate",
                                    "spring_length", "rcond")}
    for key in ("contact_patch", "fvic", "svic", "roll_center_point",
                "kingpin_point", "kingpin_dir", "patch_rate",
                "center_rate"):
        out[key] = np.zeros((n, 3))
    col = _driver_col(res, s, c.travel)
    veh = model.vehicle
    spin_local = model.spin_axis
    axis_points = model.steering_axis_points()
    pts = model.upright_points()
    if axis_points is not None:
        out["kingpin_method"] = ("strut" if axis_points[1][0] == "chassis"
                                 else "ball_joints")
    elif c.steer is not None:
        out["kingpin_method"] = "instant_axis"
    else:
        # No rack and no geometric axis: the steering-axis metrics do not
        # exist (NaN).
        out["kingpin_method"] = "none"
    for k in range(n):
        s.sweep(c.travel, [res["values"][k]])  # put the rig at pose k
        r_up = res["positions"][k, up]
        a_up = mt.rot(res["quaternions"][k, up])
        wg = mt.wheel_geometry(r_up, a_up, spin_local, model.wheel_radius,
                               model.side)
        cp = wg["contact_patch"]
        _, v_cp = mt.point_rate(r_up, a_up, a_up.T @ (cp - r_up),
                                res["dr_ds"][k, up], res["domega_ds"][k, up])
        # The rates per unit travel of the contact patch (an upright-fixed
        # point) and of the wheel center: the lever arms of a tire force
        # in the virtual work (outboard brakes and inboard drive).
        out["patch_rate"][k] = v_cp
        out["center_rate"][k] = res["dr_ds"][k, up]
        r_rk = res["positions"][k, rk]
        a_rk = mt.rot(res["quaternions"][k, rk])
        p_sp, v_sp = mt.point_rate(r_rk, a_rk, c.spring_points[0],
                                   res["dr_ds"][k, rk],
                                   res["domega_ds"][k, rk])
        d = p_sp - c.spring_points[1]
        length = float(np.linalg.norm(d))
        out["travel"][k] = res["values"][k] - c.z0
        out["camber"][k] = wg["camber"]
        out["toe"][k] = wg["toe"]
        out["contact_patch"][k] = cp
        out["spring_length"][k] = length
        out["motion_ratio"][k] = -float(d @ v_sp) / length
        out["roll_center"][k] = mt.roll_center_height(cp, v_cp)
        w_up = res["domega_ds"][k, up]
        out["fvic"][k] = mt.front_view_ic(cp, v_cp, w_up)
        out["svic"][k] = mt.side_view_ic(cp, v_cp, w_up)
        out["roll_center_point"][k] = mt.roll_center_point(cp, v_cp,
                                                           out["fvic"][k])
        if veh:
            out["anti_dive"][k] = mt.anti_dive(v_cp, veh["wheelbase"],
                                               veh["cg_height"],
                                               veh["front_brake_fraction"])[1]
        out["rcond"][k] = res["rcond"][k]
        if axis_points is not None:
            # Double wishbone: the line through the ball joints at this
            # pose; strut: the lower ball joint to the top mount.
            ends = []
            for frame, name in axis_points:
                if "upright" == frame:
                    ends.append(r_up + a_up @ (pts[name]
                                               - pts["wheel_center"]))
                else:
                    ends.append(model.chassis_point(name))
            n_ax, p_ax = mt.ball_joint_axis(ends[0], ends[1])
        elif c.steer is not None:
            # Virtual steering axis: screw axis of the upright for a rack
            # motion.
            st = result_arrays(s.sweep(c.steer, [steer]))
            n_ax, p_ax = mt.screw_axis(r_up, st["dr_ds"][0, up],
                                       st["domega_ds"][0, up])
        else:
            for key in ("caster", "kpi", "scrub", "trail"):
                out[key][k] = np.nan
            out["kingpin_point"][k] = np.nan
            out["kingpin_dir"][k] = np.nan
            continue
        sa = mt.steering_axis_metrics(n_ax, p_ax, cp, model.side)
        for key in ("caster", "kpi", "scrub", "trail"):
            out[key][k] = sa[key]
        # Kingpin axis for the 3D view: the ground point (z = contact patch z)
        # and the upward unit direction.
        out["kingpin_point"][k] = sa["ground_point"]
        out["kingpin_dir"][k] = n_ax if n_ax[2] >= 0 else -n_ax
    # Wheel rate: d(vertical force at the wheel center)/d(travel) from driver
    # efforts.
    eff = res["efforts"][:, col]
    if n > 1:
        out["wheel_rate"] = np.gradient(eff, out["travel"][:n])
    out["completed"] = bool(res["completed"])
    out["message"] = res["message"]
    return out


def steer_sweep(mbs, model, racks, travel=0.0):
    """Rack travel sweep: road wheel angle (toe), camber, and steering effort.

    :return: dict of numpy arrays plus "completed" and "message"
    :rtype: dict
    """
    if not model.has_rack:
        raise ValueError(f"model '{model.name}' has no steering rack")
    s, c = corner_mod.kc_rig(mbs, model, travel=travel, steer=float(racks[0]))
    up = c.nums["upright"]
    res = result_arrays(s.sweep(c.steer, list(racks)))
    n = len(res["values"])
    out = {"rack": np.array(res["values"]), "toe": np.zeros(n),
           "camber": np.zeros(n)}
    for k in range(n):
        wg = mt.wheel_geometry(res["positions"][k, up],
                               mt.rot(res["quaternions"][k, up]),
                               model.spin_axis, model.wheel_radius, model.side)
        out["toe"][k] = wg["toe"]
        out["camber"][k] = wg["camber"]
    out["completed"] = bool(res["completed"])
    out["message"] = res["message"]
    return out

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
