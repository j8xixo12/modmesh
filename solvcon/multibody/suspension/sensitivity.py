# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Hardpoint batch sweep: K&C summaries against one coordinate (roadmap
P11.9)."""

import copy

import numpy as np

from . import kc

SUMMARY_KEYS = ("camber_gain", "bump_steer", "roll_center", "caster", "kpi",
                "scrub", "trail", "motion_ratio")


def design_summary(mbs, model, h=0.005):
    """K&C summary at design from a bump sweep at [-h, 0, +h].

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model
    :type model: suspension.model.CornerModel
    :param h: travel step of the central differences (m)
    :type h: float
    :return: (summary dict of floats, sweep result); camber_gain and
        bump_steer in rad/m, roll_center, scrub, trail in m, caster, kpi in
        rad, motion_ratio without unit; the summary is None if the sweep
        failed
    :rtype: tuple
    """
    try:
        r = kc.bump_sweep(mbs, model, [-h, 0.0, h])
    except RuntimeError as e:  # the rig does not assemble at design
        return None, {"completed": False, "message": str(e)}
    if not r["completed"]:
        return None, r
    out = {"camber_gain": (r["camber"][2] - r["camber"][0]) / (2 * h),
           "bump_steer": (r["toe"][2] - r["toe"][0]) / (2 * h)}
    for key in SUMMARY_KEYS[2:]:
        out[key] = r[key][1]
    return out, r


def hardpoint_sweep(mbs, model, point, axis, offsets, h=0.005):
    """Sweep one hardpoint coordinate and record the K&C summary at design.

    For each offset, a copy of the model gets the point moved by the offset
    along the axis (``CornerModel.set_hardpoint``, so the limit box applies),
    then a bump sweep at [-h, 0, +h] gives the summary. The sweep stops at
    the first offset outside the limit box or with a failed bump sweep. The
    input model is not modified.

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model
    :type model: suspension.model.CornerModel
    :param point: hardpoint name
    :type point: str
    :param axis: coordinate index, 0 = x, 1 = y, 2 = z
    :type axis: int
    :param offsets: offsets from the current position (m)
    :type offsets: list
    :param h: travel step of the central differences (m)
    :type h: float
    :return: dict of numpy arrays keyed by "offset" and the summary names
        (see ``design_summary``), plus "completed" and "message"
    :rtype: dict
    :raises ValueError: for an unknown point or an axis not in 0, 1, 2
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2 (x, y, z), got {axis!r}")
    if point not in model.hp:
        raise ValueError(f"unknown hardpoint '{point}'")
    base = model.hp[point].copy()
    rows = {key: [] for key in ("offset",) + SUMMARY_KEYS}
    completed, message = True, ""
    for d in offsets:
        m = copy.deepcopy(model)
        p = base.copy()
        p[axis] += d
        try:
            m.set_hardpoint(point, p)
        except ValueError as e:
            completed, message = False, f"stopped at offset {d:.6g} m: {e}"
            break
        summary, r = design_summary(mbs, m, h)
        if summary is None:
            completed = False
            message = (f"stopped at offset {d:.6g} m: bump sweep failed: "
                       f"{r['message']}")
            break
        rows["offset"].append(float(d))
        for key in SUMMARY_KEYS:
            rows[key].append(summary[key])
    out = {key: np.array(v, float) for key, v in rows.items()}
    out["completed"] = completed
    out["message"] = message
    return out

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
