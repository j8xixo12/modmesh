# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Front axle on a K&C rig: two corners, rack, and anti-roll bar (roadmap
P9.6).

The anti-roll bar is two half-bars on coaxial revolute joints to the chassis
with an RSDA between them (bar torsion rate). Vertical drop links connect the
bar arm tips to the rockers. Roll is applied as opposite wheel travel
z_L = z0 + s, z_R = z0 - s (ground plates), roll angle atan(2 s / track).
"""

import csv
import math

import numpy as np

from ..tools.arrays import result_arrays
from . import corner as corner_mod
from . import metrics as mt

Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]


def _add_arb_half(mbs, s, model, chassis, corner, num):
    piv = model.hp["arb_pivot"]
    half = mbs.create_mobilized_body_quat(num, list(piv), Z, Z, [1, 0, 0, 0],
                                          Z, Z, model.arb["mass"],
                                          model.arb["inertia"], Z, Z)
    s.Add(half)
    ax = list(model.arb["axis"])
    s.Add(mbs.create_joint(1, list(piv), Z, Z, Z, ax, ax, chassis, half))
    tip_local = list(model.hp["arb_tip"] - piv)
    rocker_local = list(model.hp["arb_rocker"] - model.hp["rocker_pivot"])
    length = float(np.linalg.norm(model.hp["arb_tip"]
                                  - model.hp["arb_rocker"]))
    s.Add(mbs.create_joint_distance(6, rocker_local, tip_local, Z, Z, EZ, EZ,
                                    corner.rocker, half, length))
    return half


def kc_axle(mbs, model_left, with_arb=True):
    """Build the axle rig with parameter drivers on both wheel centers.

    :param mbs: compiled solver module
    :type mbs: module
    :param model_left: left corner model (the right corner is its mirror)
    :type model_left: suspension.model.CornerModel
    :param with_arb: include the anti-roll bar
    :type with_arb: bool
    :return: (system, dict with corners "left", "right", "arb" halves, and z0)
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    s.Add(g)
    z0 = float(model_left.hp["wheel_center"][2])
    models = {"left": model_left, "right": model_left.mirrored()}
    out = {"z0": z0, "models": models}
    num = 1
    for side in ("left", "right"):
        c = corner_mod.add_corner(mbs, s, models[side], g, g, np.zeros(3), num,
                                  travel_func=mbs.DriveFunction.parameter(z0))
        c.z0 = z0
        out[side] = c
        num += c.nbody
    if with_arb:
        if model_left.arb is None:
            raise ValueError("model has no 'arb' block")
        halves = [_add_arb_half(mbs, s, models[side], g, out[side], num + k)
                  for k, side in enumerate(("left", "right"))]
        ax = list(model_left.arb["axis"])
        s.Add(mbs.create_rsda(halves[0], halves[1], ax, ax,
                              k=model_left.arb["k"]))
        out["arb"] = halves
        out["arb_nums"] = (num, num + 1)
    s.init()
    return s, out


def roll_sweep(mbs, model_left, travels, with_arb=True):
    """Opposite wheel travel sweep: roll moment and roll stiffness.

    :param travels: values of s (m); z_L = z0 + s, z_R = z0 - s
    :type travels: list
    :return: dict with s, roll angle (rad), roll moment (N m), roll stiffness
        (N m/rad), generalized stiffness dQ/ds (N/m), and the raw sweep
    :rtype: dict
    """
    s, ax = kc_axle(mbs, model_left, with_arb)
    z0 = ax["z0"]
    half_track = abs(float(model_left.hp["wheel_center"][1]))
    drivers = [ax["left"].travel, ax["right"].travel]
    res = result_arrays(s.sweep(drivers, [[z0 + t, z0 - t] for t in travels],
                                [1.0, -1.0]))
    joints = s.get_joints()
    cols = [next(k for k, j in enumerate(res["driver_joints"])
                 if joints[j] is d)
            for d in drivers]
    sv = np.array(res["values"]) - z0
    f_l, f_r = res["efforts"][:, cols[0]], res["efforts"][:, cols[1]]
    q_s = f_l - f_r  # generalized force along s (virtual work)
    roll = np.arctan2(2 * sv, 2 * half_track)
    moment = half_track * (f_l - f_r)
    out = {"s": sv, "roll": roll, "moment": moment, "q_s": q_s, "raw": res,
           "completed": bool(res["completed"]), "message": res["message"],
           "axle": ax}
    if len(sv) > 1:
        out["dq_ds"] = np.gradient(q_s, sv)
        out["roll_stiffness"] = np.gradient(moment, roll)
    return out


PROFILE_COLUMNS = {"heave_mm": 1e-3, "roll_deg": math.pi / 180.0,
                   "rack_mm": 1e-3}


def profile_sweep(mbs, model_left, rows, with_arb=True):
    """Drive the axle rig through a table of (heave, roll, rack) rows (roadmap
    P11.11).

    Row k sets the wheel-center heights z_L = z0 + heave + s and
    z_R = z0 + heave - s with s = half_track * tan(roll), the same roll
    definition as ``roll_sweep`` (roll > 0: left wheel in bump). The rack is
    one common rack: both inner tie rod points move by ``rack`` along the
    left rack axis (+y), so the right corner rack driver, whose axis is the
    mirror (-y), gets -rack. Each row starts from the previous pose; the
    sweep stops at the first row that does not assemble.

    :param mbs: compiled solver module
    :type mbs: module
    :param model_left: left corner model (the right corner is its mirror)
    :type model_left: suspension.model.CornerModel
    :param rows: (heave m, roll rad, rack m) per row
    :type rows: list
    :param with_arb: include the anti-roll bar
    :type with_arb: bool
    :return: dict of numpy arrays over the solved rows: "heave", "roll",
        "rack", "travel_left", "travel_right" (m), "camber_left",
        "toe_left", "camber_right", "toe_right" (rad), "contact_patch_left",
        "contact_patch_right" ((n, 3), m), "positions" ((n, nbody, 3), m),
        "quaternions" ((n, nbody, 4)), "efforts" ((n, ndriver), driver
        efforts), plus "bodies" ({left, right} body numbers of the
        uprights), "completed", and "message"
    :rtype: dict
    """
    s, ax = kc_axle(mbs, model_left, with_arb)
    z0 = ax["z0"]
    half_track = abs(float(model_left.hp["wheel_center"][1]))
    rows = [tuple(float(v) for v in row) for row in rows]
    for row in rows:
        if len(row) != 3:
            raise ValueError("profile row needs (heave, roll, rack), got "
                             f"{row!r}")
    steered = model_left.has_rack
    if not steered and any(rack != 0.0 for _, _, rack in rows):
        raise ValueError(f"model '{model_left.name}' has no steering rack")
    drivers = [ax["left"].travel, ax["right"].travel]
    if steered:
        drivers += [ax["left"].steer, ax["right"].steer]
    values = []
    for heave, roll, rack in rows:
        sr = half_track * math.tan(roll)
        values.append([z0 + heave + sr, z0 + heave - sr]
                      + ([rack, -rack] if steered else []))
    res = result_arrays(s.sweep(drivers, values)) if values else None
    n = len(res["values"]) if res is not None else 0
    out = {"heave": np.array([r[0] for r in rows[:n]]),
           "roll": np.array([r[1] for r in rows[:n]]),
           "rack": np.array([r[2] for r in rows[:n]]),
           "travel_left": np.array([v[0] - z0 for v in values[:n]]),
           "travel_right": np.array([v[1] - z0 for v in values[:n]])}
    nums = {side: ax[side].nums["upright"] for side in ("left", "right")}
    for side in ("left", "right"):
        m = ax["models"][side]
        cam, toe, cps = np.zeros(n), np.zeros(n), np.zeros((n, 3))
        for k in range(n):
            wg = mt.wheel_geometry(res["positions"][k, nums[side]],
                                   mt.rot(res["quaternions"][k, nums[side]]),
                                   m.spin_axis, m.wheel_radius, m.side)
            cam[k], toe[k], cps[k] = (wg["camber"], wg["toe"],
                                      wg["contact_patch"])
        out["camber_" + side] = cam
        out["toe_" + side] = toe
        out["contact_patch_" + side] = cps
    if res is not None:
        out["positions"] = np.asarray(res["positions"])
        out["quaternions"] = np.asarray(res["quaternions"])
        out["efforts"] = np.asarray(res["efforts"])
        out["completed"] = bool(res["completed"])
        out["message"] = res["message"]
    else:
        out["completed"], out["message"] = True, "no rows"
    out["bodies"] = nums
    return out


def read_profile_csv(path):
    """Read a motion profile CSV with a header row.

    Known columns (any order, any subset; a missing column is 0):
    ``heave_mm``, ``roll_deg``, ``rack_mm``. Empty lines are skipped.

    :param path: CSV file
    :type path: str or pathlib.Path
    :return: rows of (heave m, roll rad, rack m)
    :rtype: list
    :raises ValueError: for an unknown or repeated column, no known column,
        or a value that is not a number
    """
    with open(path, newline="") as f:
        lines = list(csv.reader(f))
    header = [h.strip() for h in lines[0]] if lines else []
    unknown = [h for h in header if h not in PROFILE_COLUMNS]
    if unknown or not header:
        raise ValueError(f"{path}: unknown column(s) "
                         f"{', '.join(unknown) or '(none)'}; "
                         f"known: {', '.join(PROFILE_COLUMNS)}")
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: repeated column in header {header}")
    rows = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not any(c.strip() for c in line):
            continue
        if len(line) != len(header):
            raise ValueError(f"{path}: line {lineno} has {len(line)} values, "
                             f"header has {len(header)}")
        vals = {"heave_mm": 0.0, "roll_deg": 0.0, "rack_mm": 0.0}
        for name, cell in zip(header, line):
            try:
                vals[name] = float(cell)
            except ValueError:
                raise ValueError(f"{path}: line {lineno}, column {name}: "
                                 f"'{cell}' is not a number") from None
        rows.append(tuple(vals[c] * PROFILE_COLUMNS[c]
                          for c in PROFILE_COLUMNS))
    return rows

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
