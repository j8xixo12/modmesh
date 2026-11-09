# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Longitudinal dynamics of the full vehicle: braking and traction.

The steady-state rig (:func:`steady_state`) loads every body with the
D'Alembert force -m a_x and every tire with its share of the braking or
the drive force, and solves the static equilibrium of the vehicle on its
springs and tires: pitch, the travel of each corner (dive and squat or
lift), and the wheel loads.  Where the tire force enters the suspension
matters (Milliken & Milliken, RCVD, 1995, Ch. 17; Gillespie, Fundamentals
of Vehicle Dynamics, 1992, Ch. 7):

* outboard brakes: the brake torque stays in the upright, so the force acts
  at the contact patch;
* inboard drive (half shafts from a body-mounted differential) and inboard
  brakes: the shaft torque goes to the body, so the upright sees the force
  at the wheel center and the body the reaction torque.

By virtual work the force F_x moves the travel s of a corner as the
generalized force F_x a_x, with a_x = dx/ds of the point it acts at, while
the load change dF_z acts through a_z = dz/ds of the contact patch:

    k_w ds = dF_z a_z + F_x a_x - m_u a_x g c_x,

the last term the D'Alembert load of the upright at its center, c_x =
dx/ds of the wheel center.

The anti fraction is the share of the load change the geometry carries:
anti = -F_x a_x / (dF_z a_z).  With the side-view swing arm angle
tan(theta) = a_x / a_z it is the anti-dive of RCVD Ch. 17, beta tan(theta)
L / h for a front axle with the brake share beta.

Brake balance (Gillespie Ch. 3): with the brake share beta of the front
axle, the front locks when beta W d exceeds the grip of the front tires at
their dynamic loads W_f + W d h / L, the rear when (1 - beta) W d exceeds
the rear grip at W_r - W d h / L; the ideal share is the front share of
the grip.  :func:`brake_balance` uses the dynamic loads of the rig and the
load-sensitive friction of the vehicle tires.
"""

import math

import numpy as np

from . import car
from . import kc
from .metrics import rot

__all__ = [  # noqa: F822
    'DRIVES',
    'anti_geometry',
    'brake_balance',
    'predicted_travel',
    'steady_state',
    'traction_limit',
]

#: Drive layouts: the front share of the drive force.
DRIVES = {'rwd': 0.0, 'fwd': 1.0, 'awd': 0.4}
#: The largest step (g) of the continuation of the steady states.
MAX_STEP_G = 0.1


def _pitch(quat):
    r = rot(np.asarray(quat, dtype="float64"))
    # Nose down is + (a rotation about +y).
    return float(math.atan2(r[0, 2], r[2, 2]))


def _shares(a, brake_bias, drive_split):
    """The front share of the tire force: the brake bias when braking (a <
    0), the drive split when driving."""
    return float(brake_bias) if a < 0.0 else float(drive_split)


def steady_state(mbs, vm, accelerations_g, brake_bias=0.6, drive_split=0.0,
                 inboard_brakes=False, gravity=9.81):
    """The vehicle at steady longitudinal accelerations (g, + forward,
    - braking).

    :param brake_bias: front share of the braking force
    :param drive_split: front share of the drive force (0 rear drive, 1
        front drive)
    :param inboard_brakes: the brakes on the body (the braking force acts
        at the wheel center like a drive force)
    :return: dict of arrays over the accelerations: ``a_x`` (g), ``pitch``
        (rad, + nose down), ``loads`` (N, the corners of
        :data:`car.CORNERS`), ``travel`` (m, + bump, each corner from its
        static travel), ``force`` (N, the tire force of each corner),
        ``inboard`` (bool, the force acted at the wheel center), ``mass``
        (kg), and ``cg_height`` (m) of all the bodies; ``poses``
        (:func:`car.vehicle_pose` of each state) and the corner ``models``
    """
    models = car.corner_models(mbs, vm, gravity)
    s, h = car.build(mbs, vm, 0.0, gravity,
                     side_forces={name: 0.0 for name in car.CORNERS},
                     models=models)
    uprights = {h["corners"][name].nums["upright"]: name
                for name in car.CORNERS}
    total_mass = sum(mass for _, mass in h["bodies"])
    pos0 = np.asarray(s.get_body_positions())
    # The center of gravity height of every body with a D'Alembert load.
    cg_height = sum(mass * pos0[num][2] for num, mass in h["bodies"]) \
        / total_mass
    rows = {"pitch": [], "loads": [], "travel": [], "force": [],
            "inboard": []}
    poses = []

    def corner_travel():
        pos = np.asarray(s.get_body_positions())
        quat = np.asarray(s.get_body_quaternions())
        r_ch = rot(quat[1])
        out = []
        for name in car.CORNERS:
            num = h["corners"][name].nums["upright"]
            local = r_ch.T @ (pos[num] - pos[1])
            out.append(float(local[2]))
        return np.array(out)

    def apply(a):
        share = _shares(a, brake_bias, drive_split)
        inboard = (a > 0.0) or inboard_brakes
        # The tires push the car with the force M a; each axle takes its
        # share, split between its two wheels.
        total = total_mass * a * gravity
        force = {name: 0.5 * total * (share if name.startswith("front")
                                      else 1.0 - share)
                 for name in car.CORNERS}
        field = np.array([-a * gravity, 0.0, -gravity])
        pos = np.asarray(s.get_body_positions())
        quat = np.asarray(s.get_body_quaternions())
        chassis_torque = np.zeros(3)
        for num, mass in h["bodies"]:
            if num not in uprights:
                continue
            name = uprights[num]
            fx = np.array([force[name], 0.0, 0.0])
            spin = rot(quat[num]) @ models[name].spin_axis
            down = np.array([0.0, 0.0, -1.0]) + spin[2] * spin
            down /= np.linalg.norm(down)
            lever = pos[num][2] / -down[2]
            if inboard:
                # The shaft takes the moment of the tire force about the
                # wheel center to the body.
                s.set_applied_load(num, list(mass * field + fx), [0, 0, 0])
                chassis_torque += np.cross(lever * down, fx)
            else:
                s.set_applied_load(num, list(mass * field + fx),
                                   list(np.cross(lever * down, fx)))
        for num, mass in h["bodies"]:
            if num in uprights:
                continue
            torque = chassis_torque if 1 == num else np.zeros(3)
            s.set_applied_load(num, list(mass * field), list(torque))
        return force, inboard

    last = 0.0
    for a in accelerations_g:
        count = max(1, int(math.ceil(abs(a - last) / MAX_STEP_G)))
        for step in range(1, count + 1):
            force, inboard = apply(last + (a - last) * step / count)
            res = s.find_equilibrium(tol=1e-10)
            if not res["converged"]:
                raise RuntimeError(f"vehicle: no equilibrium at a_x "
                                   f"{a:g} g: {res['message']}")
        last = a
        quat = np.asarray(s.get_body_quaternions())
        rows["pitch"].append(_pitch(quat[1]))
        rows["loads"].append([float(mbs.get_tire_state(h["tires"][name])
                                    ["normal_force"])
                              for name in car.CORNERS])
        rows["travel"].append(corner_travel())
        rows["force"].append([force[name] for name in car.CORNERS])
        rows["inboard"].append(inboard)
        poses.append(car.vehicle_pose(s, h))
    out = {key: np.array(val) for key, val in rows.items()}
    out["a_x"] = np.asarray(accelerations_g, dtype="float64")
    # The travel from the static state (the first acceleration must be 0
    # for a travel from static; else from the first one).
    out["travel"] = out["travel"] - out["travel"][0]
    out["pitch"] = out["pitch"] - out["pitch"][0]
    out["mass"] = total_mass
    out["cg_height"] = cg_height
    out["poses"] = poses
    out["models"] = models
    return out


def anti_geometry(mbs, vm, gravity=9.81):
    """The K&C numbers of the longitudinal analysis, per axle at the
    static travel: the wheel rate, the contact patch rates (a_x, a_z) and
    the wheel center rate a_x, the side-view angles, and the anti
    fractions of RCVD Ch. 17 for braking (outboard) and driving (inboard)
    forces of a unit share.

    :return: ``{"front": {...}, "rear": {...}}``
    """
    models = car.corner_models(mbs, vm, gravity)
    h = vm.cg_height
    length = vm.wheelbase
    out = {}
    for axle_name in ("front", "rear"):
        model = models[f"{axle_name}_left"]
        s0 = model.static_travel
        sweep = kc.bump_sweep(mbs, model, [s0 - 1e-3, s0, s0 + 1e-3])
        patch = sweep["patch_rate"][1]
        center = sweep["center_rate"][1]
        sign = 1.0 if "front" == axle_name else -1.0
        out[axle_name] = {
            "wheel_rate": float(sweep["wheel_rate"][1]),
            "patch_rate": patch, "center_rate": center,
            # The side-view swing arm angles (+ for an anti effect).
            "patch_angle": math.atan2(sign * patch[0], patch[2]),
            "center_angle": math.atan2(sign * center[0], center[2]),
        }
        # RCVD: anti = tan(theta) L / h per unit share of the force.
        out[axle_name]["anti_brake"] = \
            sign * patch[0] / patch[2] * length / h
        out[axle_name]["anti_drive"] = \
            sign * center[0] / center[2] * length / h
        out[axle_name]["upright_mass"] = model.upright_mass
    return out


def predicted_travel(mbs, vm, state, gravity=9.81):
    """The corner travels of :func:`steady_state` by virtual work, k_w ds
    = dF_z a_z + F_x a_x - m_u a_x g c_x, from the K&C rates and the rig
    loads.

    :return: (travel (m), anti fraction of each corner)
    """
    geo = anti_geometry(mbs, vm, gravity)
    travel, anti = [], []
    base = state["loads"][0]
    for k in range(len(state["a_x"])):
        row_t, row_a = [], []
        for c, name in enumerate(car.CORNERS):
            g = geo["front" if name.startswith("front") else "rear"]
            rate = g["center_rate"] if state["inboard"][k] \
                else g["patch_rate"]
            dfz = state["loads"][k][c] - base[c]
            fx = state["force"][k][c]
            # The D'Alembert load -m a_x of the upright acts at its center.
            inertia = -g["upright_mass"] * state["a_x"][k] * gravity
            q = dfz * g["patch_rate"][2] + fx * rate[0] \
                + inertia * g["center_rate"][0]
            row_t.append(q / g["wheel_rate"])
            row_a.append(-fx * rate[0] / (dfz * g["patch_rate"][2])
                         if abs(dfz) > 1e-9 else 0.0)
        travel.append(row_t)
        anti.append(row_a)
    return np.array(travel), np.array(anti)


def _capacity(vm, front, loads):
    """The longitudinal peak force (N) of an axle's tires."""
    tire = vm.tire(front)
    return sum(tire.friction(max(float(fz), 0.0))[0] * max(float(fz), 0.0)
               for fz in loads)


def brake_balance(mbs, vm, brake_bias, decelerations_g=None,
                  inboard_brakes=False, gravity=9.81):
    """The braking limit of a brake bias: where the first axle uses all of
    its grip (it locks), with the dynamic loads of the rig.

    :return: dict with ``decel`` (g), ``use`` (front, rear grip use),
        ``ideal_bias`` (the front share of the grip at each deceleration),
        ``limit`` (g), ``locks`` ("front" or "rear"), and ``state``
    """
    if decelerations_g is None:
        decelerations_g = list(np.round(np.arange(0.0, 1.61, 0.05), 3))
    state = steady_state(mbs, vm, [-d for d in decelerations_g],
                         brake_bias=brake_bias,
                         inboard_brakes=inboard_brakes, gravity=gravity)
    total = state["mass"] * gravity
    use, ideal = [], []
    for k, d in enumerate(decelerations_g):
        loads = state["loads"][k]
        cap_f = _capacity(vm, True, loads[:2])
        cap_r = _capacity(vm, False, loads[2:])
        use.append((brake_bias * total * d / cap_f,
                    (1.0 - brake_bias) * total * d / cap_r))
        ideal.append(cap_f / (cap_f + cap_r))
    use = np.array(use)
    limit, locks = _crossing(decelerations_g, use)
    return {"decel": np.asarray(decelerations_g, dtype="float64"),
            "use": use, "ideal_bias": np.array(ideal), "limit": limit,
            "locks": locks, "state": state}


def traction_limit(mbs, vm, drive_split=0.0, accelerations_g=None,
                   gravity=9.81):
    """The traction limit of a drive layout: where the first driven axle
    uses all of its grip (it spins its wheels).

    :return: dict with ``accel`` (g), ``use``, ``limit`` (g), ``spins``
        ("front" or "rear"), and ``state``
    """
    if accelerations_g is None:
        accelerations_g = list(np.round(np.arange(0.0, 1.21, 0.05), 3))
    state = steady_state(mbs, vm, accelerations_g, drive_split=drive_split,
                         gravity=gravity)
    total = state["mass"] * gravity
    use = []
    for k, a in enumerate(accelerations_g):
        loads = state["loads"][k]
        use.append((drive_split * total * a
                    / _capacity(vm, True, loads[:2]),
                    (1.0 - drive_split) * total * a
                    / _capacity(vm, False, loads[2:])))
    use = np.array(use)
    limit, spins = _crossing(accelerations_g, use)
    return {"accel": np.asarray(accelerations_g, dtype="float64"),
            "use": use, "limit": limit, "spins": spins, "state": state}


def _crossing(values, use):
    """The value where the busier axle reaches a grip use of 1 (linear
    between samples), and that axle; the last value when none does."""
    busy = use.max(axis=1)
    for k in range(1, len(values)):
        if busy[k] >= 1.0 > busy[k - 1]:
            frac = (1.0 - busy[k - 1]) / (busy[k] - busy[k - 1])
            axle = "front" if use[k][0] >= use[k][1] else "rear"
            return values[k - 1] + frac * (values[k] - values[k - 1]), axle
    return values[-1], None


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
