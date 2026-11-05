# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Suspension corner model: hardpoints, masses, and adjusters.

The JSON file stores lengths in mm, stiffness in N/mm, and damping in
N.s/mm. This module converts every value to SI on load. Coordinates follow
ISO 8855: x forward, y left, z up, origin on the ground below the axle
centerline.
"""

import copy
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
MM = 1e-3


def damper_table(spec):
    """The force-velocity table of a damper block, in SI.

    An ``adjustable`` block is a click damper: ``clicks`` positions,
    ``soft`` and ``firm`` asymmetric curves, and the settings
    ``bump_clicks`` and ``rebound_clicks`` (0 is soft).
    A ``table`` block lists the curve: ``velocity`` (mm/s, > 0 rebound) and
    ``force`` (N, > 0 in rebound).  An ``asymmetric`` block gives the slopes
    in N.s/mm: ``bump`` and ``rebound``, each one value (linear) or a pair
    (low-speed and high-speed slope, a digressive curve), and ``knee``, the
    speed (mm/s) where the slope changes.

    :param spec: the damper block of a model file
    :type spec: dict
    :return: (velocities in m/s, forces in N), velocities increasing
    :rtype: tuple
    :raises ValueError: for an unknown kind or a bad curve
    """
    kind = spec.get("kind", "asymmetric")
    if "adjustable" == kind:
        # A click-adjustable damper: the bump and rebound slopes move from
        # the soft curve (click 0) to the firm curve (the last click) with
        # their own adjusters.
        clicks = int(spec["clicks"])
        if clicks < 1:
            raise ValueError("damper: an adjustable damper needs clicks >= 1")
        out = {"kind": "asymmetric",
               "knee": spec["soft"].get("knee", 100.0)}
        for side, key in (("bump", "bump_clicks"),
                          ("rebound", "rebound_clicks")):
            click = float(spec.get(key, 0))
            if not 0.0 <= click <= clicks:
                raise ValueError(f"damper: {key} is 0 to {clicks}")
            w = click / clicks

            def pair(value):
                return value if isinstance(value, (list, tuple)) else \
                    [value, value]
            soft, firm = pair(spec["soft"][side]), pair(spec["firm"][side])
            out[side] = [(1.0 - w) * a + w * b for a, b in zip(soft, firm)]
        return damper_table(out)
    if "table" == kind:
        velocities = [float(v) * MM for v in spec["velocity"]]
        forces = [float(f) for f in spec["force"]]
    elif "asymmetric" == kind:
        def slopes(value):
            pair = value if isinstance(value, (list, tuple)) else \
                [value, value]
            if len(pair) != 2 or min(pair) < 0.0:
                raise ValueError("damper: a slope is one value or a "
                                 "(low-speed, high-speed) pair, not "
                                 "negative")
            return [float(c) * 1e3 for c in pair]

        bump, rebound = slopes(spec["bump"]), slopes(spec["rebound"])
        knee = float(spec.get("knee", 100.0)) * MM
        if knee <= 0.0:
            raise ValueError("damper: the knee speed must be positive")
        # Two points past each knee: the end segments carry the high-speed
        # slopes outward.
        velocities = [-2.0 * knee, -knee, 0.0, knee, 2.0 * knee]
        forces = [-(bump[0] + bump[1]) * knee, -bump[0] * knee, 0.0,
                  rebound[0] * knee, (rebound[0] + rebound[1]) * knee]
    else:
        raise ValueError(f"damper: unknown kind {kind!r} (table or "
                         "asymmetric)")
    if len(velocities) != len(forces) or len(velocities) < 2:
        raise ValueError("damper: the curve needs two or more points")
    if any(b <= a for a, b in zip(velocities[:-1], velocities[1:])):
        raise ValueError("damper: the velocities must increase")
    if any(f * v < 0.0 for v, f in zip(velocities, forces)):
        raise ValueError("damper: the curve must be passive: the force "
                         "has the sign of the velocity")
    return velocities, forces


def damper_force(curve, velocity):
    """The force (N) of a damper curve at ``velocity`` (m/s): piecewise
    linear, extended with the slopes of the end segments (the engine curve
    damper does the same).

    :param curve: (velocities, forces) of :func:`damper_table`
    :type curve: tuple
    :param velocity: damper rate, > 0 in rebound; a scalar or an array
    :rtype: float or numpy.ndarray
    """
    vs = np.asarray(curve[0], dtype="float64")
    fs = np.asarray(curve[1], dtype="float64")
    v = np.asarray(velocity, dtype="float64")
    hi = np.clip(np.searchsorted(vs, v, side="right"), 1, len(vs) - 1)
    lo = hi - 1
    slope = (fs[hi] - fs[lo]) / (vs[hi] - vs[lo])
    out = fs[lo] + slope * (v - vs[lo])
    return float(out) if out.ndim == 0 else out


def curve_slope(curve, velocity):
    """The slope dF/dv (N.s/m) of a damper curve at ``velocity``."""
    vs = np.asarray(curve[0], dtype="float64")
    fs = np.asarray(curve[1], dtype="float64")
    hi = int(np.clip(np.searchsorted(vs, velocity, side="right"), 1,
                     len(vs) - 1))
    return float((fs[hi] - fs[hi - 1]) / (vs[hi] - vs[hi - 1]))


class CornerModel:
    """Geometry and properties of one suspension corner in SI units.

    :ivar name: model name
    :vartype name: str
    :ivar side: +1 for a left corner, -1 for a right corner
    :vartype side: int
    :ivar hp: named hardpoints in m (global, design position)
    :vartype hp: dict
    :ivar links: link name -> (inner kind, inner point name, upright point
        name)
    :vartype links: dict
    :ivar limits: hardpoint name -> {"min": array, "max": array} box in m
    :vartype limits: dict

    Rule for hardpoints and adjusters: ``hp`` holds the unadjusted design
    points. The adjusters apply on top of them when a rig is built
    (``upright_points`` and ``link_length``), so ``set_hardpoint`` and
    ``set_adjusters`` commute.
    """

    #: The adjusters a model can define; each is a value in m.
    ADJUSTERS = ("camber_shim", "caster_shim", "toe_link")
    #: The kinds of an adjuster: shift upright points on the upright (the
    #: links keep their length), shift upright points in the car (the links
    #: reach them), shift chassis points, or change a link length.
    ADJUSTER_KINDS = ("upright_point", "joint_in_car", "chassis_point",
                      "link_length")

    def __init__(self, data):
        """Build the model from a parsed JSON dictionary.

        A corner has the upright (with the wheel) and, optionally, a rack
        (when a link ends on it), a rocker with a pushrod or pullrod (else
        the spring acts on the upright), a strut (the upright point
        ``strut_outer`` slides on the line to the top mount
        ``strut_inner``), and a trailing arm (the upright turns about the
        chassis axis ``arm_inner`` to ``arm_outer``).  The links, the strut
        (2 constraints), and the arm (5 constraints) must hold the upright
        with exactly 5 constraints, so the wheel travel is its one degree
        of freedom.

        :param data: parsed model file
        :type data: dict
        :raises ValueError: if the topology does not leave the upright
            exactly one degree of freedom, or a part is incomplete
        """
        self.raw = copy.deepcopy(data)
        self.name = data["name"]
        self.type = data.get("type", "double_wishbone")
        self.source = data.get("source", "")
        self.side = 1 if data.get("side", "left") == "left" else -1
        w = data["wheel"]
        self.wheel_radius = w["radius"] * MM
        self.spin_axis = np.array(w["spin_axis"], float)
        self.hp = {"wheel_center": np.array(w["center"]) * MM}
        self.links = {}
        for name, link in data.get("links", {}).items():
            inner_key = "rack" if "rack" in link else "chassis"
            self.hp[name + "_inner"] = np.array(link[inner_key]) * MM
            self.hp[name + "_outer"] = np.array(link["upright"]) * MM
            self.links[name] = (inner_key, name + "_inner", name + "_outer")
        self.upright_mass = data["upright"]["mass"]
        self.upright_inertia = list(data["upright"]["inertia"])
        # Wishbones as rigid arm bodies (a revolute about the two inner
        # pivots, a ball joint at the upright): needed when a pushrod or
        # the spring sits on the arm.  Kinematically the same as the two
        # links.
        self.arms = {}
        for arm, spec in data.get("arms", {}).items():
            names = list(spec["links"])
            if len(names) != 2 or any(n not in self.links for n in names):
                raise ValueError(f"arm '{arm}': needs two of the links")
            if any("rack" == self.links[n][0] for n in names):
                raise ValueError(f"arm '{arm}': a link on the rack cannot "
                                 "be part of an arm")
            outers = [self.hp[self.links[n][2]] for n in names]
            if float(np.linalg.norm(outers[0] - outers[1])) > 1e-9:
                raise ValueError(f"arm '{arm}': the two links must share "
                                 "the outer (ball joint) point")
            self.arms[arm] = {"links": names,
                              "mass": float(spec.get("mass", 1.0)),
                              "inertia": list(spec.get("inertia",
                                                       [0.01, 0.01, 0.01]))}
        # The rack: only a corner with a link on it has one.
        self.has_rack = any(kind == "rack" for kind, _, _
                            in self.links.values())
        self.rack_mass = 0.0
        self.rack_axis = np.array([0.0, 1.0, 0.0])
        if self.has_rack:
            if "rack" not in data:
                raise ValueError("a link ends on the rack, but the model "
                                 "has no 'rack' block")
            self.rack_mass = data["rack"]["mass"]
            self.rack_axis = np.array(data["rack"]["axis"], float)
        # The rocker with its pushrod (or pullrod), or none.
        self.has_rocker = "rocker" in data
        self.rod_kind = None
        self.rocker_mass = 0.0
        self.rocker_inertia = [0.0, 0.0, 0.0]
        self.rocker_axis = None
        s = data["spring"]
        self.hp["spring_chassis"] = np.array(s["chassis"]) * MM
        # The body of the outer end of the pushrod and of the spring:
        # "upright" or an arm name.
        self.rod_body = "upright"
        self.spring_on = "upright"
        if self.has_rocker:
            if "pushrod" not in data:
                raise ValueError("a rocker needs a 'pushrod' block")
            rod = data["pushrod"]
            self.rod_kind = rod.get("kind", "pushrod")
            self.rod_body = self._attach_body(rod.get("on", "upright"),
                                              "pushrod")
            self.hp["pushrod_outer"] = np.array(rod["upright"]) * MM
            self.hp["pushrod_inner"] = np.array(rod["rocker"]) * MM
            r = data["rocker"]
            self.hp["rocker_pivot"] = np.array(r["pivot"]) * MM
            self.rocker_axis = np.array(r["axis"], float)
            self.hp["spring_rocker"] = np.array(r["spring_point"]) * MM
            self.rocker_mass = r["mass"]
            self.rocker_inertia = list(r["inertia"])
        else:
            if "upright" not in s:
                raise ValueError("without a rocker the spring needs an "
                                 "'upright' point")
            self.spring_on = self._attach_body(s.get("on", "upright"),
                                               "spring")
            self.hp["spring_upright"] = np.array(s["upright"]) * MM
        self.has_strut = "strut" in data
        if self.has_strut:
            self.hp["strut_outer"] = np.array(data["strut"]["upright"]) * MM
            self.hp["strut_inner"] = \
                np.array(data["strut"]["top_mount"]) * MM
        self.has_arm = "arm" in data
        if self.has_arm:
            self.hp["arm_inner"] = np.array(data["arm"]["inner"]) * MM
            self.hp["arm_outer"] = np.array(data["arm"]["outer"]) * MM
        count = len(self.links) + 2 * self.has_strut + 5 * self.has_arm
        if count != 5:
            raise ValueError(
                f"model '{self.name}': the links ({len(self.links)}), the "
                f"strut (2), and the arm (5) give {count} constraints on "
                "the upright; exactly 5 leave the wheel travel as its one "
                "degree of freedom")
        self.spring_k = s["k"] * 1e3
        # The linear damper (N.s/m); a damper block below replaces it.
        self.spring_c = s["c"] * 1e3 if "damper" not in s else 0.0
        self.spring_free_offset = s.get("free_length_offset", 0.0) * MM
        # The static wheel travel (m, + bump) the spring holds: 0 at the
        # design pose; a ride height setting moves it (lower is + bump).
        self.static_travel = -data.get("ride_height", 0.0) * MM
        # The damper force-velocity curve ((velocities m/s, forces N), the
        # rate > 0 in rebound), or None for the linear damper spring_c.
        self.damper_curve = None
        # The damper block as written (the setup re-reads its clicks).
        self.damper_spec = copy.deepcopy(s.get("damper"))
        if "damper" in s:
            self.damper_curve = damper_table(s["damper"])
            # The linear analyses (K&C, modes, the linear ride response)
            # use the mean of the bump and rebound slopes at zero speed.
            self.spring_c = 0.5 * (curve_slope(self.damper_curve, -1e-9)
                                   + curve_slope(self.damper_curve, 1e-9))
        # Travel stops on the spring line (bump stop, droop stop), in SI:
        # the wheel travels (m) where they touch, rates N/m and N/m^2, and
        # damping N.s/m.  None: the corner has no stops.
        self.stops = None
        if "stops" in data:
            st = data["stops"]
            self.stops = {
                "bump_travel": st["bump_travel"] * MM,
                "droop_travel": st["droop_travel"] * MM,
                "k": st["k"] * 1e3, "k2": st.get("k2", 0.0) * 1e6,
                "c": st.get("c", 0.0) * 1e3}
            if not self.stops["droop_travel"] < 0.0 < \
                    self.stops["bump_travel"]:
                raise ValueError("stops: droop_travel < 0 < bump_travel "
                                 "is required")
        self.vehicle = {k: (v * MM if k in ("wheelbase", "cg_height",
                                            "half_track") else v)
                        for k, v in data.get("vehicle", {}).items()}
        self.adjusters = self._parse_adjusters(data.get("adjusters", {}))
        self.adjust = {name: 0.0 for name in self.ADJUSTERS}
        self.arb = None
        if "arb" in data:
            # The drop link of the bar ends on the rocker (``rocker_point``)
            # or, on a direct-acting corner, on the upright (``link_point``,
            # for example the strut tube).
            a = data["arb"]
            self.hp["arb_pivot"] = np.array(a["bar_pivot"]) * MM
            self.hp["arb_tip"] = np.array(a["arm_tip"]) * MM
            if "rocker_point" in a:
                if not self.has_rocker:
                    raise ValueError("the anti-roll bar rocker_point needs "
                                     "a rocker; use link_point")
                self.hp["arb_rocker"] = np.array(a["rocker_point"]) * MM
                link_body = "rocker"
            else:
                self.hp["arb_upright"] = np.array(a["link_point"]) * MM
                link_body = "upright"
            self.arb = {"axis": np.array(a["axis"], float), "k": a["k"],
                        "mass": a["mass"], "inertia": list(a["inertia"]),
                        "link_body": link_body}
        t = data.get("tire", {})
        self.tire_k = t.get("k", 300.0) * 1e3
        self.tire_c = t.get("c", 0.0) * 1e3
        b = data.get("bushings", {})
        self.bushing_k = b.get("k", 20000.0) * 1e3
        self.bushing_k_rot = b.get("k_rot", 50.0)
        # Steering axis definition: "auto" uses the ball joints when the
        # corner has two of them (see kingpin_joints) and the instantaneous
        # screw axis otherwise; "instant" forces the screw axis.
        self.steer_axis = data.get("steer_axis", "auto")
        if self.steer_axis not in ("auto", "instant"):
            raise ValueError("steer_axis must be 'auto' or 'instant', not "
                             f"{self.steer_axis!r}")
        self.limits = {}
        for name, box in data.get("limits", {}).items():
            self.set_limit(name, np.array(box["min"], float) * MM,
                           np.array(box["max"], float) * MM)

    #: The adjusters of a double wishbone file without explicit kinds.
    LEGACY_ADJUSTERS = {
        "camber_shim": {"kind": "upright_point",
                        "points": ["upper_front_outer", "upper_rear_outer"],
                        "direction": [0.0, 1.0, 0.0]},
        "caster_shim": {"kind": "joint_in_car",
                        "points": ["upper_front_outer", "upper_rear_outer"],
                        "direction": [1.0, 0.0, 0.0]},
        "toe_link": {"kind": "link_length", "link": "tie_rod", "sign": 1.0},
    }

    def _parse_adjusters(self, data):
        """``{name: spec}`` of the adjusters the model defines.

        A spec has ``kind`` and, by kind, ``points`` with a global
        ``direction`` at design, or ``link`` with a ``sign``.  A block
        without a kind is a double wishbone file of the first format (the
        upper ball joints and the tie rod).
        """
        out = {}
        for name, spec in data.items():
            if name not in self.ADJUSTERS:
                raise ValueError(f"unknown adjuster '{name}'; known: "
                                 f"{', '.join(self.ADJUSTERS)}")
            if "kind" not in spec:
                spec = copy.deepcopy(self.LEGACY_ADJUSTERS[name])
                if "camber_shim" == name:
                    # Outboard on this side.
                    spec["direction"] = [0.0, float(self.side), 0.0]
            kind = spec["kind"]
            if kind not in self.ADJUSTER_KINDS:
                raise ValueError(f"adjuster '{name}': unknown kind "
                                 f"'{kind}'")
            if "link_length" == kind:
                if spec["link"] not in self.links:
                    raise ValueError(f"adjuster '{name}': no link "
                                     f"'{spec['link']}'")
                if any(spec["link"] in a["links"]
                       for a in self.arms.values()):
                    raise ValueError(f"adjuster '{name}': the link "
                                     f"'{spec['link']}' is part of a rigid "
                                     "arm")
                out[name] = {"kind": kind, "link": spec["link"],
                             "sign": float(spec.get("sign", 1.0))}
                continue
            points = [p for p in spec["points"] if p in self.hp]
            if len(points) != len(spec["points"]):
                missing = sorted(set(spec["points"]) - set(points))
                raise ValueError(f"adjuster '{name}': unknown points "
                                 f"{missing}")
            upright = set(self.upright_point_names())
            on_upright = [p in upright for p in points]
            if ("chassis_point" == kind) == any(on_upright) or \
                    ("chassis_point" != kind and not all(on_upright)):
                raise ValueError(f"adjuster '{name}': a {kind} adjuster "
                                 "moves "
                                 + ("chassis" if "chassis_point" == kind
                                    else "upright") + " points")
            direction = np.array(spec["direction"], float)
            out[name] = {"kind": kind, "points": points,
                         "direction": direction / np.linalg.norm(direction)}
        return out

    def _attach_body(self, name, what):
        if "upright" == name or name in self.arms:
            return name
        raise ValueError(f"{what}: 'on' must be 'upright' or an arm "
                         f"({', '.join(sorted(self.arms)) or 'none'}), not "
                         f"{name!r}")

    def upright_point_names(self):
        """Names of the hardpoints fixed to the upright."""
        names = ["wheel_center"]
        names += [outer for _, _, outer in self.links.values()]
        if "pushrod_outer" in self.hp and "upright" == self.rod_body:
            names.append("pushrod_outer")
        if "spring_upright" in self.hp and "upright" == self.spring_on:
            names.append("spring_upright")
        if "strut_outer" in self.hp:
            names.append("strut_outer")
        if "arb_upright" in self.hp:
            names.append("arb_upright")
        return names

    def arm_points(self, arm):
        """``(inner front, inner rear, ball joint)`` of an arm at design with
        the adjusters applied (global, m): the inner pivots of its two
        links and their shared outer point (moved by a joint-in-car
        adjuster, not by a shim on the upright)."""
        names = self.arms[arm]["links"]
        inner = [self.chassis_point(self.links[n][1]) for n in names]
        outer = self.links[names[0]][2]
        ball = self.hp[outer] + self._shift(("joint_in_car",), outer)
        return inner[0], inner[1], ball

    def arm_of(self, point):
        """The arm a hardpoint rides on (``pushrod_outer`` or
        ``spring_upright`` set ``on`` an arm), or None."""
        if "pushrod_outer" == point and self.rod_body in self.arms:
            return self.rod_body
        if "spring_upright" == point and self.spring_on in self.arms:
            return self.spring_on
        return None

    def spring_body(self):
        """The body the spring acts on: ``'rocker'``, ``'upright'``, or
        ``'arm_<name>'``."""
        if self.has_rocker:
            return "rocker"
        return ("upright" if "upright" == self.spring_on
                else "arm_" + self.spring_on)

    def spring_point_name(self):
        """The hardpoint of the spring on its moving body."""
        return "spring_rocker" if self.has_rocker else "spring_upright"

    @classmethod
    def load(cls, name_or_path):
        """Load a model from ``suspension/data/<name>.json`` or a path.

        :param name_or_path: file stem in the data directory, or a file path
        :type name_or_path: str or pathlib.Path
        :return: model
        :rtype: CornerModel
        """
        path = Path(name_or_path)
        if not path.exists():
            path = DATA_DIR / f"{name_or_path}.json"
        with open(path) as f:
            return cls(json.load(f))

    def mirrored(self):
        """Return the mirror image about the vehicle centerplane y = 0.

        :return: model of the opposite corner
        :rtype: CornerModel
        """
        m = copy.deepcopy(self)
        m.side = -self.side
        for k in m.hp:
            m.hp[k] = m.hp[k] * np.array([1.0, -1.0, 1.0])
        m.spin_axis = self.spin_axis * np.array([1.0, -1.0, 1.0])
        m.rack_axis = self.rack_axis * np.array([1.0, -1.0, 1.0])
        # the bar axis is a rotation axis too, but it is along y
        if m.arb is not None:
            m.arb["axis"] = self.arb["axis"] * np.array([-1.0, 1.0, -1.0])
        # An axis vector of a rotation is a pseudo-vector: mirror x and z.
        if self.rocker_axis is not None:
            m.rocker_axis = self.rocker_axis * np.array([-1.0, 1.0, -1.0])
        # An adjuster direction is a displacement: mirror y.
        for spec in m.adjusters.values():
            if "direction" in spec:
                spec["direction"] = spec["direction"] * np.array(
                    [1.0, -1.0, 1.0])
        # Mirror of a box: the y bounds change sign and swap.
        flip = np.array([1.0, -1.0, 1.0])
        for name, box in self.limits.items():
            a, b = box["min"] * flip, box["max"] * flip
            m.limits[name] = {"min": np.minimum(a, b), "max": np.maximum(a, b)}
        return m

    def translated(self, offset):
        """Return a copy moved by ``offset`` (m, global): a corner placed in
        a vehicle, for example a rear corner one wheelbase behind the front
        axle.

        :param offset: translation (m)
        :type offset: sequence
        :rtype: CornerModel
        """
        d = np.asarray(offset, dtype="float64")
        m = copy.deepcopy(self)
        for k in m.hp:
            m.hp[k] = m.hp[k] + d
        for box in m.limits.values():
            box["min"] = box["min"] + d
            box["max"] = box["max"] + d
        return m

    def hardpoint_names(self):
        """Names of the editable hardpoints.

        Each link end is its own point: a double wishbone has two coincident
        outer points per wishbone (for example ``lower_front_outer`` and
        ``lower_rear_outer``). ``coincident_points`` lists them.

        :return: sorted hardpoint names
        :rtype: list
        """
        return sorted(self.hp)

    def kingpin_joints(self, tol=1e-9):
        """The lower and upper ball joints of a double wishbone, or None.

        A ball joint is an upright point that two links share (the outer
        ends of a wishbone). With exactly two of them, the steering axis is
        the line through them; a multi-link without shared outer points has a
        virtual steering axis instead (the instantaneous screw axis).

        :return: (lower, upper) upright point names, or None
        :rtype: tuple or None
        """
        if self.steer_axis == "instant":
            return None
        joints = self.shared_outer_points(tol)
        if len(joints) != 2:
            return None
        lower, upper = joints
        return lower, upper

    def shared_outer_points(self, tol=1e-9):
        """Ball joints: the first name of each group of link outer points
        at the same design position, sorted by height.

        :return: point names
        :rtype: list
        """
        outers = [outer for _, (_, _, outer) in self.links.items()]
        joints = []
        for name in outers:
            shared = [o for o in outers if o != name
                      and float(np.linalg.norm(self.hp[o] - self.hp[name]))
                      <= tol]
            if shared and not any(name in group for group in joints):
                joints.append([name] + shared)
        return sorted((group[0] for group in joints),
                      key=lambda n: self.hp[n][2])

    def coincident_points(self, name, tol=1e-9):
        """Other hardpoints at the same design position as ``name``.

        :param name: hardpoint name
        :type name: str
        :param tol: distance below which two points coincide (m)
        :type tol: float
        :return: sorted names, without ``name``
        :rtype: list
        """
        self._check_name(name)
        p = self.hp[name]
        return sorted(k for k, v in self.hp.items()
                      if k != name and float(np.linalg.norm(v - p)) <= tol)

    def _check_name(self, name):
        if name not in self.hp:
            raise ValueError(f"unknown hardpoint '{name}'; known: "
                             f"{', '.join(sorted(self.hp))}")

    def _check_box(self, name, p, box):
        for i, axis in enumerate("xyz"):
            if p[i] < box["min"][i]:
                raise ValueError(f"hardpoint '{name}': {axis} = {p[i]:.6g} m "
                                 "is below the "
                                 f"limit min {box['min'][i]:.6g} m")
            if p[i] > box["max"][i]:
                raise ValueError(f"hardpoint '{name}': {axis} = {p[i]:.6g} m "
                                 "is above the "
                                 f"limit max {box['max'][i]:.6g} m")

    def set_hardpoint(self, name, xyz):
        """Move one hardpoint (unadjusted design value, global, m).

        Every rig reads the hardpoints when it is built (link lengths,
        upright and rocker attachment points, wheel center), so no other
        data needs an update. Adjusters still apply on top.

        :param name: hardpoint name (see ``hardpoint_names``)
        :type name: str
        :param xyz: new position (m)
        :type xyz: list or numpy.ndarray
        :raises ValueError: if the name is unknown, the position is not 3
            finite numbers, or the position is outside the limit box; the
            model is then unchanged
        """
        self._check_name(name)
        p = np.array(xyz, float).reshape(-1)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError(f"hardpoint '{name}': need 3 finite coordinates, "
                             f"got {xyz!r}")
        if name in self.limits:
            self._check_box(name, p, self.limits[name])
        self.hp[name] = p

    def set_limit(self, name, lo, hi):
        """Set or remove the limit box of a hardpoint (m).

        :param name: hardpoint name
        :type name: str
        :param lo: minimum x, y, z (m), or None with ``hi`` None to remove
        :type lo: list or numpy.ndarray or None
        :param hi: maximum x, y, z (m), or None with ``lo`` None to remove
        :type hi: list or numpy.ndarray or None
        :raises ValueError: if the name is unknown, a min is above its max,
            or the box excludes the current point; the box is then unchanged
        """
        self._check_name(name)
        if lo is None and hi is None:
            self.limits.pop(name, None)
            return
        lo = np.array(lo, float).reshape(-1)
        hi = np.array(hi, float).reshape(-1)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError(f"limit of '{name}': need 3 values for min and "
                             "for max")
        for i, axis in enumerate("xyz"):
            if lo[i] > hi[i]:
                raise ValueError(f"limit of '{name}': {axis} min {lo[i]:.6g} "
                                 "m is above "
                                 f"max {hi[i]:.6g} m")
        box = {"min": lo, "max": hi}
        self._check_box(name, self.hp[name], box)
        self.limits[name] = box

    def set_adjusters(self, camber_shim=0.0, caster_shim=0.0, toe_link=0.0):
        """Set adjuster values in m.

        What an adjuster moves is the model's (``adjusters`` in the file):
        for the double wishbone, camber_shim spaces the upper ball joint
        outboard on the upright (the wheel top tilts inboard: more negative
        camber), caster_shim moves the upper ball joint forward in the car
        (an eccentric or shims on the upper wishbone: the arms reach the
        moved joint, so the steering axis tilts and the caster drops), and
        toe_link lengthens the tie rod.

        A shim between the upright and the ball joint in the fore/aft
        direction would not change caster: the wishbones hold the joint, so
        only the upright pitches about the unchanged steering axis.

        :param camber_shim: camber adjuster (m)
        :type camber_shim: float
        :param caster_shim: caster adjuster (m)
        :type caster_shim: float
        :param toe_link: toe adjuster (m)
        :type toe_link: float
        :raises ValueError: for a nonzero value of an adjuster the model
            does not have
        """
        values = {"camber_shim": camber_shim, "caster_shim": caster_shim,
                  "toe_link": toe_link}
        for name, value in values.items():
            if value != 0.0 and name not in self.adjusters:
                raise ValueError(f"model '{self.name}' has no {name} "
                                 "adjuster")
        self.adjust = values

    def _shift(self, kinds, name):
        """Sum of the adjuster shifts of ``kinds`` on point ``name`` (m)."""
        out = np.zeros(3)
        for adj, spec in self.adjusters.items():
            if spec["kind"] in kinds and name in spec.get("points", ()):
                out = out + self.adjust[adj] * spec["direction"]
        return out

    def upright_points(self):
        """Upright attachment points at design, with adjusters applied (global,
        m).

        :return: point name -> position
        :rtype: dict
        """
        return {k: self.hp[k] + self._shift(("upright_point",
                                             "joint_in_car"), k)
                for k in self.upright_point_names()}

    def chassis_point(self, name):
        """A chassis (or rack) hardpoint with the adjusters applied (global,
        m)."""
        return self.hp[name] + self._shift(("chassis_point",), name)

    def link_length(self, name):
        """Length of a link at design with adjusters applied (m).

        :param name: link name
        :type name: str
        :return: length
        :rtype: float
        """
        _, inner, outer = self.links[name]
        # A joint moved in the car: the link reaches it.  A shim on the
        # upright or a moved chassis point keeps the link length.
        end = self.hp[outer] + self._shift(("joint_in_car",), outer)
        length = float(np.linalg.norm(self.hp[inner] - end))
        for adj, spec in self.adjusters.items():
            if "link_length" == spec["kind"] and spec["link"] == name:
                length += spec["sign"] * self.adjust[adj]
        return length

    def steering_axis_points(self):
        """The two points of a geometric steering axis, or None.

        :return: ``[(frame, name), (frame, name)]``, lower point first, with
            frame ``'upright'`` or ``'chassis'``: the two ball joints of a
            double wishbone, or the lower ball joint and the top mount of a
            strut; None for a virtual (instantaneous) axis
        :rtype: list or None
        """
        joints = self.kingpin_joints()
        if joints is not None:
            return [("upright", joints[0]), ("upright", joints[1])]
        if self.has_strut and self.steer_axis != "instant":
            lower = self.shared_outer_points()
            if len(lower) == 1:
                return [("upright", lower[0]), ("chassis", "strut_inner")]
        return None

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
