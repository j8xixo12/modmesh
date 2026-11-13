# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Full vehicle: four corners on one sprung mass.

A vehicle file names a front and a rear corner model (the left corners;
the right ones are their mirrors) and gives the wheelbase, the sprung mass,
its center of gravity, and the tires.  The rear corners sit one wheelbase
behind the front axle.  Each spring gets the free length that holds the
design pose under the static load of its corner.

Steady-state cornering (:func:`steady_state`): the lateral acceleration
a_y acts as a D'Alembert load -m a_y on every body; the tires carry the
vertical loads, and a stiff lateral and longitudinal support at each
contact patch carries the side force at the ground, as the tires do in a
steady turn.  The chassis is free, so statics alone split the side force
between the axles (yaw moment balance).

The closed form (:func:`closed_form`) is the roll center model of Milliken
and Milliken, Race Car Vehicle Dynamics (1995), Ch. 18, and Gillespie,
Fundamentals of Vehicle Dynamics (1992), Ch. 7: with the axle roll
stiffnesses K_f and K_r (springs and bar in series with the tires), the
roll center heights h_f and h_r, the sprung weight W_s and its height h'
above the roll axis,

    roll gradient   phi / A_y = W_s h' / (K_f + K_r - W_s h'),
    load transfer   dW_f / A_y = (W_s h' K_f / (K_f + K_r - W_s h')
                                  + W_sf h_f + W_uf z_uf) / t_f,

with A_y in g, W_sf the static sprung weight on the front axle, W_uf and
z_uf the unsprung weight and its height; LLTD = dW_f / (dW_f + dW_r).

The understeer gradient (Gillespie Eq. 6-15 with roll steer, Sec. 8) is
K = W_f / C_f - W_r / C_r + (roll steer, front minus rear, per g), with the
axle cornering stiffnesses C; the steady yaw rate gain of the linear bicycle
model is r / delta = V / (L + K V^2 / g).
"""

import copy
import json
import math
from pathlib import Path

import numpy as np

from . import axle
from . import corner as corner_mod
from . import kc
from . import tuning
from . import vehicle
from .model import MM, CornerModel

__all__ = [  # noqa: F822
    'VehicleModel',
    'build',
    'closed_form',
    'axle_loads',
    'corner_models',
    'handling_diagram',
    'kinematic_steer',
    'peak_slip_angle',
    'step_steer',
    'steady_state',
    'tire_models',
    'tire_side_force',
    'understeer_gradient',
    'vehicle_pose',
]

DATA_DIR = Path(__file__).parent / "data" / "vehicles"
Z = [0.0, 0.0, 0.0]
DEG = math.pi / 180.0
#: The largest step (g) of the continuation of the steady-state turns.
MAX_STEP_G = 0.04
#: Stiffness (N/m, N m/rad) of the chassis support of the steady-state
#: rig.  At the fixed point the side forces balance the turn and the
#: support carries nothing, so its stiffness does not enter the result; it
#: must exceed the negative yaw stiffness of the side forces on the loaded
#: outer wheels (about a_y sum F_z y, 1e3 N m/rad at 0.3 g for a car).
SOFT_SUPPORT = 1e6
#: The four corners, in body numbering order.
CORNERS = ("front_left", "front_right", "rear_left", "rear_right")


#: The keys of the ``tire`` block of a vehicle file.
TIRE_KEYS = ("cornering_stiffness", "mu", "shape", "curvature",
             "load_exponent", "reference_load", "longitudinal_stiffness",
             "mu_x", "relaxation_length", "magic_formula")
#: The load exponent n (mu ~ F_z^(n - 1)) of a tire block without one: the
#: friction falls 0.12 per unit relative load at the reference load.
DEFAULT_LOAD_EXPONENT = 0.88


def _pair(value, name):
    out = [float(v) for v in (value if np.ndim(value) else [value, value])]
    if len(out) != 2:
        raise ValueError(f"tire {name} is one value or (front, rear)")
    return out


def tire_models(block, static_loads):
    """The Magic Formula tires (front, rear) of a vehicle ``tire`` block.

    The block gives either ``magic_formula``, ``{"front": {...}, "rear":
    {...}}`` with the arguments of :class:`MagicFormulaTire` (``fz0``,
    ``longitudinal``, ``lateral``), or the values of one tire at its
    reference load (each one value or a front and rear pair):

    * ``cornering_stiffness`` (N/deg, default 1200) and ``mu`` (default 1)
      at ``reference_load`` (N, default the static wheel load);
    * ``shape`` C (default 1.3) and ``curvature`` E (default 0) of the
      Magic Formula; ``load_exponent`` n (default
      :data:`DEFAULT_LOAD_EXPONENT`): the peak scales as F_z^n and, with B
      fixed, so does the cornering stiffness;
    * ``longitudinal_stiffness`` (N per unit slip ratio, default the same B
      as the side force) and ``mu_x`` (default ``mu``).

    :param static_loads: the static wheel loads (N), front and rear
    :raises ValueError: for an unknown key
    """
    from solvcon import multibody
    unknown = sorted(set(block) - set(TIRE_KEYS))
    if unknown:
        raise ValueError(f"tire: unknown keys {unknown}")
    if "magic_formula" in block:
        extra = sorted(set(block) - {"magic_formula", "relaxation_length"})
        if extra:
            raise ValueError(f"tire: {extra} conflict with magic_formula")
        mf = block["magic_formula"]
        return [multibody.MagicFormulaTire(**mf[axle_name])
                for axle_name in ("front", "rear")]
    stiff = _pair(block.get("cornering_stiffness", 1200.0),
                  "cornering_stiffness")
    mu = _pair(block.get("mu", 1.0), "mu")
    mu_x = _pair(block.get("mu_x", mu), "mu_x")
    ref = _pair(block.get("reference_load", static_loads), "reference_load")
    shape = float(block.get("shape", 1.3))
    curvature = float(block.get("curvature", 0.0))
    exponent = float(block.get("load_exponent", DEFAULT_LOAD_EXPONENT))
    out = []
    for k in range(2):
        b_y = stiff[k] / DEG / (shape * mu[k] * ref[k])
        if "longitudinal_stiffness" in block:
            b_x = _pair(block["longitudinal_stiffness"],
                        "longitudinal_stiffness")[k] \
                / (shape * mu_x[k] * ref[k])
        else:
            b_x = b_y
        out.append(multibody.MagicFormulaTire(
            fz0=ref[k],
            longitudinal=dict(mu=mu_x[k], load_exponent=exponent,
                              stiffness=b_x, shape=shape,
                              curvature=curvature),
            lateral=dict(mu=mu[k], load_exponent=exponent, stiffness=b_y,
                         shape=shape, curvature=curvature)))
    return out


class VehicleModel(object):
    """A vehicle of two corner models.

    :ivar front: left front corner model (at x = 0)
    :ivar rear: left rear corner model (one wheelbase behind)
    :ivar wheelbase: (m)
    :ivar sprung_mass: (kg)
    :ivar front_fraction: share of the sprung weight on the front axle
    :ivar cg_height: sprung mass center of gravity height (m)
    :ivar tires: the Magic Formula tire of the front and the rear axle (see
        :func:`tire_models`)
    :ivar cornering_stiffness: per tire at the static load, front and rear
        (N/rad)
    :ivar mu_axle: lateral peak friction at the static load, front and rear
    :ivar mu: their mean
    :ivar relaxation_length: of the slip angle (m), 0 for none
    """

    def __init__(self, data, base_dir=None):
        self.raw = copy.deepcopy(data)
        self.name = data["name"]
        self.source = data.get("source", "")

        def corner(ref):
            if isinstance(ref, dict):
                return CornerModel(ref)
            path = Path(ref)
            if base_dir is not None and not path.is_absolute() \
                    and (Path(base_dir) / path).exists():
                path = Path(base_dir) / path
            return CornerModel.load(path if path.exists() else ref)

        self.wheelbase = data["wheelbase"] * MM
        setups = data.get("setup", {})
        self.front = tuning.apply(corner(data["front"]),
                                  setups.get("front", {}))
        self.rear = tuning.apply(corner(data["rear"]),
                                 setups.get("rear", {})).translated(
            [-self.wheelbase, 0.0, 0.0])
        self.sprung_mass = float(data["sprung_mass"])
        self.front_fraction = float(data["front_fraction"])
        if not 0.0 < self.front_fraction < 1.0:
            raise ValueError("front_fraction must be between 0 and 1")
        self.cg_height = data["cg_height"] * MM
        self.sprung_inertia = data.get("sprung_inertia",
                                       [500.0, 2000.0, 2200.0])
        #: Rack travel (m) per radian of the steering wheel, or None.
        self.rack_ratio = None
        if data.get("rack_ratio") is not None:
            self.rack_ratio = math.degrees(float(data["rack_ratio"]) * MM)
        tire = data.get("tire", {})
        self.relaxation_length = float(tire.get("relaxation_length", 0.0))
        static = [0.5 * w for w in axle_loads(self)]
        self.tires = tire_models(tire, static)
        # The values at the static wheel loads: the cornering stiffness
        # (N/rad) of one tire and the lateral friction of each axle (front,
        # rear); ``mu`` is their mean.
        self.cornering_stiffness = [
            t.cornering_stiffness(fz) for t, fz in zip(self.tires, static)]
        self.mu_axle = [t.friction(fz)[1]
                        for t, fz in zip(self.tires, static)]
        self.mu = float(np.mean(self.mu_axle))

    def tire(self, front):
        """The :class:`MagicFormulaTire` of the front or the rear tires."""
        return self.tires[0 if front else 1]

    def with_setup(self, front=None, rear=None):
        """A copy with the setups (see :mod:`suspension.tuning`) applied on
        top of the current front and rear corners."""
        vm = copy.deepcopy(self)
        if front:
            vm.front = tuning.apply(vm.front, front)
        if rear:
            vm.rear = tuning.apply(vm.rear, rear)
        return vm

    @classmethod
    def load(cls, name_or_path):
        """Load ``suspension/data/vehicles/<name>.json`` or a path."""
        path = Path(name_or_path)
        if not path.exists():
            path = DATA_DIR / f"{name_or_path}.json"
        with open(path) as fobj:
            return cls(json.load(fobj), base_dir=path.parent)

    @property
    def cg_x(self):
        """Longitudinal CG position (m): the front axle is at x = 0."""
        return -(1.0 - self.front_fraction) * self.wheelbase

    def corner_sprung_mass(self, front):
        """The sprung mass (kg) one corner of an axle carries."""
        share = self.front_fraction if front else 1.0 - self.front_fraction
        return 0.5 * share * self.sprung_mass

    def track(self, front):
        model = self.front if front else self.rear
        return 2.0 * abs(float(model.hp["wheel_center"][1]))


def _preloaded(mbs, model, sprung_mass, gravity):
    """The corner model whose spring holds the design pose under the
    sprung mass (the quarter car finds its free length)."""
    _, handles = vehicle.quarter_car(mbs, model, sprung_mass,
                                     gravity=gravity)
    return handles["model"]


def _unsprung_mass(model):
    """The unsprung mass of a corner: the upright (with the wheel) and the
    arms."""
    return model.upright_mass + sum(a["mass"] for a in model.arms.values())


def _corner_mass(model):
    """Every corner body: the unsprung mass, the rocker, and the rack (the
    vehicle sprung mass does not hold them)."""
    return _unsprung_mass(model) + model.rocker_mass + model.rack_mass


def _stop_lengths(mbs, model):
    """The spring-line lengths of the stops of a corner model, cached on
    the model (a mirrored corner has the same lengths)."""
    key = (model.stops["bump_travel"], model.stops["droop_travel"])
    cached = getattr(model, "_stop_cache", None)
    if cached is None or cached[0] != key:
        cached = (key, vehicle.travel_stop_lengths(mbs, model))
        model._stop_cache = cached
    return cached[1]


def corner_models(mbs, vm, gravity=9.81):
    """``{corner: model}`` of the four preloaded corner models."""
    front = _preloaded(mbs, vm.front, vm.corner_sprung_mass(True), gravity)
    rear = _preloaded(mbs, vm.rear, vm.corner_sprung_mass(False), gravity)
    return {"front_left": front, "front_right": front.mirrored(),
            "rear_left": rear, "rear_right": rear.mirrored()}


def build(mbs, vm, lateral_acceleration=0.0, gravity=9.81, side_forces=None,
          steer_func=None, dt=1e-3, speed=0.0, with_slip=False, models=None,
          support=(SOFT_SUPPORT, SOFT_SUPPORT), support_damping=(0.0, 0.0),
          patch_support=None):
    """Build the vehicle at its design pose.

    :param lateral_acceleration: a_y (m/s^2) of the D'Alembert load
    :param side_forces: not None for the steady-state rig: a soft support
        of the chassis in x, y, and yaw (the side forces are applied loads,
        see :func:`steady_state`)
    :return: (system, handles); the handles list ``bodies``, ``(number,
        mass)`` of every body with a D'Alembert load
    :param steer_func: rack drive functions of the front corners, (left,
        right): the right one is the negative of the left one, since each
        corner has its own rack body along its mirrored rack axis (as the
        axle rig drives them)
    :param speed: initial forward speed (m/s) of every body
    :param with_slip: tires with the lateral slip force (C_alpha, mu) for a
        dynamic run
    :param models: the preloaded corner models of :func:`corner_models`
    :param support: (N/m, N m/rad) stiffness of the chassis support in x
        and y and in yaw
    :param support_damping: (N s/m, N m s/rad) damping of the chassis
        support in x and y and in yaw, for a dynamic run on the support
    :param patch_support: None, or the lateral stiffness (N/m) and damping
        (N s/m) of each tire at its contact patch, front and rear: ((k_f,
        c_f), (k_r, c_r)); it holds the patch on the road sideways (the
        tire carcass and its slip) for a rig without slip forces
    :return: (system, handles) with ``corners``, ``tires``, ``models``,
        ``chassis``, ``arbs``
    """
    models = models or corner_models(mbs, vm, gravity)
    field = [0.0, -float(lateral_acceleration), -float(gravity)]
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    cg = np.array([vm.cg_x, 0.0, vm.cg_height])
    vel = [float(speed), 0.0, 0.0]
    chassis = mbs.create_mobilized_body_quat(
        1, list(cg), vel, Z, [1, 0, 0, 0], Z, Z, vm.sprung_mass,
        list(vm.sprung_inertia),
        [vm.sprung_mass * a for a in field], Z)
    s.Add(chassis)
    out = {"corners": {}, "tires": {}, "models": models,
           "chassis": chassis, "arbs": {}, "cg": cg,
           "bodies": [(1, vm.sprung_mass)]}
    num = 2
    for name in CORNERS:
        model = models[name]
        front = name.startswith("front")
        c = corner_mod.add_corner(
            mbs, s, model, ground, chassis, cg, num, travel_func=None,
            steer_func=(steer_func[name.endswith("right")]
                        if front and steer_func else None), gravity=field,
            velocity=vel)
        out["corners"][name] = c
        if model.stops is not None:
            # The bump and droop stops on the spring line, as in the
            # quarter car (:func:`vehicle.quarter_car`).
            lengths = _stop_lengths(mbs, model)
            st = model.stops
            stop = mbs.create_length_stop(
                c.body(c.spring_body), chassis, list(c.spring_points[0]),
                list(c.spring_points[1]), min_length=lengths[0],
                max_length=lengths[1], k=st["k"], k2=st["k2"], c=st["c"])
            s.Add(stop)
            out.setdefault("stops", {})[name] = stop
        masses = {"upright": model.upright_mass, "rocker": model.rocker_mass,
                  "rack": model.rack_mass}
        masses.update({"arm_" + arm: spec["mass"]
                       for arm, spec in model.arms.items()})
        out["bodies"] += [(c.nums[key], masses[key]) for key in c.nums]
        num += c.nbody
        weight = (vm.corner_sprung_mass(front)
                  + model.upright_mass + model.rocker_mass + model.rack_mass
                  + sum(a["mass"] for a in model.arms.values())) * gravity
        deflection = weight / model.tire_k
        tire = mbs.create_tire(ground, c.upright, list(model.spin_axis),
                               model.wheel_radius + deflection, model.tire_k,
                               model.tire_c, road_z=0.0,
                               magic_formula=(vm.tire(front) if with_slip
                                              else None),
                               relaxation_length=vm.relaxation_length)
        s.Add(tire)
        out["tires"][name] = tire
        if patch_support is not None:
            k_y, c_y = patch_support[0 if front else 1]
            wc = np.asarray(model.hp["wheel_center"], dtype="float64")
            spin = np.asarray(model.spin_axis, dtype="float64")
            down = np.array([0.0, 0.0, -1.0]) + spin[2] * spin
            down /= np.linalg.norm(down)
            patch = wc + (wc[2] / -down[2]) * down
            k6, c6 = np.zeros((6, 6)), np.zeros((6, 6))
            k6[1, 1], c6[1, 1] = float(k_y), float(c_y)
            bush = mbs.create_bushing(ground, c.upright, list(patch),
                                      list(patch - wc), k6.tolist(),
                                      c6.tolist())
            s.Add(bush)
            out.setdefault("patch_supports", {})[name] = bush
    for axle_name in ("front", "rear"):
        left = models[f"{axle_name}_left"]
        if left.arb is None:
            continue
        halves = [axle._add_arb_half(mbs, s, models[f"{axle_name}_{side}"],
                                     chassis, out["corners"][
                                         f"{axle_name}_{side}"], num + k,
                                     origin=cg, velocity=vel)
                  for k, side in enumerate(("left", "right"))]
        out.setdefault("arb_nums", {})[axle_name] = (num, num + 1)
        num += 2
        ax = list(left.arb["axis"])
        s.Add(mbs.create_rsda(halves[0], halves[1], ax, ax,
                              k=left.arb["k"]))
        out["arbs"][axle_name] = halves
    if side_forces is not None:
        # The side forces balance the D'Alembert load only at the fixed
        # point of steady_state(); this soft support holds the free chassis
        # in x, y, and yaw on the way there.
        k = np.zeros((6, 6))
        k[0, 0] = k[1, 1] = float(support[0])
        k[5, 5] = float(support[1])
        c = np.zeros((6, 6))
        c[0, 0] = c[1, 1] = float(support_damping[0])
        c[5, 5] = float(support_damping[1])
        out["chassis_support"] = mbs.create_bushing(
            ground, chassis, list(cg), Z, k.tolist(), c.tolist())
        s.Add(out["chassis_support"])
    s.init()
    return s, out


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def vehicle_pose(s, h, follow=False):
    """The pose of every body of the vehicle for a 3D view.

    :param follow: put the chassis back over the origin, heading +x (a
        camera that follows the car: the roll, pitch, heave, and the wheel
        angles stay, the path does not)
    :return: dict with ``chassis`` (position m, quaternion), ``cg`` (the
        design position of the chassis frame origin, m), ``corners``
        (``{corner: {body: (position, quaternion)}}``), and ``arbs``
        (``{axle: [left half, right half]}``)
    """
    from .metrics import rot
    pos = np.asarray(s.get_body_positions(), dtype="float64")
    quat = np.asarray(s.get_body_quaternions(), dtype="float64")

    def place(num):
        return pos[num].copy(), quat[num].copy()

    turn = np.array([1.0, 0.0, 0.0, 0.0])
    now = design = np.zeros(3)
    if follow:
        # Turn about z through the chassis origin and put it back over its
        # design position.
        heading = rot(quat[1]) @ np.array([1.0, 0.0, 0.0])
        yaw = math.atan2(heading[1], heading[0])
        turn = np.array([math.cos(-0.5 * yaw), 0.0, 0.0,
                         math.sin(-0.5 * yaw)])
        now = np.array([pos[1][0], pos[1][1], 0.0])
        design = np.array([h["cg"][0], h["cg"][1], 0.0])
    r_turn = rot(turn)

    def take(num):
        p, q = place(num)
        return r_turn @ (p - now) + design, _quat_mul(turn, q)

    out = {"chassis": take(1), "cg": np.array(h["cg"], dtype="float64"),
           "corners": {}, "arbs": {}}
    for name, c in h["corners"].items():
        out["corners"][name] = {body: take(num)
                                for body, num in c.nums.items()}
    for axle_name, nums in h.get("arb_nums", {}).items():
        out["arbs"][axle_name] = [take(nums[0]), take(nums[1])]
    return out


def wheel_angle(spin):
    """The heading (rad, + to the left) of a wheel plane with the global
    spin axis ``spin``: the forward direction z x s, or its opposite."""
    fwd = np.cross([0.0, 0.0, 1.0], np.asarray(spin, dtype="float64"))
    if fwd[0] < 0.0:
        fwd = -fwd
    return float(math.atan2(fwd[1], fwd[0]))


def _roll_angle(quat):
    from .metrics import rot
    r = rot(np.asarray(quat, dtype="float64"))
    return float(math.atan2(r[2, 1], r[2, 2]))


def steady_state(mbs, vm, accelerations_g=(0.0, 0.1, 0.2, 0.3),
                 gravity=9.81, rtol=1e-4, partial=False):
    """The vehicle in steady turns (left, a_y > 0) at lateral
    accelerations given in g.

    Each tire carries the side force F_y = F_z a_y / g (every tire uses the
    same share of its load, the quasi-static assumption of the roll center
    model): the side forces then sum to the D'Alembert load and hold the
    yaw moment.  A fixed-point iteration finds the loads F_z; the turns are
    solved in the given order, each from the last equilibrium.

    :return: dict of arrays over the accelerations: ``a_y`` (g), ``roll``
        (rad, + to the outside of the turn), ``loads`` (N, 4 corners in
        :data:`CORNERS` order), ``load_transfer`` (front, rear: half the
        outer minus inner load, N), ``steer`` (rad, toe of each corner
        from design, + to the left), ``wheel_angle`` (rad, the heading of
        each wheel plane to the vehicle, with the static toe, + to the
        left), ``side_force`` (front, rear, N),
        ``support`` (largest chassis support force, N), and the gradients
        ``roll_gradient`` (deg/g), ``load_transfer_gradient`` (N/g),
        ``lltd`` (front share of the load transfer), ``roll_steer`` (front,
        rear axle steer per g, rad/g, + into the turn: the slope of the
        wheel angles weighted by the cornering stiffness at the wheel
        loads, so it holds the static toe under load transfer); with
        ``partial``
        the turns up to the first one the rig cannot balance, and
        ``failed_at`` (g) that one (None when all are solved); ``poses``,
        the :func:`vehicle_pose` of each turn, and ``models``, the corner
        models
    """
    from .metrics import rot
    models = corner_models(mbs, vm, gravity)
    s, h = build(mbs, vm, 0.0, gravity,
                 side_forces={name: 0.0 for name in CORNERS}, models=models)
    uprights = {h["corners"][name].nums["upright"]: name
                for name in CORNERS}

    def apply(a, side):
        field = np.array([0.0, -a * gravity, -gravity])
        pos = np.asarray(s.get_body_positions())
        quat = np.asarray(s.get_body_quaternions())
        for num, mass in h["bodies"]:
            force = mass * field
            torque = np.zeros(3)
            if num in uprights:
                name = uprights[num]
                fy = np.array([0.0, side[name], 0.0])
                # The contact patch: down the wheel plane from the wheel
                # center to the road plane z = 0 (the tire deflects).
                spin = rot(quat[num]) @ models[name].spin_axis
                down = np.array([0.0, 0.0, -1.0]) + spin[2] * spin
                down /= np.linalg.norm(down)
                lever = pos[num][2] / -down[2]
                force = force + fy
                torque = np.cross(lever * down, fy)
            s.set_applied_load(num, list(force), list(torque))

    weights = [(vm.corner_sprung_mass(name.startswith("front"))
                + _corner_mass(models[name])) * gravity for name in CORNERS]
    guess = weights
    rows = {"roll": [], "loads": [], "steer": [], "side_force": [],
            "support": [], "wheel_angle": []}
    poses = []

    def settle(a):
        """The fixed point at a; returns the loads and the side forces."""
        loads = None
        for _ in range(30):
            side = {name: (guess if loads is None else loads)[k] * a
                    for k, name in enumerate(CORNERS)}
            apply(a, side)
            res = s.find_equilibrium(tol=1e-10)
            if not res["converged"]:
                raise RuntimeError(f"vehicle: no equilibrium at {a:g} g: "
                                   f"{res['message']}")
            new = [float(mbs.get_tire_state(h["tires"][name])
                         ["normal_force"]) for name in CORNERS]
            # The equilibrium solve settles the loads to about 1e-5 of
            # their size; the fixed point stops at rtol.
            done = loads is not None and \
                max(abs(x - y) for x, y in zip(new, loads)) \
                < rtol * max(new)
            loads = new
            if done:
                return loads, side
        raise RuntimeError(f"vehicle: the side forces did not converge at "
                           f"{a:g} g")

    last = 0.0
    failed_at = None
    solved = []
    for a in accelerations_g:
        # Continuation in steps of at most MAX_STEP_G from the last turn.
        count = max(1, int(math.ceil(abs(a - last) / MAX_STEP_G)))
        try:
            for step in range(1, count + 1):
                loads, side = settle(last + (a - last) * step / count)
                guess = loads
        except RuntimeError:
            if not partial:
                raise
            failed_at = a
            break
        last = a
        solved.append(a)
        guess = loads
        quat = np.asarray(s.get_body_quaternions())
        rows["roll"].append(_roll_angle(quat[1]))
        steer, wheel = [], []
        for name in CORNERS:
            c = h["corners"][name]
            r_up = rot(quat[c.nums["upright"]])
            fwd = r_up @ np.array([1.0, 0.0, 0.0])
            steer.append(float(math.atan2(fwd[1], fwd[0])))
            wheel.append(wheel_angle(r_up @ models[name].spin_axis))
        rows["loads"].append(loads)
        rows["steer"].append(steer)
        rows["wheel_angle"].append(wheel)
        rows["side_force"].append([side["front_left"] + side["front_right"],
                                   side["rear_left"] + side["rear_right"]])
        sup = mbs.get_force_element_loads(h["chassis_support"])
        rows["support"].append(float(np.max(np.abs(sup["force_j"]))))
        poses.append(vehicle_pose(s, h))
    out = {key: np.array(val) for key, val in rows.items()}
    out["a_y"] = np.asarray(solved, dtype="float64")
    out["failed_at"] = failed_at
    out["poses"] = poses
    out["models"] = models
    loads = out["loads"]
    out["load_transfer"] = np.column_stack(
        [0.5 * (loads[:, 1] - loads[:, 0]), 0.5 * (loads[:, 3] - loads[:, 2])])
    out["steer"] = out["steer"] - out["steer"][0]
    ay = out["a_y"]
    if ay.size > 1:
        out["roll_gradient"] = float(np.polyfit(ay, out["roll"], 1)[0]) / DEG
        dw = [float(np.polyfit(ay, out["load_transfer"][:, k], 1)[0])
              for k in (0, 1)]
        out["load_transfer_gradient"] = dw
        out["lltd"] = dw[0] / (dw[0] + dw[1])
        # The axle steer is the mean wheel angle weighted by the cornering
        # stiffness at the wheel load: the axle side force is sum K_i (x +
        # psi_i).  With a load-sensitive tire the static toe of an axle
        # then steers it under load transfer (the outer tire gains
        # stiffness, the inner one loses it).
        angles = out["wheel_angle"]
        axle_steer = []
        for k, front in ((0, True), (2, False)):
            tire = vm.tire(front)
            w = np.array([[tire.cornering_stiffness(max(float(fz), 0.0))
                           for fz in row[k:k + 2]] for row in loads])
            mean = (w * angles[:, k:k + 2]).sum(axis=1) / w.sum(axis=1)
            axle_steer.append(mean - mean[0])
        axle_steer = np.column_stack(axle_steer)
        out["roll_steer"] = [float(np.polyfit(ay, axle_steer[:, k], 1)[0])
                             for k in (0, 1)]
    return out


def closed_form(mbs, vm, gravity=9.81):
    """The roll center model of the vehicle (Milliken Ch. 18): roll
    gradient (deg/g), load transfer per g of each axle (N/g), and LLTD,
    from the axle roll stiffness of the K&C axle rig and the kinematic roll
    center heights.

    :return: dict with the inputs (``roll_stiffness`` front and rear, N
        m/rad, suspension in series with the tires; ``roll_center``, m;
        ``h_prime``, m) and ``roll_gradient``, ``load_transfer_gradient``,
        ``lltd``
    """
    models = corner_models(mbs, vm, gravity)
    w_s = vm.sprung_mass * gravity
    length = vm.wheelbase
    a = -vm.cg_x
    stiff, rc, unsprung, tracks = [], [], [], []
    for axle_name in ("front", "rear"):
        left = models[f"{axle_name}_left"]
        track = vm.track(axle_name == "front")
        with_arb = left.arb is not None
        sweep = axle.roll_sweep(mbs, left, [-2e-3, 0.0, 2e-3], with_arb,
                                offset=left.static_travel)
        k_susp = float(sweep["roll_stiffness"][1])
        k_tire = left.tire_k * track ** 2 / 2.0
        stiff.append(k_susp * k_tire / (k_susp + k_tire))
        rc.append(float(kc.bump_sweep(
            mbs, left, [left.static_travel])["roll_center"][0]))
        unsprung.append((2.0 * _unsprung_mass(left) * gravity,
                         float(left.hp["wheel_center"][2])))
        tracks.append(track)
    h_axis = rc[0] + (rc[1] - rc[0]) * a / length
    h_prime = vm.cg_height - h_axis
    k_sum = stiff[0] + stiff[1]
    denom = k_sum - w_s * h_prime
    roll = w_s * h_prime / denom
    w_sf = w_s * vm.front_fraction
    w_sr = w_s - w_sf
    dw_f = (w_s * h_prime * stiff[0] / denom + w_sf * rc[0]
            + unsprung[0][0] * unsprung[0][1]) / tracks[0]
    dw_r = (w_s * h_prime * stiff[1] / denom + w_sr * rc[1]
            + unsprung[1][0] * unsprung[1][1]) / tracks[1]
    return {"roll_stiffness": stiff, "roll_center": rc, "h_prime": h_prime,
            "roll_gradient": roll / DEG,
            "load_transfer_gradient": [dw_f, dw_r],
            "lltd": dw_f / (dw_f + dw_r)}


def understeer_gradient(vm, roll_steer=(0.0, 0.0), gravity=9.81):
    """Gillespie's understeer gradient (rad/g): W_f / C_f - W_r / C_r minus
    the roll steer into the turn, front minus rear, per g.

    :param roll_steer: front and rear axle steer per g (rad/g, + into the
        turn), for example from :func:`steady_state`
    """
    w_f, w_r = axle_loads(vm, gravity)
    c_f, c_r = (2.0 * c for c in vm.cornering_stiffness)
    return w_f / c_f - w_r / c_r - (roll_steer[0] - roll_steer[1])


def step_steer(mbs, vm, speed, rack, start=0.5, ramp=0.2, duration=3.0,
               dt=1e-3, gravity=9.81, pose_dt=0.02):
    """A ramp-step steer at constant speed (ISO 7401 style): the vehicle
    drives straight, the rack moves by ``rack`` (m) over ``ramp`` seconds
    from ``start``, and the run lasts ``duration`` seconds.  The tires
    carry the slip forces of :attr:`VehicleModel.tires`.

    :return: dict with ``time``, ``yaw_rate`` (rad/s), ``speed`` (m/s),
        ``lateral_acceleration`` (m/s^2, the chassis acceleration normal
        to its heading in the road plane; V r only in a steady turn),
        ``sideslip`` (rad, the chassis velocity angle to its heading),
        ``normal_force`` (N, (n, 4) in :data:`CORNERS` order),
        ``wheel_angle`` (rad, (n, 4), each wheel plane to the chassis, + to
        the left), ``steer`` (rad, the front and rear axle steer: the
        wheel angles weighted by the cornering stiffness at the wheel
        loads, see :func:`steady_state`), and the means over the last 0.5
        s: ``steady`` dict of the same keys; ``poses``
        (:func:`vehicle_pose`, following the car) every ``pose_dt`` seconds
        at ``pose_times``, the corner ``models``, and the
        :func:`plausibility` ``checks`` with ``plausible``
    """
    times = [0.0, start, start + ramp, duration + 1.0]
    values = [0.0, 0.0, float(rack), float(rack)]
    steer = (mbs.DriveFunction.table(times, values),
             mbs.DriveFunction.table(times, [-v for v in values]))
    s, h = build(mbs, vm, 0.0, gravity, steer_func=steer, dt=dt,
                 speed=speed, with_slip=True)
    s.set_integrator("rk45")
    s.set_tolerances(1e-7, 1e-10)
    from .metrics import rot
    rec = {"time": [], "yaw_rate": [], "speed": [], "steer": [],
           "sideslip": [], "normal_force": [], "wheel_angle": [],
           "velocity": [], "lateral": []}
    poses, pose_times = [], []
    every = max(1, int(round(pose_dt / dt)))
    tires = [h["tires"][name] for name in CORNERS]
    for step in range(int(round(duration / dt))):
        s.solve()
        if vm.relaxation_length > 0.0:
            for tire in tires:
                mbs.commit_tire(tire)
        if 0 == step % every:
            poses.append(vehicle_pose(s, h, follow=True))
            pose_times.append(s.get_sim_time())
        quat = np.asarray(s.get_body_quaternions())
        vel = np.asarray(s.get_body_velocities())
        omega = np.asarray(s.get_body_angular_velocities())
        r_ch = rot(quat[1])
        heading = r_ch @ np.array([1.0, 0.0, 0.0])
        yaw = math.atan2(heading[1], heading[0])
        angles = []
        for name in CORNERS:
            c = h["corners"][name]
            # The heading of the wheel plane (the spin axis carries the
            # static toe).
            spin = rot(quat[c.nums["upright"]]) @ np.asarray(
                h["models"][name].spin_axis, dtype="float64")
            angles.append(math.remainder(wheel_angle(spin) - yaw,
                                         2.0 * math.pi))
        lateral = np.array([-math.sin(yaw), math.cos(yaw)])
        v_xy = np.asarray(vel[1][:2], dtype="float64")
        rec["time"].append(s.get_sim_time())
        rec["yaw_rate"].append(float(omega[1][2]))
        rec["speed"].append(float(np.linalg.norm(v_xy)))
        loads = [float(mbs.get_tire_state(h["tires"][name])["normal_force"])
                 for name in CORNERS]
        axle_steer = []
        for k, front in ((0, True), (2, False)):
            w = [vm.tire(front).cornering_stiffness(fz)
                 for fz in loads[k:k + 2]]
            total = w[0] + w[1]
            axle_steer.append(
                (w[0] * angles[k] + w[1] * angles[k + 1]) / total
                if total > 0.0 else 0.5 * (angles[k] + angles[k + 1]))
        rec["steer"].append(axle_steer)
        rec["wheel_angle"].append(angles)
        rec["sideslip"].append(math.atan2(float(v_xy @ lateral),
                                          float(v_xy @ heading[:2])))
        rec["normal_force"].append(loads)
        rec["velocity"].append(v_xy)
        rec["lateral"].append(lateral)
    out = {key: np.array(val) for key, val in rec.items()}
    # The chassis acceleration (central differences of the velocity)
    # normal to the heading.
    accel = np.gradient(out.pop("velocity"), out["time"], axis=0)
    out["lateral_acceleration"] = np.einsum("ij,ij->i", accel,
                                            out.pop("lateral"))
    out["poses"] = poses
    out["pose_times"] = np.array(pose_times)
    out["models"] = h["models"]
    tail = out["time"] >= out["time"][-1] - 0.5
    out["steady"] = {key: np.mean(out[key][tail], axis=0) for key in
                     ("yaw_rate", "speed", "lateral_acceleration", "steer",
                      "sideslip")}
    out["checks"] = plausibility(out, vm.mu, gravity)
    out["plausible"] = not any("invalid" == c["level"]
                               for c in out["checks"])
    return out


#: The sideslip (rad) past which a run has left the linear handling range:
#: a car beyond 10 deg is spinning (Milliken, RCVD, Ch. 5, beta-method).
SPIN_SIDESLIP = math.radians(10.0)
#: The steady yaw rate spread (std / |mean| over the last 0.5 s) of a
#: settled run.
SETTLED_SPREAD = 0.05


def plausibility(run, mu, gravity=9.81):
    """Physical checks of a dynamic vehicle run.

    * ``invalid``: a non-finite state; a sideslip past
      :data:`SPIN_SIDESLIP` (the car spun); a lateral acceleration above
      1.1 mu g (more than the tires can give).
    * ``warning``: a wheel off the road (F_z = 0); a yaw rate that has not
      settled in the last 0.5 s (the ``steady`` values are not steady).

    :param run: a run of :func:`step_steer`
    :type run: dict
    :param mu: the tire friction coefficient
    :type mu: float
    :return: the checks, each a dict with ``level`` (``invalid`` or
        ``warning``), ``key``, and ``message``
    :rtype: list
    """
    out = []

    def add(level, key, message):
        out.append({"level": level, "key": key, "message": message})

    for key in ("yaw_rate", "speed", "lateral_acceleration", "sideslip"):
        if not np.isfinite(run[key]).all():
            add("invalid", "finite", f"{key} is not finite: the run "
                "diverged")
            return out
    beta = float(np.max(np.abs(run["sideslip"])))
    if beta > SPIN_SIDESLIP:
        add("invalid", "spin", f"the car spun: sideslip "
            f"{math.degrees(beta):.1f} deg > "
            f"{math.degrees(SPIN_SIDESLIP):.0f} deg")
    ay = float(np.max(np.abs(run["lateral_acceleration"])))
    if ay > 1.1 * mu * gravity:
        add("invalid", "friction", f"lateral acceleration {ay / gravity:.2f}"
            f" g exceeds the tire friction {mu:.2f} (by more than 10 %)")
    fz = np.asarray(run["normal_force"])
    if fz.size:
        lifted = [CORNERS[k] for k in range(fz.shape[1])
                  if np.min(fz[:, k]) <= 0.0]
        if lifted:
            add("warning", "lift", "wheel off the road: "
                + ", ".join(lifted))
    tail = run["time"] >= run["time"][-1] - 0.5
    r = np.asarray(run["yaw_rate"])[tail]
    if abs(np.mean(r)) > 1e-6 and \
            np.std(r) > SETTLED_SPREAD * abs(np.mean(r)):
        add("warning", "settle", "the yaw rate has not settled in the last "
            f"0.5 s (spread {np.std(r) / abs(np.mean(r)) * 100:.0f} %); "
            "the steady values are not steady")
    return out


def kinematic_steer(mbs, vm, rack, gravity=9.81):
    """The mean front wheel angle (rad, + to the left) of a rack travel
    (m) with the vehicle at rest on level ground (no roll)."""
    from .metrics import rot
    times = [0.0, 1.0]
    steer = (mbs.DriveFunction.table(times, [rack, rack]),
             mbs.DriveFunction.table(times, [-rack, -rack]))
    s, h = build(mbs, vm, 0.0, gravity,
                 side_forces={name: 0.0 for name in CORNERS},
                 steer_func=steer)
    if not s.find_equilibrium(tol=1e-10)["converged"]:
        raise RuntimeError("vehicle: no equilibrium with the rack moved")
    quat = np.asarray(s.get_body_quaternions())
    heading = rot(quat[1]) @ np.array([1.0, 0.0, 0.0])
    yaw = math.atan2(heading[1], heading[0])
    out = []
    for name in ("front_left", "front_right"):
        c = h["corners"][name]
        fwd = rot(quat[c.nums["upright"]]) @ np.array([1.0, 0.0, 0.0])
        out.append(math.atan2(fwd[1], fwd[0]) - yaw)
    s0, h0 = build(mbs, vm, 0.0, gravity,
                   side_forces={name: 0.0 for name in CORNERS})
    s0.find_equilibrium(tol=1e-10)
    quat0 = np.asarray(s0.get_body_quaternions())
    rest = []
    for name in ("front_left", "front_right"):
        c = h0["corners"][name]
        fwd = rot(quat0[c.nums["upright"]]) @ np.array([1.0, 0.0, 0.0])
        rest.append(math.atan2(fwd[1], fwd[0]))
    return 0.5 * (out[0] - rest[0] + out[1] - rest[1])


def peak_slip_angle(tire):
    """The slip angle (rad) of the lateral peak of a Magic Formula tire:
    C atan(B s - E (B s - atan(B s))) = pi / 2 with s = tan(alpha); pi / 2
    for a curve without a peak (C <= 1)."""
    lat = tire.lateral
    b, c, e = lat["stiffness"], lat["shape"], lat["curvature"]
    if c <= 1.0:
        return 0.5 * math.pi

    def inner(x):
        return x - e * (x - math.atan(x))

    target = math.tan(0.5 * math.pi / c)
    lo, hi = 0.0, 1.0
    while inner(hi) < target:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if inner(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.atan(0.5 * (lo + hi) / b)


def tire_side_force(vm, front, slip, load):
    """The side force (N, + to the left for a slip + to the left) of the
    tire of the axle (:attr:`VehicleModel.tires`, the Magic Formula of the
    engine tire) at pure side slip.  Past the peak the force stays at the
    peak: a steady state does not use the falling branch.
    """
    load = max(float(load), 0.0)
    if load <= 0.0:
        return 0.0
    tire = vm.tire(front)
    angle = min(abs(slip), peak_slip_angle(tire))
    return math.copysign(-tire.force(load, 0.0, angle)[1], slip)


def _axle_capacity(vm, front, loads):
    """The largest side force (N) of an axle: the lateral peaks."""
    tire = vm.tire(front)
    return sum(tire.friction(max(float(fz), 0.0))[1] * max(float(fz), 0.0)
               for fz in loads)


def _axle_slip(vm, front, force, loads, angles):
    """The axle slip x (rad) with sum F(x + wheel angle, F_z) = ``force``,
    or None past the grip of the axle."""
    def total(x):
        return sum(tire_side_force(vm, front, x + psi, fz)
                   for fz, psi in zip(loads, angles))

    hi = peak_slip_angle(vm.tire(front)) + max(abs(a) for a in angles)
    if total(hi) < force:
        return None
    lo = -hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if total(mid) < force:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def handling_diagram(mbs, vm, accelerations_g=None, gravity=9.81):
    """The steady-state handling of the vehicle with the nonlinear tire
    (Pacejka 2012, Sec. 1.3; Milliken & Milliken, RCVD Ch. 5).

    At each lateral acceleration the steady-state rig gives the wheel loads
    and the wheel angles (static toe and roll steer).  Each axle needs the
    side force W a_y; its slip x solves sum F(x + psi_i, F_zi) = W a_y.  The
    understeer angle is delta - L / R = x_f - x_r; its slope at a_y -> 0 is
    the understeer gradient.  The first axle that cannot carry its force
    sets the limit: the front (the car pushes) or the rear (it spins).

    :return: dict with ``a_y`` (g), ``understeer_angle`` (rad), ``slip``
        (front, rear, rad), ``grip_use`` (front, rear: the side force over
        the peak force of the axle), ``limit`` (g, where the busier axle
        reaches its grip), ``limited_by`` ("front": the car pushes, "rear":
        it spins, "lift-off": an inner wheel leaves the road first), and
        the steady state ``steady``
    """
    if accelerations_g is None:
        accelerations_g = list(np.round(np.arange(0.0, 1.21, 0.05), 3))
    w_f, w_r = axle_loads(vm, gravity)
    # The rig marches by continuation and stops where it cannot balance
    # the turn (a wheel off the ground).
    steady = steady_state(mbs, vm, accelerations_g, gravity, partial=True)
    ays, under, slips, usage = [], [], [], []
    limited_by = None
    for k, a in enumerate(steady["a_y"]):
        loads = steady["loads"][k]
        angles = steady["wheel_angle"][k]
        # The grip each axle uses: its side force over the sum of the peak
        # forces mu(F_z) F_z of its tires.
        use = (w_f * a / _axle_capacity(vm, True, loads[:2]),
               w_r * a / _axle_capacity(vm, False, loads[2:]))
        x_f = _axle_slip(vm, True, w_f * a, loads[:2], angles[:2])
        x_r = _axle_slip(vm, False, w_r * a, loads[2:], angles[2:])
        if x_f is None or x_r is None:
            limited_by = "front" if x_f is None else "rear"
            usage.append(use)
            break
        ays.append(float(a))
        under.append(x_f - x_r)
        slips.append((x_f, x_r))
        usage.append(use)
    if limited_by is None and steady.get("failed_at") is not None:
        limited_by = "lift-off"
    # The limit: where the busier axle reaches its grip, between the last
    # two accelerations (the grip use is near linear there).
    limit = ays[-1] if ays else 0.0
    if len(usage) >= 2:
        busy = [max(u) for u in usage]
        a_all = list(steady["a_y"][:len(usage)])
        if busy[-1] >= 1.0 > busy[-2]:
            frac = (1.0 - busy[-2]) / (busy[-1] - busy[-2])
            limit = a_all[-2] + frac * (a_all[-1] - a_all[-2])
        if limited_by is None and busy[-1] >= 1.0:
            limited_by = "front" if usage[-1][0] >= usage[-1][1] \
                else "rear"
    return {"a_y": np.array(ays), "understeer_angle": np.array(under),
            "slip": np.array(slips),
            "grip_use": np.array(usage[:len(ays)]), "limit": limit,
            "limited_by": limited_by, "steady": steady}


def axle_loads(vm, gravity=9.81):
    """Static axle loads (N) with the corner bodies: (front, rear)."""
    w = vm.sprung_mass * gravity
    w_f = w * vm.front_fraction + 2.0 * _corner_mass(vm.front) * gravity
    w_r = w * (1.0 - vm.front_fraction) \
        + 2.0 * _corner_mass(vm.rear) * gravity
    return w_f, w_r


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
