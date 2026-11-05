# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Quarter car on a one-sided tire (roadmap P9.7, P9.9, P9.10).

* chassis: quarter sprung mass on a vertical prismatic joint to ground;
* corner: as in suspension.corner, attached to the chassis, no travel driver;
* stops: a bump stop and a droop stop on the spring line (model.stops),
  so the wheel stays in its travel range;
* tire: the engine point-contact tire on the upright. It pushes along the
  normal of the road plane only in compression, so the wheel can leave the
  road. The road plane is the effective road of a profile driven over at a
  speed (suspension.road.road_input, with tire enveloping), or a shaker
  plate moving under the tire (suspension.road.rig_input, or a drive
  function road_func of the height).

The static equilibrium is the design pose: the spring free length gives a
spring force W / MR at design (W the sprung weight), and the tire free
radius is the loaded radius plus the deflection that carries the total
weight.
"""

import numpy as np

from ..tools.arrays import result_arrays
from . import corner as corner_mod

Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]

#: Body number of the sprung mass; the corner bodies follow from 2 (see
#: ``suspension.corner.body_numbers``).
CHASSIS = 1
UPRIGHT = 2


def body_numbers(model):
    """Engine body numbers of the quarter car of ``model``.

    :return: ``{"chassis": 1, "upright": 2, ...}`` with the rocker and the
        rack when the model has them
    :rtype: dict
    """
    nums = {"chassis": CHASSIS}
    nums.update(corner_mod.body_numbers(model, UPRIGHT))
    return nums


def static_tire_deflection(model, sprung_mass, gravity=9.81):
    """The tire deflection (m) that carries the quarter car weight.

    :return: deflection, total weight (N)
    :rtype: tuple
    """
    weight = (sprung_mass + model.upright_mass + model.rocker_mass
              + model.rack_mass
              + sum(a["mass"] for a in model.arms.values())) * gravity
    return weight / model.tire_k, weight


def travel_stop_lengths(mbs, model):
    """Spring-line lengths (m) where the bump stop and the droop stop touch.

    The model gives the stops as wheel travels; a K&C sweep maps them to
    the length between the spring points.

    :return: ``(bump length, droop length)``, or None when the model has no
        stops
    :rtype: tuple or None
    """
    from . import kc

    if model.stops is None:
        return None
    travels = [model.stops["bump_travel"], model.stops["droop_travel"]]
    res = kc.bump_sweep(mbs, model, travels)
    if not res["completed"]:
        raise RuntimeError("travel stops: the K&C sweep to the stop "
                           f"travels failed: {res['message']}")
    lengths = np.asarray(res["spring_length"], dtype="float64")
    if not lengths[0] < lengths[1]:
        raise ValueError("travel stops: the spring line must shorten in "
                         "bump; check the stop travels")
    return float(lengths[0]), float(lengths[1])


def quarter_car(mbs, model, sprung_mass, road_func=None, gravity=9.81,
                dt=1e-3, motion_ratio=None, road=None):
    """Build the quarter car at its design pose.

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model (its spring free length is adjusted in a copy)
    :type model: suspension.model.CornerModel
    :param sprung_mass: quarter sprung mass (kg)
    :type sprung_mass: float
    :param road_func: road plate height drive function (m) of a rig; None
        and no ``road`` is a flat road
    :param gravity: gravity along -z (m/s^2)
    :type gravity: float
    :param dt: time step
    :type dt: float
    :param motion_ratio: motion ratio at design for the preload; None computes
        it with a K&C sweep
    :type motion_ratio: float or None
    :param road: road input (suspension.road.RoadInput); takes precedence
        over ``road_func``
    :return: (system, handles dict)
    :rtype: tuple
    """
    import copy

    from . import kc

    m = copy.deepcopy(model)
    if motion_ratio is None:
        motion_ratio = kc.bump_sweep(mbs, model, [0.0])["motion_ratio"][0]
    # First guess: the spring carries the sprung weight through the motion
    # ratio.  The corner bodies also load it, so the free length is then
    # corrected until the static equilibrium is at the static travel (the
    # ride height setting; 0 is the design pose).
    preload = sprung_mass * gravity / motion_ratio
    m.spring_free_offset = preload / m.spring_k  # compression at design
    if road is not None:
        height, slope = road.drive_functions(mbs)
    else:
        height = road_func or mbs.DriveFunction.parameter(0.0)
        slope = mbs.DriveFunction.parameter(0.0)
    ground_z = float(m.hp["wheel_center"][2] - m.wheel_radius)
    if abs(ground_z) > 1e-12:
        raise ValueError("the corner model must put the contact patch at "
                         f"z = 0 at design, not {ground_z:g} m")
    deflection, weight = static_tire_deflection(m, sprung_mass, gravity)
    stop_lengths = travel_stop_lengths(mbs, m)

    def build():
        s = mbs.DynamicsSystem(dt)
        g = mbs.create_ground(0)
        s.Add(g)
        chassis = mbs.create_mobilized_body_quat(
            CHASSIS, Z, Z, Z, [1, 0, 0, 0], Z, Z, sprung_mass,
            [1.0, 1.0, 1.0], [0, 0, -gravity * sprung_mass], Z)
        s.Add(chassis)
        s.Add(mbs.create_joint(2, Z, Z, Z, Z, EZ, EZ, g, chassis))
        c = corner_mod.add_corner(mbs, s, m, g, chassis, np.zeros(3),
                                  UPRIGHT, travel_func=None, gravity=gravity)
        # The road plane at design is z = 0 under a wheel center at the
        # loaded radius; the free radius adds the static deflection.
        tire = mbs.create_tire(g, c.upright, list(m.spin_axis),
                               m.wheel_radius + deflection, m.tire_k,
                               m.tire_c, road_height=height,
                               road_slope=slope)
        s.Add(tire)
        stop = None
        if stop_lengths is not None:
            st = m.stops
            stop = mbs.create_length_stop(
                c.body(c.spring_body),
                chassis, list(c.spring_points[0]),
                list(c.spring_points[1]), min_length=stop_lengths[0],
                max_length=stop_lengths[1], k=st["k"], k2=st["k2"],
                c=st["c"])
            s.Add(stop)
        s.init()
        return s, chassis, c, tire, stop

    def travel_error(s, c):
        if not s.find_equilibrium(tol=1e-11)["converged"]:
            raise RuntimeError("quarter car: no static equilibrium")
        pos = np.asarray(s.get_body_positions())
        # The static travel (+ bump) of the ride height setting.
        return float(pos[c.nums["upright"]][2] - pos[CHASSIS][2]
                     - m.hp["wheel_center"][2] - m.static_travel)

    # Secant iteration on the spring free length: the suspension travel at
    # equilibrium is linear in it near design.
    offsets = [m.spring_free_offset]
    s, chassis, c, tire, stop = build()
    errors = [travel_error(s, c)]
    for _ in range(8):
        if abs(errors[-1]) < 1e-10:
            break
        if len(offsets) == 1:
            step = offsets[0] + 1e-4
        else:
            rate = (errors[-1] - errors[-2]) / (offsets[-1] - offsets[-2])
            step = offsets[-1] - errors[-1] / rate
        offsets.append(step)
        m.spring_free_offset = step
        s, chassis, c, tire, stop = build()
        errors.append(travel_error(s, c))
    if abs(errors[-1]) > 1e-8:
        raise RuntimeError(f"quarter car: the static equilibrium is "
                           f"{errors[-1]:.3g} m off the static travel")
    # The returned system is at its static equilibrium (time 0).
    handles = {"chassis": chassis, "corner": c, "tire": tire, "stop": stop,
               "stop_lengths": stop_lengths, "model": m,
               "motion_ratio": motion_ratio, "road": road,
               "tire_deflection": deflection, "weight": weight}
    return s, handles


def effective_unsprung_mass(mbs, model):
    """Unsprung mass for wheel travel from kinetic energy (chassis fixed).

    m_u = sum over corner bodies of m |dr/dz|^2 + w^T J w with the K&C
    sensitivities per unit wheel travel.

    :return: effective mass (kg)
    :rtype: float
    """
    return linear_quarter_car(mbs, model, 0.0, 0.0, 0.0)["M"][1, 1]


def linear_quarter_car(mbs, model, sprung_mass, wheel_rate, tire_rate,
                       tire_load=0.0, h=1e-5, travel=0.0):
    """Exact linear quarter car in (chassis z, suspension travel) coordinates.

    Built from K&C sensitivities and the Lagrangian, independent of the
    engine dynamics. With dr_b/dz and w_b per unit travel for each corner body:
    M11 = m_s + sum m_b, M12 = sum m_b (dr_b/dz)_z, M22 = sum m_b |dr_b/dz|^2 +
    w^T J w; the tire acts on the height z_c + a z of the lowest point of the
    wheel (the engine tire contact) with a = dz_c/dz: K = [[k_t, k_t a],
    [k_t a, k_w + k_t a^2 - F_t a']] with the static tire load F_t
    (compression) and a' = d^2 z_c / dz^2 (geometric stiffness of the
    preload).

    The forward rate a_x = dx_c/dz of the contact point gives the travel
    the generalized force a_x F_x of a longitudinal tire force F_x.

    :param travel: the wheel travel (m, + bump) of the linearization, the
        static travel of a ride height setting
    :return: dict with M, K (2x2), a, a_prime, and a_x
    :rtype: dict
    """
    from .corner import kc_rig
    from .metrics import rot

    s, c = kc_rig(mbs, model)
    z_lin = c.z0 + float(travel)
    res = result_arrays(s.sweep(c.travel, [z_lin]))

    up_n = c.nums["upright"]
    spin = np.asarray(model.spin_axis, dtype="float64")
    spin = spin / np.linalg.norm(spin)
    # The engine tire touches the road at the lowest point of a wheel of the
    # free radius: z_c = z_w - R sqrt(1 - s_z^2), s the spin axis.  Its
    # height is stationary over the wheel circle, so its rate is the
    # vertical velocity of the material point there (and the tire force
    # derives from the potential k (z_r - z_c)^2 / 2).
    radius = model.wheel_radius + (tire_load / tire_rate if tire_rate
                                   else 0.0)

    def patch_rate(k_res, k):
        am = rot(k_res["quaternions"][k, up_n])
        sv = am @ spin
        sdot = np.cross(k_res["domega_ds"][k, up_n], sv)
        root = np.sqrt(1.0 - sv[2] ** 2)
        return float(k_res["dr_ds"][k, up_n][2]
                     + radius * sv[2] * sdot[2] / root)

    def patch_rate_x(k_res, k):
        # Forward rate of the contact material point (the lowest point of
        # the wheel), for the longitudinal tire force on a sloped road.
        am = rot(k_res["quaternions"][k, up_n])
        sv = am @ spin
        down = -np.array([0.0, 0.0, 1.0]) + sv[2] * sv
        arm = radius * down / np.linalg.norm(down)
        vel = k_res["dr_ds"][k, up_n] + np.cross(k_res["domega_ds"][k, up_n],
                                                 arm)
        return float(vel[0])

    side_res = result_arrays(s.sweep(c.travel, [z_lin - h, z_lin + h]))
    a_prime = (patch_rate(side_res, 1) - patch_rate(side_res, 0)) / (2 * h)
    s.sweep(c.travel, [z_lin])
    m11, m12, m22 = sprung_mass, 0.0, 0.0
    bodies = [("upright", model.upright_mass, model.upright_inertia),
              ("rocker", model.rocker_mass, model.rocker_inertia),
              ("rack", model.rack_mass, [1e-3, 1e-3, 1e-3])]
    bodies += [("arm_" + arm, spec["mass"], spec["inertia"])
               for arm, spec in model.arms.items()]
    for name, mass, inertia in bodies:
        if name not in c.nums:
            continue
        n = c.nums[name]
        v = res["dr_ds"][0, n]
        wb = rot(res["quaternions"][0, n]).T @ res["domega_ds"][0, n]
        m11 += mass
        m12 += mass * v[2]
        m22 += mass * v @ v + wb @ (np.array(inertia) * wb)
    a = patch_rate(res, 0)
    a_x = patch_rate_x(res, 0)
    mm = np.array([[m11, m12], [m12, m22]])
    kk = np.array([[tire_rate, tire_rate * a],
                   [tire_rate * a,
                    wheel_rate + tire_rate * a * a - tire_load * a_prime]])
    return {"M": mm, "K": kk, "a": a, "a_prime": a_prime, "a_x": a_x}


def frequencies(mm, kk):
    """Undamped natural frequencies (Hz) of M x'' + K x = 0, ascending.

    :return: frequencies
    :rtype: numpy.ndarray
    """
    lam = np.linalg.eigvals(np.linalg.solve(mm, kk))
    return np.sort(np.sqrt(lam.real)) / (2 * np.pi)


def two_dof_frequencies(ms, mu, kw, kt):
    """Undamped natural frequencies (Hz) of the classic 2-DOF quarter car.

    :return: (ride, wheel hop) in Hz
    :rtype: tuple
    """
    mmat = np.diag([ms, mu])
    kmat = np.array([[kw, -kw], [-kw, kw + kt]])
    lam = np.linalg.eigvals(np.linalg.solve(mmat, kmat))
    f = np.sort(np.sqrt(lam.real)) / (2 * np.pi)
    return float(f[0]), float(f[1])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
