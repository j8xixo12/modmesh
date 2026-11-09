# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Provisional planar bicycle dynamics with traction and braking limits.

The drive split, turbo time constant, efficiency, yaw inertia and drag are
assumptions. This is not an Assetto Corsa tire or AWD model.  The axle
forces come from the combined-slip tire of :mod:`.tire`: each axle has two
identical tires, and the longitudinal force is the driver demand that the
tires can deliver at the axle slip angle.

Longitudinal load transfer is quasi-static, ``dW = m a_x h / L``, with the
acceleration ``a_x`` of the forces that the tires deliver after the
friction limit, not of the requested force.  :func:`forces` solves the
algebraic loop between the axle loads and the delivered forces by a
fixed-point iteration; the contraction factor is at most ``mu h / L``.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from .tire import GRAVITY, Tire, force_for_demand

#: The default tire of the golf_r models.
TIRE = Tire()
#: Iteration limit and tolerance (m/s^2) of the load transfer loop.
LOAD_ITERATIONS = 50
LOAD_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Vehicle:
    """Vehicle parameters in SI units; the defaults are the golf_r source.

    Values from the AC car data (``data/`` of the Golf R mod):
    ``mass`` (car.ini TOTALMASS), ``radius`` and ``wheel_inertia`` (per
    wheel; tyres.ini RADIUS, ANGULAR_INERTIA), ``gears`` and
    ``final_drive`` (drivetrain.ini [GEARS] GEAR_1..GEAR_7, FINAL; the
    file has one final drive), ``gearbox_inertia`` (drivetrain.ini
    [GEARBOX] INERTIA), ``engine_inertia``, ``idle_rpm``, ``limiter_rpm``
    (engine.ini INERTIA, MINIMUM, LIMITER), ``max_boost``,
    ``boost_reference_rpm``, ``boost_gamma`` (engine.ini [TURBO_0]),
    ``max_brake_torque`` and ``front_brake_share`` (brakes.ini).  The
    other values are assumptions.
    """

    mass: float = 1654.0
    wheelbase: float = 2.61
    front_fraction: float = 0.60
    cg_height: float = 0.55
    yaw_inertia: float = 2200.0
    radius: float = 0.320
    wheel_inertia: float = 1.45
    final_drive: float = 4.47
    gears: tuple = (3.19, 2.75, 1.90, 1.04, 0.86, 0.79, 0.66)
    efficiency: float = 0.85
    front_drive_share: float = 0.6
    front_brake_share: float = 0.76
    max_brake_torque: float = 2400.0
    engine_inertia: float = 0.150
    gearbox_inertia: float = 0.017
    idle_rpm: float = 850.0
    limiter_rpm: float = 8000.0
    cda: float = 0.34 * 2.04
    rolling_coefficient: float = 0.012
    air_density: float = 1.225
    turbo_tau: float = 0.4
    max_boost: float = 1.9
    boost_reference_rpm: float = 2200.0
    boost_gamma: float = 2.5


def read_ini(path):
    """Read an AC ini file into upper-case sections and keys.

    :param path: The file path
    :type path: str or pathlib.Path
    :return: Mapping of section name to a mapping of key to string value
    :rtype: dict
    """
    sections, current = {}, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(';', 1)[0].strip()
        if line.startswith('[') and line.endswith(']'):
            current = sections.setdefault(line[1:-1].strip().upper(), {})
        elif '=' in line and current is not None:
            key, value = line.split('=', 1)
            current[key.strip().upper()] = value.strip()
    return sections


def drivetrain(data_dir):
    """Read the forward gear ratios and final drive of AC drivetrain.ini.

    AC lists [GEARS] COUNT, GEAR_1 to GEAR_<COUNT> and FINAL; GEAR_R is
    the reverse gear and is not used.

    :param data_dir: The AC car data directory
    :type data_dir: str or pathlib.Path
    :return: Keyword values ``gears`` and ``final_drive`` for
        :class:`Vehicle`
    :rtype: dict
    :raises ValueError: When a ratio is missing or not positive
    """
    section = read_ini(Path(data_dir) / 'drivetrain.ini').get('GEARS', {})
    try:
        count = int(section['COUNT'])
        gears = tuple(
            float(section['GEAR_%d' % i]) for i in range(1, count + 1)
        )
        final = float(section['FINAL'])
    except (KeyError, ValueError) as error:
        raise ValueError('drivetrain.ini needs COUNT, GEAR_n and FINAL') \
            from error
    if count < 1 or not all(np.isfinite(x) and x > 0 for x in gears):
        raise ValueError('forward gear ratios must be positive')
    if not np.isfinite(final) or final <= 0:
        raise ValueError('final drive must be positive')
    return dict(gears=gears, final_drive=final)


def shares(data_dir):
    """Read the drive and brake split of the AC car data.

    ``drivetrain.ini`` [AWD] FRONT_SHARE is the front share of the drive
    torque in percent (98 for the Golf R: a front-biased coupling);
    ``brakes.ini`` [DATA] FRONT_SHARE and MAX_TORQUE are the front brake
    share and the brake torque.  The AC centre coupling controllers
    (``ctrl_awd_center_lock.ini``, which move torque to the rear on rear
    wheel speed ratio) are not modelled: the split is fixed.

    :return: Keyword values for :class:`Vehicle` of the files present
    :rtype: dict
    """
    out = {}
    path = Path(data_dir)
    if (path / 'drivetrain.ini').exists():
        awd = read_ini(path / 'drivetrain.ini').get('AWD', {})
        if 'FRONT_SHARE' in awd:
            out['front_drive_share'] = float(awd['FRONT_SHARE']) / 100.0
    if (path / 'brakes.ini').exists():
        data = read_ini(path / 'brakes.ini').get('DATA', {})
        if 'FRONT_SHARE' in data:
            out['front_brake_share'] = float(data['FRONT_SHARE'])
        if 'MAX_TORQUE' in data:
            out['max_brake_torque'] = float(data['MAX_TORQUE'])
    for key in ('front_drive_share', 'front_brake_share'):
        if key in out and not 0.0 <= out[key] <= 1.0:
            raise ValueError(f'{key} must be between 0 and 1')
    return out


def vehicle_from_data(data_dir, vehicle=Vehicle()):
    """Return ``vehicle`` with the gear ratios, the drive split, and the
    brakes of the AC car data."""
    return replace(vehicle, **drivetrain(data_dir), **shares(data_dir))


def torque_curve(data_dir):
    points = []
    for line in (Path(data_dir) / 'power.lut').read_text().splitlines():
        if '|' in line:
            points.append([float(x) for x in line.split('|')])
    return np.asarray(points, dtype='float64')


def gear_ratio(p, gear):
    """Return the overall ratio of engine speed to driven wheel speed."""
    if not 1 <= gear <= len(p.gears):
        raise ValueError('Invalid gear')
    return p.gears[gear - 1] * p.final_drive


def engine_torque(p, rpm, throttle, boost, curve):
    """Return the engine torque in N m; the limiter cuts the fuel."""
    if rpm >= p.limiter_rpm:
        return 0.0
    rpm = max(rpm, p.idle_rpm)
    return throttle * np.interp(rpm, curve[:, 0], curve[:, 1]) * (1 + boost)


def boost_target(p, throttle, rpm):
    """Return the AC style turbo boost target for the pedal and rpm."""
    ratio = max(rpm, p.idle_rpm) / p.boost_reference_rpm
    return p.max_boost * throttle * min(ratio**p.boost_gamma, 1.0)


def axle_loads(p, ax):
    """Return the front and rear axle loads in N for an acceleration.

    :param p: The vehicle
    :type p: Vehicle
    :param ax: The longitudinal acceleration in m/s^2, negative braking
    :type ax: float
    :return: The front and rear axle loads
    :rtype: tuple
    """
    weight = p.mass * GRAVITY
    transfer = p.mass * ax * p.cg_height / p.wheelbase
    front = float(np.clip(weight * p.front_fraction - transfer, 0, weight))
    return front, weight - front


def forces(p, state, throttle, brake, steer, gear, curve, tire=TIRE):
    """Return axle forces and loads for the planar state."""
    if not (
        0 <= throttle <= 1
        and 0 <= brake <= 1
        and np.isfinite(steer)
        and 1 <= gear <= len(p.gears)
    ):
        raise ValueError('Invalid driver input or gear')
    _, _, _, u, v, yaw_rate, boost = state
    speed = max(float(u), 0.0)
    ratio = gear_ratio(p, gear)
    rpm = max(p.idle_rpm, speed / p.radius * ratio * 60 / (2 * np.pi))
    torque = engine_torque(p, rpm, throttle, boost, curve)
    drive = torque * ratio * p.efficiency / p.radius
    braking = brake * 4 * p.max_brake_torque / p.radius
    drag = 0.5 * p.air_density * p.cda * speed**2
    rolling = p.rolling_coefficient * p.mass * GRAVITY if speed > 0 else 0.0
    a = p.wheelbase * (1 - p.front_fraction)
    b = p.wheelbase * p.front_fraction
    # Tire slip angles; a positive steer slip gives a positive force.
    front_alpha = np.arctan2(v + a * yaw_rate, max(speed, 1.0)) - steer
    rear_alpha = np.arctan2(v - b * yaw_rate, max(speed, 1.0))
    front_demand = (
        drive * p.front_drive_share - braking * p.front_brake_share
    )
    rear_demand = drive * (1 - p.front_drive_share) - braking * (
        1 - p.front_brake_share
    )
    c, s = np.cos(steer), np.sin(steer)
    ax = 0.0
    for _ in range(LOAD_ITERATIONS):
        front_load, rear_load = axle_loads(p, ax)
        front_x, front_y, _ = force_for_demand(
            tire, front_load / 2, front_demand / 2, front_alpha
        )
        rear_x, rear_y, _ = force_for_demand(
            tire, rear_load / 2, rear_demand / 2, rear_alpha
        )
        front_x, front_y, rear_x, rear_y = (
            2 * front_x, 2 * front_y, 2 * rear_x, 2 * rear_y
        )
        delivered = (
            front_x * c - front_y * s + rear_x - drag - rolling
        ) / p.mass
        converged = abs(delivered - ax) <= LOAD_TOLERANCE
        ax = delivered
        if converged:
            break
    return dict(
        front_x=front_x,
        rear_x=rear_x,
        front_y=front_y,
        rear_y=rear_y,
        front_load=front_load,
        rear_load=rear_load,
        drag=drag + rolling,
        rpm=rpm,
        ax=ax,
    )


def derivative(p, state, throttle, brake, steer, gear, curve, tire=TIRE):
    x, y, heading, u, v, yaw_rate, boost = state
    f = forces(p, state, throttle, brake, steer, gear, curve, tire)
    front_x = f['front_x'] * np.cos(steer) - f['front_y'] * np.sin(steer)
    front_y = f['front_y'] * np.cos(steer) + f['front_x'] * np.sin(steer)
    du = (front_x + f['rear_x'] - f['drag']) / p.mass + yaw_rate * v
    if u <= 0 and du < 0:
        du = 0.0
    dv = (front_y + f['rear_y']) / p.mass - yaw_rate * u
    a, b = p.wheelbase * (1 - p.front_fraction), p.wheelbase * p.front_fraction
    dr = (a * front_y - b * f['rear_y']) / p.yaw_inertia
    target = boost_target(p, throttle, f['rpm'])
    forward_speed = max(u, 0.0)
    return np.array(
        [
            forward_speed * np.cos(heading) - v * np.sin(heading),
            forward_speed * np.sin(heading) + v * np.cos(heading),
            yaw_rate,
            du,
            dv,
            dr,
            (target - boost) / p.turbo_tau,
        ],
        dtype='float64',
    )


def simulate(p, initial, controls, duration, dt, curve=None, tire=TIRE):
    if duration <= 0 or dt <= 0 or not np.isfinite([duration, dt]).all():
        raise ValueError('duration and dt must be finite and positive')
    n = int(round(duration / dt))
    if n < 1 or not np.isclose(n * dt, duration, atol=1e-12, rtol=1e-10):
        raise ValueError('duration must be an integer multiple of dt')
    if curve is None:
        raise ValueError('Supply a torque curve loaded from car data')
    t = np.arange(n + 1, dtype='float64') * dt
    states = np.zeros((n + 1, 7), dtype='float64')
    states[0] = initial
    for i in range(n):
        now, state = t[i], states[i]

        def rhs(time, value):
            return derivative(p, value, *controls(time), curve, tire)

        k1 = rhs(now, state)
        k2 = rhs(now + dt / 2, state + dt * k1 / 2)
        k3 = rhs(now + dt / 2, state + dt * k2 / 2)
        k4 = rhs(now + dt, state + dt * k3)
        states[i + 1] = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states[i + 1, 3] = max(states[i + 1, 3], 0)
        states[i + 1, 6] = np.clip(states[i + 1, 6], 0, p.max_boost)
    return t, states


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
