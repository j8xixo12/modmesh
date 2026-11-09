# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Construct and analyze an explicitly provisional four-corner vehicle."""

import math
from pathlib import Path

import numpy as np

from .suspension_candidate import cg_height, candidate, read_ini, tyre_section

GRAVITY = 9.81
#: Rack body mass (kg) of each front corner (assumed).
RACK_MASS = 1.5


def tire_data(data_dir, front_load, rear_load):
    """The vehicle tire block from ``tyres.ini`` (AC tyre model v10).

    The AC friction is mu(F_z) = DY_REF (F_z / FZ0)^(LS_EXPY - 1).  The
    brush model (Pacejka, Tire and Vehicle Dynamics, 2012, Sec. 3.2) peaks
    at tan(alpha_sl) = 3 mu F_z / C_alpha, so the slip angle of the peak,
    ``FRICTION_LIMIT_ANGLE``, gives C_alpha(F_z) = 3 mu(F_z) F_z /
    tan(alpha_sl), proportional to F_z^LS_EXPY.  The engine tire takes the
    values at the static loads with the exponent LS_EXPY, the brush curve,
    and ``RELAXATION_LENGTH``.

    :param front_load: static load (N) of one front tire
    :param rear_load: static load (N) of one rear tire
    :return: the ``tire`` block of a vehicle model (N/deg at the static
        loads), with the load sensitivities linearized there for the
        handling diagram
    :rtype: dict
    """
    out = {'cornering_stiffness': [], 'mu': [], 'relaxation_length': [],
           'reference_load': [front_load, rear_load], 'brush': True}
    exponents = []
    for axle, load in (('front', front_load), ('rear', rear_load)):
        t = tyre_section(axle, data_dir)
        fz0, n = float(t['FZ0']), float(t['LS_EXPY'])
        mu = float(t['DY_REF']) * (load / fz0) ** (n - 1.0)
        peak = math.radians(float(t['FRICTION_LIMIT_ANGLE']))
        c_alpha = 3.0 * mu * load / math.tan(peak)
        out['cornering_stiffness'].append(c_alpha * math.pi / 180.0)
        out['mu'].append(mu)
        out['relaxation_length'].append(
            float(t.get('RELAXATION_LENGTH', '0')))
        exponents.append(n)
    if max(exponents) - min(exponents) > 1e-12:
        raise ValueError('tyres.ini: LS_EXPY differs between the axles')
    out['load_exponent'] = exponents[0]
    out['relaxation_length'] = float(np.mean(out['relaxation_length']))
    # mu ~ F_z^(n-1), C_alpha ~ F_z^n: both linear sensitivities are 1 - n
    # at the static load.
    out['mu_load_sensitivity'] = 1.0 - exponents[0]
    out['stiffness_load_sensitivity'] = 1.0 - exponents[0]
    return out


def vehicle_data(data_dir):
    """The vehicle model data of the AC car (``car.ini``,
    ``suspensions.ini``, ``tyres.ini``).

    * total mass ``TOTALMASS`` and front share ``CG_LOCATION``;
    * the unsprung mass ``HUB_MASS`` per corner (the upright body), and a
      1.5 kg rack body per front corner;
    * the sprung CG height: AC puts the CG -BASEY above the wheel centre,
      radius - BASEY above the road per axle, weighted by ``CG_LOCATION``;
    * the sprung inertia of the AC box ``INERTIA`` = width, height, length
      (m): I = m (a^2 + b^2) / 12.

    :return: vehicle model data (mm, kg) with ``assumptions``
    :rtype: dict
    """
    data_dir = Path(data_dir)
    car = read_ini(data_dir / 'car.ini')['BASIC']
    susp = read_ini(data_dir / 'suspensions.ini')
    front, rear = candidate('front', data_dir), candidate('rear', data_dir)
    total = float(car['TOTALMASS'])
    share = float(susp['BASIC']['CG_LOCATION'])
    hub_f, hub_r = front['upright']['mass'], rear['upright']['mass']
    sprung = total - 2.0 * (hub_f + RACK_MASS) - 2.0 * hub_r
    front_sprung = total * share - 2.0 * (hub_f + RACK_MASS)
    height = cg_height(data_dir)
    width, tall, length = (float(v) for v in car['INERTIA'].split(','))
    inertia = [sprung * (width ** 2 + tall ** 2) / 12.0,
               sprung * (tall ** 2 + length ** 2) / 12.0,
               sprung * (width ** 2 + length ** 2) / 12.0]
    load_f = 0.5 * total * share * GRAVITY
    load_r = 0.5 * total * (1.0 - share) * GRAVITY
    return {
        'name': 'Golf R (AC data)',
        'source': 'Assetto Corsa car.ini, suspensions.ini, tyres.ini',
        'front': front,
        'rear': rear,
        'wheelbase': 1000.0 * float(susp['BASIC']['WHEELBASE']),
        'sprung_mass': sprung,
        'front_fraction': front_sprung / sprung,
        'cg_height': 1000.0 * height,
        'sprung_inertia': inertia,
        'tire': tire_data(data_dir, load_f, load_r),
        'assumptions': sorted(set(front['assumptions']
                                  + rear['assumptions'])) + [
            'sprung CG height from BASEY (radius - BASEY per axle)',
            'sprung inertia of the AC INERTIA box',
            'cornering stiffness from FRICTION_LIMIT_ANGLE by the brush '
            'model',
            f'{RACK_MASS} kg rack body per front corner',
        ],
    }


def vehicle_model(data_dir):
    """The :class:`VehicleModel` of :func:`vehicle_data`."""
    from solvcon.multibody.suspension.car import VehicleModel

    return VehicleModel(vehicle_data(data_dir))


def run(data_dir):
    from solvcon import multibody
    from solvcon.multibody.suspension import car, longitudinal
    from solvcon.multibody.suspension.car import VehicleModel

    data = vehicle_data(data_dir)
    vm = VehicleModel(data)
    turning = car.steady_state(multibody, vm, [0.0, 0.1, 0.2])
    symmetric = car.steady_state(multibody, vm, [0.0, 0.1, -0.1])
    braking = longitudinal.steady_state(
        multibody, vm, [0.0, -0.1, -0.2], brake_bias=0.76
    )
    turning_loads = np.asarray(turning['loads'])
    braking_loads = np.asarray(braking['loads'])
    weight = (vm.sprung_mass
              + 2 * (vm.front.upright_mass + vm.front.rack_mass)
              + 2 * vm.rear.upright_mass) * GRAVITY
    if not np.allclose(turning_loads.sum(axis=1), weight, rtol=5e-4):
        raise AssertionError('Cornering loads do not balance vehicle weight')
    if not np.allclose(braking_loads.sum(axis=1), weight, rtol=5e-4):
        raise AssertionError('Braking loads do not balance vehicle weight')
    if not braking_loads[-1, :2].sum() > braking_loads[0, :2].sum():
        raise AssertionError('Front load must increase while braking')
    symmetry = float(
        np.max(
            np.abs(symmetric['loads'][1] - symmetric['loads'][2][[1, 0, 3, 2]])
        )
    )
    if symmetry > 0.01:
        raise AssertionError(f'Left/right symmetry mismatch: {symmetry} N')
    return {
        'status': 'provisional',
        'assumptions': data['assumptions'],
        'sprung_mass_kg': vm.sprung_mass,
        'sprung_front_fraction': vm.front_fraction,
        'corner_order': list(car.CORNERS),
        'cornering': {
            'acceleration_g': turning['a_y'].tolist(),
            'loads_N': turning_loads.tolist(),
            'roll_rad': turning['roll'].tolist(),
        },
        'braking': {
            'acceleration_g': braking['a_x'].tolist(),
            'loads_N': braking_loads.tolist(),
            'pitch_rad': braking['pitch'].tolist(),
        },
        'validation': {
            'static_weight_N': weight,
            'max_cornering_weight_residual_N': float(
                np.max(np.abs(turning_loads.sum(axis=1) - weight))
            ),
            'max_braking_weight_residual_N': float(
                np.max(np.abs(braking_loads.sum(axis=1) - weight))
            ),
            'left_right_symmetry_residual_N': symmetry,
        },
    }


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
