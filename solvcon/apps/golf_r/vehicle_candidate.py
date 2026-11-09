# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Construct and analyze an explicitly provisional four-corner vehicle."""

from pathlib import Path

import numpy as np

from .suspension_candidate import cg_height, candidate, read_ini, tyre_section
from .tire import ac_tire

GRAVITY = 9.81
#: Rack body mass (kg) of each front corner (assumed).
RACK_MASS = 1.5


def tire_data(data_dir):
    """The vehicle tire block from ``tyres.ini`` (AC tyre model v10): the
    Magic Formula of :func:`.tire.ac_tire` per axle and the relaxation
    length.

    :return: the ``tire`` block of a vehicle model and the assumptions of
        the mapping
    :rtype: tuple
    """
    block, lengths, assumptions = {'magic_formula': {}}, [], []
    for axle in ('front', 'rear'):
        tire, assumptions = ac_tire(tyre_section(axle, data_dir))
        block['magic_formula'][axle] = tire.parameters()
        lengths.append(tire.relaxation_length)
    if max(lengths) - min(lengths) > 1e-12:
        raise ValueError('tyres.ini: RELAXATION_LENGTH differs between the '
                         'axles')
    block['relaxation_length'] = lengths[0]
    return block, assumptions


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
    tire, tire_notes = tire_data(data_dir)
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
        'tire': tire,
        'assumptions': sorted(set(front['assumptions']
                                  + rear['assumptions'])) + [
            'sprung CG height from BASEY (radius - BASEY per axle)',
            'sprung inertia of the AC INERTIA box',
            f'{RACK_MASS} kg rack body per front corner',
        ] + tire_notes,
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
