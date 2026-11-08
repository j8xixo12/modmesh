# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Construct and analyze an explicitly provisional four-corner vehicle."""

import numpy as np

from .suspension_candidate import candidate


def vehicle_model(data_dir):
    from solvcon.multibody.suspension.car import VehicleModel

    front, rear = candidate('front', data_dir), candidate('rear', data_dir)
    # The front rack is modelled separately on each side by solvcon.
    # All component masses and inertias here are explicit assumptions.
    total = 1654.0
    front_axle_mass = total * 0.6
    front_sprung = front_axle_mass - 2 * (45.0 + 1.5)
    sprung = total - 4 * 45.0 - 2 * 1.5
    return VehicleModel(
        {
            'name': 'Golf R candidate; assumptions not validated',
            'front': front,
            'rear': rear,
            'wheelbase': 2610.0,
            'sprung_mass': sprung,
            'front_fraction': front_sprung / sprung,
            'cg_height': 550.0,
            'sprung_inertia': [500, 2000, 2200],
            'tire': {'cornering_stiffness': [1200, 1200], 'mu': 1.0},
        }
    )


def run(data_dir):
    from solvcon import multibody
    from solvcon.multibody.suspension import car, longitudinal

    vm = vehicle_model(data_dir)
    turning = car.steady_state(multibody, vm, [0.0, 0.1, 0.2])
    symmetric = car.steady_state(multibody, vm, [0.0, 0.1, -0.1])
    braking = longitudinal.steady_state(
        multibody, vm, [0.0, -0.1, -0.2], brake_bias=0.76
    )
    turning_loads = np.asarray(turning['loads'])
    braking_loads = np.asarray(braking['loads'])
    weight = 1654.0 * 9.81
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
        'assumptions': [
            'unverified hardpoint frames',
            '45 kg per upright',
            '1.5 kg per front rack',
            'CG height 0.55 m',
            'inertia 500,2000,2200 kg m^2',
            'vertical wheel-centre springs',
            'no anti-roll bars',
            'assumed tire cornering stiffness',
        ],
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
