# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Experimental AC hardpoint mapping for independent K&C comparison.

The mapping is a hypothesis: chassis points are measured from the axle
centre, tyre points from the wheel centre, with X inboard, Y up, Z forward.
It requires external validation before it can be called a Golf R model.
"""

import configparser
import math
from pathlib import Path

import numpy as np


def candidate(axle, data_dir):
    cfg = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=(';',)
    )
    path = Path(data_dir) / 'suspensions.ini'
    if not cfg.read(path):
        raise FileNotFoundError(path)
    section = cfg[axle.upper()]
    half_track = float(section['TRACK']) / 2
    radius = 0.320

    def point(key, tyre=False):
        x, y, z = (float(v) for v in section[key].split(','))
        lateral = half_track - x if tyre else x
        return [1000 * z, 1000 * lateral, 1000 * (radius + y)]

    def link(car, tyre, rack=False):
        return {
            ('rack' if rack else 'chassis'): point(car),
            'upright': point(tyre, True),
        }

    links = {
        'lower_front': link('WBCAR_BOTTOM_FRONT', 'WBTYRE_BOTTOM'),
        'lower_rear': link('WBCAR_BOTTOM_REAR', 'WBTYRE_BOTTOM'),
        'tie_rod': link('WBCAR_STEER', 'WBTYRE_STEER', axle == 'front'),
    }
    if axle == 'rear':
        links.update(
            {
                'upper_front': link('WBCAR_TOP_FRONT', 'WBTYRE_TOP'),
                'upper_rear': link('WBCAR_TOP_REAR', 'WBTYRE_TOP'),
            }
        )
    camber = math.radians(float(section['STATIC_CAMBER']))
    model = {
        'name': 'Golf R ' + axle + ' candidate',
        'source': 'Experimental hardpoint interpretation; not validated',
        'side': 'left',
        'type': 'macpherson' if axle == 'front' else 'double_wishbone',
        'wheel': {
            'radius': 320.0,
            'center': [0, half_track * 1000, 320],
            'spin_axis': [0, math.cos(camber), math.sin(camber)],
        },
        'upright': {'mass': 45.0, 'inertia': [1.0, 1.0, 1.0]},
        'links': links,
        'spring': {
            'upright': [0, half_track * 1000, 320],
            'chassis': [0, half_track * 1000, 620],
            'k': float(section['SPRING_RATE']) / 1000,
            'c': 1.0,
        },
        'tire': {'k': 298.926, 'c': 0.1},
    }
    if axle == 'front':
        model['strut'] = {
            'top_mount': point('STRUT_CAR'),
            'upright': point('STRUT_TYRE', True),
        }
        model['rack'] = {'mass': 1.5, 'axis': [0, 1, 0]}
    return model


def compare(axle, data_dir):
    from solvcon import multibody
    from solvcon.multibody.suspension import kc
    from solvcon.multibody.suspension.model import CornerModel
    from solvcon.multibody.suspension.reference import ReferenceCorner

    model = CornerModel(candidate(axle, data_dir))
    travels = np.linspace(-0.02, 0.02, 9, dtype='float64')
    result = kc.bump_sweep(multibody, model, travels)
    if not result['completed'] or len(result['travel']) != len(travels):
        raise RuntimeError(f'{axle} sweep failed: {result["message"]}')
    ref = ReferenceCorner(model)
    errors = {}
    for key in ('camber', 'toe', 'motion_ratio'):
        expected = np.array(
            [
                ref.metrics(model.hp['wheel_center'][2] + t)[key]
                for t in travels
            ],
            dtype='float64',
        )
        errors[key] = float(np.max(np.abs(result[key] - expected)))
    if max(errors.values()) > 1e-5:
        raise AssertionError(f'{axle} independent solver mismatch: {errors}')
    return {
        'candidate': candidate(axle, data_dir),
        'travel_m': travels.tolist(),
        'camber_rad': result['camber'].tolist(),
        'toe_rad': result['toe'].tolist(),
        'motion_ratio': result['motion_ratio'].tolist(),
        'contact_patch_m': result['contact_patch'].tolist(),
        'spring_length_m': result['spring_length'].tolist(),
        'roll_center_m': result['roll_center'].tolist(),
        'reference_max_abs_error': errors,
        'warning': 'Both solvers use the same unverified geometry.',
    }


def steering(data_dir):
    from solvcon import multibody
    from solvcon.multibody.suspension import kc
    from solvcon.multibody.suspension.model import CornerModel

    racks = np.linspace(-0.01, 0.01, 9)
    result = kc.steer_sweep(
        multibody, CornerModel(candidate('front', data_dir)), racks
    )
    if not result['completed'] or len(result['rack']) != len(racks):
        raise RuntimeError(f"Steering sweep failed: {result['message']}")
    if (
        not np.isfinite(result['toe']).all()
        or not np.isfinite(result['camber']).all()
    ):
        raise AssertionError('Steering sweep returned nonfinite results')
    return {
        'rack_m': result['rack'].tolist(),
        'toe_rad': result['toe'].tolist(),
        'camber_rad': result['camber'].tolist(),
        'warning': 'Rack travel and geometry are candidate assumptions.',
    }


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
