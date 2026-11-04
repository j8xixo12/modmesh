# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Rigid multibody dynamics in Euler parameters (Haug, Computer-Aided Kinematics
and Dynamics of Mechanical Systems, Vol. I).

The names and arguments follow the standalone ``multibody_solver`` extension
module.  Numeric results (body states, sweep arrays, joint reactions,
constraint evaluations, modal results) are ``SimpleArrayFloat64`` (integer
sweep arrays ``SimpleArrayInt32``); ``numpy.asarray`` reads them without a
copy.  Lists and dicts of scalars stay Python objects.  Inputs take Python
sequences and numpy arrays.

The subpackages are imported on demand: ``tools`` (helpers around the
engine) and ``models`` (Haug book examples, validation benchmarks, example
mechanisms).
"""

try:
    from _solvcon import multibody as _impl  # noqa: F401
except ImportError:
    from .._solvcon import multibody as _impl  # noqa: F401

__all__ = [
    'HAS_MULTIBODY',
    'FEATURE_ROAD_TIRE',
    'FEATURE_LENGTH_STOP',
    'Body',
    'Ground',
    'MobilizedBody',
    'Joint',
    'ForceElement',
    'RotationalSDA',
    'GearCoupler',
    'PythonForce',
    'TireForce',
    'Bushing',
    'CamProfile',
    'DriveFunction',
    'DynamicsSystem',
    'ModalResult',
    'vec_to_list',
    'list_to_vec',
    'create_ground',
    'create_mobilized_body',
    'create_mobilized_body_quat',
    'create_joint',
    'create_joint_driving',
    'create_joint_distance',
    'create_absolute_driver',
    'create_distance_driver',
    'create_translational_driver',
    'create_rotational_driver',
    'create_screw_joint',
    'create_revolute_spherical_joint',
    'create_revolute_revolute_joint',
    'create_revolute_translational_joint',
    'create_strut_joint',
    'create_gear_coupler',
    'create_point_follower',
    'create_flat_follower',
    'create_spring_damper',
    'create_bushing',
    'create_tire',
    'create_length_stop',
    'create_rsda',
    'create_python_force',
    'get_force_element_loads',
    'get_tire_state',
    'get_length_stop_state',
    'modal_analysis',
]

HAS_MULTIBODY = _impl.HAS_MULTIBODY

if HAS_MULTIBODY:
    for _name in __all__[1:]:
        globals()[_name] = getattr(_impl, _name)
    del _name
else:
    __all__ = ['HAS_MULTIBODY']

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
