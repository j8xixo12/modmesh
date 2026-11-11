# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Corner models from Assetto Corsa suspension data (``suspensions.ini``).

The AC hardpoints are measured from the wheel centre: X inboard, Y up, Z
forward, in metres.  Both the car side (``WBCAR_*``, ``STRUT_CAR``) and the
tyre side (``WBTYRE_*``, ``STRUT_TYRE``) use this origin (AC modding forum,
"Suspension geometry": "wheel centreline. To check just set a value to
0,0,0 and see where it is").  The geometry confirms it: the front strut
gives a kingpin inclination of 14.8 deg and the rear toe link is 508 mm long;
the axle-centre reading gives 33.5 deg and 114 mm, which no road car has.

AC gives the spring, damper, bump stop, and anti-roll bar as wheel rates.
The corner model holds them on its spring line (the strut axis of the front
MacPherson, a vertical line at the rear wheel centre), so this module
divides them by the motion ratio of a K&C sweep.  The anti-roll bar is an
equivalent bar (assumed geometry) whose torsion rate gives the AC axle roll
stiffness.

Assumptions (also in the ``assumptions`` list of each model):

* the AC design pose is the static pose (the spring holds it);
* ``[ARB] FRONT/REAR`` is the force per unit differential wheel travel (N/m),
  an axle roll stiffness of k t^2 (N m/rad);
* the droop stop rate is ``BUMP_STOP_RATE``;
* ``PROGRESSIVE_SPRING_RATE`` (N/m^2) is left out: it adds 1.4 % (front)
  and 0.6 % (rear) to the wheel rate at 50 mm of bump;
* ``ROD_LENGTH`` only sets the AC ride height, which is the design pose here.

Custom Shaders Patch cars may give an axle as ``TYPE=COSMIC``: free links
(``DJ<n>``, two ball joints between ``BODY_A`` and ``BODY_B``), steering
joints (``STEER_JOINT_<n>``), and a coilover section
(``<AXLE>_COILOVER_0``) between two bodies.  :func:`cosmic_points` takes a
COSMIC axle of four chassis-to-hub links that share two hub points (two
wishbones) and one steering joint as a double wishbone.  Unlike the DWB
wheel rates, the coilover acts on its own axis: ``RATE`` (in series with
``INSTALL_RATE``) and the ``DAMP_*`` slopes are coilover rates, and its
length limits are the stops (assumptions in the list of the model):

* the COSMIC points are from the wheel centre, as the DWB ones;
  ``RIM_OFFSET`` is left out;
* the bump stop touches at ``MIN_LENGTH + BUMPSTOP_GAP``, the droop stop at
  ``MAX_LENGTH``, both at ``END_RATE``;
* ``PRELOAD`` and ``PRELOAD_FORCE`` are left out (the design pose is the
  static pose), as are ``[_EXTENSION]`` and ``[_EXTENSION_FLEX]``.
"""

import configparser
import math
from pathlib import Path

import numpy as np

#: Equivalent anti-roll bar geometry relative to the wheel centre (mm, x
#: forward, y outboard, z up): the drop link point on the upright, the arm
#: tip above it, and the bar pivot ahead of the tip (a 200 mm arm).
ARB_LINK = np.array([-150.0, -150.0, 0.0])
ARB_TIP = np.array([-150.0, -150.0, 150.0])
ARB_PIVOT = np.array([-350.0, -150.0, 150.0])
#: A small travel (m) for the roll stiffness.
PROBE = 1e-3


def read_ini(path):
    """Read one AC ini file (``;`` comments, no interpolation).

    :param path: path of the ini file
    :type path: pathlib.Path or str
    :return: the parsed file
    :rtype: configparser.ConfigParser
    :raises FileNotFoundError: when the file does not exist
    """
    cfg = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=(';',), strict=False
    )
    if not cfg.read(path):
        raise FileNotFoundError(path)
    return cfg


def tyre_section(axle, data_dir):
    """The ``tyres.ini`` section of the axle (the default compound).

    :raises FileNotFoundError: when ``tyres.ini`` is missing
    """
    return read_ini(Path(data_dir) / 'tyres.ini')[axle.upper()]


def cg_height(data_dir):
    """The sprung CG height (m) of the AC car.

    AC puts the CG -BASEY above the wheel centre, so radius - BASEY above
    the road per axle; the axles weigh by the front share ``CG_LOCATION``.
    """
    cfg = read_ini(Path(data_dir) / 'suspensions.ini')
    share = float(cfg['BASIC']['CG_LOCATION'])
    front, rear = (float(tyre_section(a, data_dir)['RADIUS'])
                   - float(cfg[a.upper()]['BASEY'])
                   for a in ('front', 'rear'))
    return share * front + (1.0 - share) * rear


def _damper(section, motion_ratio, note=None):
    """The AC two-stage damper as a force-velocity table on the spring line.

    AC gives wheel-rate slopes (N.s/m) below and above a threshold wheel
    speed (m/s), in bump and in rebound.  On the spring line the speed is
    mr v and the force F / mr.  A COSMIC coilover gives them on its own
    axis: ``motion_ratio`` 1.
    """
    def side(slow, fast, threshold):
        c_slow = float(section[slow])
        c_fast = float(section[fast])
        knee = float(section[threshold])
        if min(c_slow, c_fast) < 0.0 or knee <= 0.0:
            raise ValueError(f'{slow}: negative slope or threshold')
        # Wheel speeds (m/s) and forces (N): the knee and one point past it.
        return [knee, 3.0 * knee], [c_slow * knee,
                                    c_slow * knee + c_fast * 2.0 * knee]

    bump_v, bump_f = side('DAMP_BUMP', 'DAMP_FAST_BUMP',
                          'DAMP_FAST_BUMPTHRESHOLD')
    reb_v, reb_f = side('DAMP_REBOUND', 'DAMP_FAST_REBOUND',
                        'DAMP_FAST_REBOUNDTHRESHOLD')
    wheel_v = [-v for v in reversed(bump_v)] + [0.0] + reb_v
    wheel_f = [-f for f in reversed(bump_f)] + [0.0] + reb_f
    return {
        'kind': 'table',
        'velocity': [1000.0 * motion_ratio * v for v in wheel_v],
        'force': [f / motion_ratio for f in wheel_f],
        'note': note or ('AC two-stage damper (DAMP_*), wheel rates '
                         'divided by the motion ratio '
                         f'{motion_ratio:.4f}; mm/s on the spring line '
                         '(+ rebound), N'),
    }


def _kingpin_offset(tyre_point, axis_a, axis_b):
    """Distance (m) of the steering arm point from the steering axis."""
    d = axis_b - axis_a
    d = d / np.linalg.norm(d)
    r = tyre_point - axis_a
    return float(np.linalg.norm(r - (r @ d) * d))


def _vector(text):
    return np.array([float(v) for v in text.split(',')])


def cosmic_points(cfg, axle):
    """The hardpoints of a COSMIC axle in the names of a DWB one.

    :param cfg: the parsed ``suspensions.ini``
    :param axle: ``'front'`` or ``'rear'``
    :return: ``{name: (x, y, z)}`` (m, wheel-centre frame) with the
        ``WBCAR_*`` and ``WBTYRE_*`` points and the coilover ends
        ``COILOVER_CAR`` and ``COILOVER_TYRE``
    :raises ValueError: when the links are not two chassis-to-hub
        wishbones, or the steering joint or the coilover is not one
        chassis-to-hub element
    """
    section = cfg[axle.upper()]
    where = f'{axle}: COSMIC'

    def ends(prefix, pos0='POS0', pos1='POS1'):
        bodies = (section.get(f'{prefix}_BODY_A', '').strip().upper(),
                  section.get(f'{prefix}_BODY_B', '').strip().upper())
        if ('CHASSIS', 'HUB_L') != bodies:
            raise ValueError(f'{where} {prefix} joins {bodies}; only '
                             'CHASSIS to HUB_L is supported')
        return (_vector(section[f'{prefix}_{pos0}']),
                _vector(section[f'{prefix}_{pos1}']))

    links = []
    while f'DJ{len(links)}' in section:
        links.append(ends(f'DJ{len(links)}'))
    hubs = {}
    for car, hub in links:
        hubs.setdefault(tuple(np.round(hub, 9)), []).append((car, hub))
    if 4 != len(links) or 2 != len(hubs) or any(
            2 != len(arm) for arm in hubs.values()):
        raise ValueError(f'{where} needs four DJ links that share two hub '
                         f'points (two wishbones); it has {len(links)} '
                         f'links on {len(hubs)} hub points')
    if 'STEER_JOINT_0' not in section or 'STEER_JOINT_1' in section:
        raise ValueError(f'{where} needs exactly one STEER_JOINT')
    lower, upper = sorted(hubs.values(), key=lambda arm: arm[0][1][1])
    points = {}
    for name, arm in (('BOTTOM', lower), ('TOP', upper)):
        # The chassis point further forward (+z) is the front one.
        rear, front = sorted(arm, key=lambda link: link[0][2])
        points[f'WBCAR_{name}_FRONT'] = front[0]
        points[f'WBCAR_{name}_REAR'] = rear[0]
        points[f'WBTYRE_{name}'] = front[1]
    points['WBCAR_STEER'], points['WBTYRE_STEER'] = ends('STEER_JOINT_0')
    coil = f'{axle.upper()}_COILOVER_0'
    if coil not in cfg or f'{axle.upper()}_COILOVER_1' in cfg:
        raise ValueError(f'{where} needs exactly one coilover ({coil})')
    section, where = cfg[coil], f'{axle}: {coil}'
    bodies = (section.get('BODY_A', '').strip().upper(),
              section.get('BODY_B', '').strip().upper())
    if ('CHASSIS', 'HUB_L') != bodies:
        raise ValueError(f'{where} joins {bodies}; only CHASSIS to HUB_L '
                         'is supported')
    points['COILOVER_CAR'] = _vector(section['POS_A'])
    points['COILOVER_TYRE'] = _vector(section['POS_B'])
    return points


def geometry(axle, data_dir):
    """The corner model of the AC hardpoints with unit rates.

    The spring, damper, stop, and bar rates are placeholders that
    :func:`candidate` replaces with the AC wheel rates.

    :param axle: ``'front'`` (MacPherson ``TYPE=STRUT``, or a COSMIC
        double wishbone on the rack) or ``'rear'`` (double wishbone with a
        toe link, ``TYPE=DWB`` or COSMIC)
    :type axle: str
    :param data_dir: directory of ``suspensions.ini`` and ``tyres.ini``
    :type data_dir: str or pathlib.Path
    :return: corner model data (mm, N/mm), left side
    :rtype: dict
    """
    cfg = read_ini(Path(data_dir) / 'suspensions.ini')
    section = cfg[axle.upper()]
    kind = section.get('TYPE', 'STRUT' if 'front' == axle else 'DWB')
    kind = kind.strip().upper()
    supported = {'front': ('STRUT', 'COSMIC'), 'rear': ('DWB', 'COSMIC')}
    if kind not in supported[axle]:
        raise ValueError(f'{axle}: suspension TYPE={kind} is not supported '
                         '(front STRUT or COSMIC, rear DWB or COSMIC)')
    cosmic = cosmic_points(cfg, axle) if 'COSMIC' == kind else {}
    half_track = float(section['TRACK']) / 2
    tyre = tyre_section(axle, data_dir)
    radius = float(tyre['RADIUS'])

    def raw(key):
        return cosmic[key] if cosmic else _vector(section[key])

    def point(key):
        # Wheel-centre frame (x inboard, y up, z forward) to the corner
        # frame (x forward, y left, z up) of the left corner.
        x, y, z = raw(key)
        return [1000.0 * z, 1000.0 * (half_track - x),
                1000.0 * (radius + y)]

    def link(car, tyre_point, rack=False):
        return {('rack' if rack else 'chassis'): point(car),
                'upright': point(tyre_point)}

    links = {
        'lower_front': link('WBCAR_BOTTOM_FRONT', 'WBTYRE_BOTTOM'),
        'lower_rear': link('WBCAR_BOTTOM_REAR', 'WBTYRE_BOTTOM'),
        'tie_rod': link('WBCAR_STEER', 'WBTYRE_STEER', 'front' == axle),
    }
    if 'STRUT' != kind:
        links['upper_front'] = link('WBCAR_TOP_FRONT', 'WBTYRE_TOP')
        links['upper_rear'] = link('WBCAR_TOP_REAR', 'WBTYRE_TOP')
        axis = raw('WBTYRE_BOTTOM'), raw('WBTYRE_TOP')
    else:
        axis = raw('STRUT_TYRE'), raw('STRUT_CAR')
    # Static camber (negative: the top leans inboard) and static toe (AC
    # TOE_OUT is a length change of the steering arm, + toe-out).
    camber = math.radians(float(section['STATIC_CAMBER']))
    arm = _kingpin_offset(raw('WBTYRE_STEER'), *axis)
    toe_out = math.atan2(float(section.get('TOE_OUT', '0')), arm)
    spin = [-math.sin(toe_out) * math.cos(camber),
            math.cos(toe_out) * math.cos(camber), -math.sin(camber)]
    center = [0.0, 1000.0 * half_track, 1000.0 * radius]
    model = {
        'name': f'{car_name(data_dir)} {axle} (AC data)',
        'source': f'Assetto Corsa suspensions.ini (TYPE={kind}) and '
                  'tyres.ini; hardpoints from the wheel centre',
        'side': 'left',
        'type': 'macpherson' if 'STRUT' == kind else 'double_wishbone',
        'wheel': {'radius': 1000.0 * radius, 'center': center,
                  'spin_axis': spin},
        'upright': {'mass': float(section['HUB_MASS']),
                    'inertia': [1.0, 1.0, 1.0]},
        'links': links,
        'tire': {'k': float(tyre['RATE']) / 1000.0,
                 'c': float(tyre.get('DAMP', '0')) / 1000.0},
    }
    if 'front' == axle:
        model['rack'] = {'mass': 1.5, 'axis': [0, 1, 0]}
    if cosmic:
        # The coilover on its own axis.
        model['spring'] = {'upright': point('COILOVER_TYRE'),
                           'chassis': point('COILOVER_CAR'), 'k': 1.0,
                           'c': 0.0}
    elif 'front' == axle:
        model['strut'] = {'top_mount': point('STRUT_CAR'),
                          'upright': point('STRUT_TYRE')}
        # The coil spring and the damper on the strut axis.
        model['spring'] = {'upright': point('STRUT_TYRE'),
                           'chassis': point('STRUT_CAR'), 'k': 1.0,
                           'c': 0.0}
    else:
        # AC gives no rear spring position: a vertical line at the wheel
        # centre (motion ratio 1), as AC applies its wheel rate.
        model['spring'] = {'upright': list(center),
                           'chassis': [0.0, center[1], center[2] + 300.0],
                           'k': 1.0, 'c': 0.0}
    c = np.array(center)
    model['arb'] = {'bar_pivot': (c + ARB_PIVOT).tolist(),
                    'arm_tip': (c + ARB_TIP).tolist(),
                    'link_point': (c + ARB_LINK).tolist(),
                    'axis': [0.0, 1.0, 0.0], 'k': 1.0, 'mass': 1.0,
                    'inertia': [0.005, 0.005, 0.005]}
    model['vehicle'] = {
        'half_track': 1000.0 * half_track,
        'wheelbase': 1000.0 * float(cfg['BASIC']['WHEELBASE']),
        'cg_height': 1000.0 * cg_height(data_dir),
        'front_brake_fraction': float(read_ini(
            Path(data_dir) / 'brakes.ini')['DATA']['FRONT_SHARE']),
        'corner_mass': 0.5 * float(read_ini(Path(data_dir) / 'car.ini')[
            'BASIC']['TOTALMASS']) * (
            float(cfg['BASIC']['CG_LOCATION']) if 'front' == axle
            else 1.0 - float(cfg['BASIC']['CG_LOCATION'])),
    }
    return model


def car_name(data_dir):
    """The ``SCREEN_NAME`` of ``car.ini`` (``[INFO]``), else the folder
    name."""
    path = Path(data_dir) / 'car.ini'
    if path.exists():
        name = read_ini(path).get('INFO', 'SCREEN_NAME', fallback='')
        if name.strip():
            return name.strip()
    return Path(data_dir).resolve().name


def _cosmic_rates(cfg, axle, model, mr):
    """Put the COSMIC coilover rates on the spring line of ``model``.

    :return: the wheel spring rate and the wheel stop rate (N/m), and the
        assumptions
    """
    section = cfg[f'{axle.upper()}_COILOVER_0']
    rate = float(section['RATE'])
    install = float(section.get('INSTALL_RATE', '0'))
    line = rate * install / (rate + install) if install > 0.0 else rate
    model['spring']['k'] = line / 1000.0
    model['spring']['damper'] = _damper(
        section, 1.0, 'AC COSMIC coilover damper (DAMP_*), coilover rates; '
        'mm/s on the spring line (+ rebound), N')
    spring = model['spring']
    length = float(np.linalg.norm(np.subtract(spring['upright'],
                                              spring['chassis']))) / 1000.0
    lo = float(section['MIN_LENGTH']) + float(section.get('BUMPSTOP_GAP',
                                                          '0'))
    hi = float(section['MAX_LENGTH'])
    if not lo < length < hi:
        raise ValueError(f'{axle}: the coilover length {length:.4f} m is not '
                         f'between MIN_LENGTH + BUMPSTOP_GAP {lo:.4f} m and '
                         f'MAX_LENGTH {hi:.4f} m')
    stop = float(section['END_RATE'])
    model['stops'] = {
        'bump_travel': 1000.0 * (length - lo) / mr,
        'droop_travel': -1000.0 * (hi - length) / mr,
        'k': stop / 1000.0,
        'note': 'AC COSMIC coilover: bump at MIN_LENGTH + BUMPSTOP_GAP, '
                'droop at MAX_LENGTH, END_RATE (coilover rate)'}
    assumptions = [
        'COSMIC: four DJ links as two wishbones, STEER_JOINT_0 as the tie '
        'rod, COILOVER_0 on its own axis',
        'COSMIC points from the wheel centre (as DWB); RIM_OFFSET left out',
        f'coilover RATE in series with INSTALL_RATE: {line:.0f} N/m',
        'coilover stops: bump at MIN_LENGTH + BUMPSTOP_GAP, droop at '
        'MAX_LENGTH, both END_RATE (END_VTAPER, END_VMAX left out)',
        'coilover PRELOAD and PRELOAD_FORCE left out (the design pose is '
        'the static pose)',
        '[_EXTENSION] and [_EXTENSION_FLEX] (chassis torsion) left out',
    ]
    return line * mr * mr, stop * mr * mr, assumptions


def motion_ratio(mbs, data):
    """Spring-line travel per unit wheel travel at the design pose."""
    from solvcon.multibody.suspension import kc
    from solvcon.multibody.suspension.model import CornerModel

    res = kc.bump_sweep(mbs, CornerModel(data), [0.0])
    if not res['completed']:
        raise RuntimeError(f'motion ratio sweep failed: {res["message"]}')
    return float(res['motion_ratio'][0])


def candidate(axle, data_dir):
    """The corner model with the AC wheel rates on its spring line.

    :param axle: ``'front'`` or ``'rear'``
    :type axle: str
    :param data_dir: directory of ``suspensions.ini`` and ``tyres.ini``
    :type data_dir: str or pathlib.Path
    :return: corner model data (mm, N/mm, N.s/mm), left side, with
        ``assumptions`` and the AC ``wheel_rates``
    :rtype: dict
    """
    from solvcon import multibody
    from solvcon.multibody.suspension import axle as axle_mod
    from solvcon.multibody.suspension.model import CornerModel

    cfg = read_ini(Path(data_dir) / 'suspensions.ini')
    section = cfg[axle.upper()]
    model = geometry(axle, data_dir)
    mr = motion_ratio(multibody, model)
    assumptions = [
        'AC design pose is the static pose',
        'ARB: force per unit differential wheel travel (N/m), roll '
        'stiffness k t^2; equivalent bar geometry',
        'upright inertia 1 kg m^2 about each axis',
    ]
    if 'COSMIC' == section.get('TYPE', '').strip().upper():
        k_wheel, k_stop_wheel, more = _cosmic_rates(cfg, axle, model, mr)
        assumptions += more
    else:
        k_wheel, k_stop_wheel = _ac_rates(section, model, mr, assumptions)
    # The equivalent bar: the roll stiffness it adds per unit torsion rate.
    track = 2.0 * float(model['vehicle']['half_track']) / 1000.0
    arb = float(cfg['ARB'][axle.upper()])
    target = arb * track * track
    unit = CornerModel(model)
    travels = [-PROBE, 0.0, PROBE]
    without = axle_mod.roll_sweep(multibody, unit, travels, False)
    with_bar = axle_mod.roll_sweep(multibody, unit, travels, True)
    for res in (without, with_bar):
        if not res['completed']:
            raise RuntimeError(f'roll sweep failed: {res["message"]}')
    per_unit = float(with_bar['roll_stiffness'][1]
                     - without['roll_stiffness'][1])
    if per_unit <= 0.0:
        raise RuntimeError('the equivalent anti-roll bar adds no roll '
                           'stiffness')
    model['arb']['k'] = target / per_unit
    model['arb']['note'] = (f'equivalent bar of AC [ARB] {axle.upper()}='
                            f'{arb:g} N/m: {target * math.pi / 180.0:.0f} '
                            'N m/deg of axle roll stiffness')
    model['assumptions'] = assumptions
    model['wheel_rates'] = {'spring': k_wheel, 'bump_stop': k_stop_wheel,
                            'arb': arb, 'motion_ratio': mr}
    return model


def _ac_rates(section, model, mr, assumptions):
    """Put the AC wheel rates (STRUT, DWB) on the spring line of ``model``.

    :return: the wheel spring rate and the wheel stop rate (N/m)
    """
    k_wheel = float(section['SPRING_RATE'])
    model['spring']['k'] = k_wheel / mr / mr / 1000.0
    model['spring']['damper'] = _damper(section, mr)
    assumptions += [
        'droop stop rate = BUMP_STOP_RATE',
        'PROGRESSIVE_SPRING_RATE left out (< 1.5 % at 50 mm)',
    ]
    bump = float(section['BUMPSTOP_UP']) * 1000.0
    droop = -float(section['BUMPSTOP_DN']) * 1000.0
    if droop >= 0.0:
        # AC puts this droop stop at the design pose: the wheel cannot drop
        # below the static height.  The stop needs a travel below static.
        assumptions.append(f'BUMPSTOP_DN={section["BUMPSTOP_DN"].strip()} '
                           'puts the droop stop at the static pose; the '
                           'model uses 1 mm below it')
        droop = -1.0
    k_stop = float(section['BUMP_STOP_RATE']) / mr / mr / 1000.0
    model['stops'] = {'bump_travel': bump, 'droop_travel': droop,
                      'k': k_stop,
                      'note': 'AC BUMPSTOP_UP/DN and BUMP_STOP_RATE (wheel '
                              'rate / mr^2)'}
    return k_wheel, float(section['BUMP_STOP_RATE'])


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
    }


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
