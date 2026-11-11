# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Read a RaceChrono CSV export and take the vehicle motion from it.

RaceChrono publishes no specification of its CSV export; this reader
follows what the third-party importers of the format do:

* CSV v3 ("Format 3"): metadata lines, then a row of channel keys that
  starts with ``timestamp``, then a row of units and a row of sources,
  then the samples (one row per sample, empty cells for a channel without
  a sample at that time).
* The older single-header export: one row of display names with the unit
  in parentheses, for example ``Lateral acceleration (G)``.

So the reader finds the header row by its first cell, takes the units from
a units row or from the parentheses, and finds a channel by a list of
names (:data:`ALIASES`); every name is normalized (lower case, the unit in
parentheses dropped, spaces to underscores).  :func:`read_csv` returns
the columns in SI units; :func:`vehicle_motion` takes the accelerations
and rates in the axes of :mod:`suspension.car` (x forward, y left, z up)
and checks the signs with the GPS: the lateral acceleration must follow V
r with the yaw rate r = -d(bearing)/dt (the bearing turns clockwise), and
the longitudinal acceleration must follow dV/dt.

A real export (RaceChrono Pro 10.2, a phone GPS and the CAN bus of the car)
shows what the checks must stand:

* The GPS bearing updates about once a second; the rows between two fixes
  repeat it.  The GPS rates come from the fixes only (:func:`_gps_rate`),
  and a check compares the two signals low-passed over
  :data:`CHECK_WINDOW` (:func:`_lowpass`).
* A channel comes more than once, for example ``lateral_acc`` from
  ``calc`` (derived from the GPS) and from ``200: canbus`` (the sensor of
  the car), with opposite signs.  A measured channel wins over a ``calc``
  one (:func:`_find`).
"""

import csv
import math
import re

import numpy as np

__all__ = [
    'ALIASES',
    'read_csv',
    'vehicle_motion',
]

#: The names a channel goes by, normalized (see :func:`_key`).
ALIASES = {
    'time': ('timestamp', 'time', 'elapsed_time'),
    'speed': ('speed', 'gps_speed'),
    'bearing': ('bearing', 'heading', 'gps_bearing'),
    'longitudinal_acc': ('longitudinal_acc', 'longitudinal_acceleration',
                         'longitudinal_g'),
    'lateral_acc': ('lateral_acc', 'lateral_acceleration', 'lateral_g'),
    'x_rate': ('x_rate_of_rotation', 'x_rotation_rate', 'gyro_x',
               'x_gyro', 'rotation_rate_x'),
    'y_rate': ('y_rate_of_rotation', 'y_rotation_rate', 'gyro_y',
               'y_gyro', 'rotation_rate_y'),
    'z_rate': ('z_rate_of_rotation', 'z_rotation_rate', 'gyro_z',
               'z_gyro', 'rotation_rate_z', 'yaw_rate'),
}

STANDARD_GRAVITY = 9.80665

#: Unit (normalized) to (scale, offset) into SI.
_UNITS = {
    '': 1.0, 's': 1.0, 'ms': 1e-3,
    'm/s': 1.0, 'km/h': 1.0 / 3.6, 'kph': 1.0 / 3.6, 'mph': 0.44704,
    'g': STANDARD_GRAVITY, 'm/s2': 1.0, 'm/s^2': 1.0, 'm/s²': 1.0,
    'deg/s': math.pi / 180.0, '°/s': math.pi / 180.0, 'rad/s': 1.0,
    'deg': math.pi / 180.0, '°': math.pi / 180.0, 'rad': 1.0,
    'm': 1.0, 'km': 1e3,
}


def _key(name):
    """A column name without its unit, in lower case, with underscores."""
    name = re.sub(r'\(.*?\)', '', name).strip().lower()
    return re.sub(r'[\s\-]+', '_', name)


def _unit_of(name):
    found = re.search(r'\(([^)]*)\)', name)
    return found.group(1).strip() if found else ''


def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path):
    """Read a RaceChrono CSV export.

    :return: dict with ``columns`` (``[(key, unit, source)]`` in file
        order), ``data`` (``{index: array}`` in SI units, NaN where a
        channel has no sample), ``meta`` (the metadata lines as ``{name:
        value}``), and ``format`` (3 with a units row, else 2)
    :raises ValueError: without a header row that starts with a time
        column, or with a unit it does not know
    """
    with open(path, newline='', encoding='utf-8-sig') as fobj:
        rows = list(csv.reader(fobj))
    head = None
    for k, row in enumerate(rows):
        if row and _key(row[0]) in ALIASES['time'] and len(row) >= 2:
            head = k
            break
    if head is None:
        raise ValueError(f"{path}: no header row that starts with a time "
                         "column")
    meta = {row[0].strip(): ",".join(row[1:]).strip()
            for row in rows[:head] if row and row[0].strip()}
    names = rows[head]
    # The rows after the header that are not numbers: units, then sources.
    extra = []
    k = head + 1
    while k < len(rows) and rows[k] and _number(rows[k][0]) is None \
            and len(extra) < 2:
        extra.append(rows[k])
        k += 1
    units = extra[0] if extra else [_unit_of(n) for n in names]
    sources = extra[1] if len(extra) > 1 else [''] * len(names)
    body = [row for row in rows[k:] if row and _number(row[0]) is not None]
    columns, data = [], {}
    for j, name in enumerate(names):
        unit = units[j].strip() if j < len(units) else ''
        norm = unit.lower().replace(' ', '')
        if norm not in _UNITS:
            # A channel of no interest may have any unit; it is kept raw.
            scale = None
        else:
            scale = _UNITS[norm]
        values = np.array([
            (_number(row[j]) if j < len(row) and row[j].strip() else None)
            for row in body], dtype='float64')
        if scale is not None:
            values = values * scale
        columns.append((_key(name), unit,
                        sources[j].strip() if j < len(sources) else ''))
        data[j] = values
    return {'columns': columns, 'data': data, 'meta': meta,
            'format': 3 if extra else 2}


def _find(log, channel, prefer=None):
    """The column index of a channel, or None.  With several columns of
    the channel the one whose source contains ``prefer`` wins, else the
    first measured one (a source that is not ``calc``), else the first."""
    hits = [j for j, (key, _, _) in enumerate(log['columns'])
            if key in ALIASES[channel]]
    if not hits:
        return None
    if prefer:
        for j in hits:
            if prefer in log['columns'][j][2].lower():
                return j
    for j in hits:
        if 'calc' != log['columns'][j][2].strip().lower():
            return j
    return hits[0]


def _fill(time, values):
    """Interpolate the missing samples (NaN) of a channel over time."""
    good = np.isfinite(values)
    if good.sum() < 2:
        return None
    return np.interp(time, time[good], values[good])


#: The smallest |agreement| that sets a sign from the GPS.
MIN_AGREEMENT = 0.3
#: The low-pass window (s) of the sign checks.
CHECK_WINDOW = 1.0


def _gps_rate(time, values):
    """The time derivative of a GPS channel from its fixes: the rows that
    repeat the last value until the next fix are left out."""
    keep = np.concatenate([[True], np.diff(values) != 0.0])
    if keep.sum() < 3:
        return np.gradient(values, time)
    return np.interp(time, time[keep], np.gradient(values[keep],
                                                   time[keep]))


def _lowpass(time, values, window=CHECK_WINDOW):
    """The moving mean of a signal over ``window`` (s), on any time
    spacing (the difference of its running integral)."""
    integral = np.concatenate([[0.0], np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(time))])
    lo = np.clip(time - 0.5 * window, time[0], time[-1])
    hi = np.clip(time + 0.5 * window, time[0], time[-1])
    return (np.interp(hi, time, integral)
            - np.interp(lo, time, integral)) / (hi - lo)


def _agreement(a, b):
    """The correlation of two signals; for a constant one, the cosine of
    the signals themselves (their means then decide the sign)."""
    ac, bc = a - a.mean(), b - b.mean()
    den = math.sqrt(float(ac @ ac) * float(bc @ bc))
    if den > 1e-12 * max(1.0, float(a @ a) * float(b @ b)):
        return float(ac @ bc) / den
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0.0 else 0.0


def _signed(time, values, reference, name, what, notes):
    """``values`` with the sign that agrees with ``reference`` (both
    low-passed for the check)."""
    c = _agreement(_lowpass(time, values), _lowpass(time, reference))
    if abs(c) < MIN_AGREEMENT:
        notes.append(f"{name}: the agreement with {what} is weak ({c:+.2f}); "
                     "the logged sign is kept")
        return values
    notes.append(f"{name} sign from {what} (agreement {abs(c):.2f}"
                 + (", the logged sign is flipped)" if c < 0.0 else ")"))
    return -values if c < 0.0 else values


def vehicle_motion(log, gyro_axes=('x', 'y', 'z'), gyro_signs=(1, 1, 1),
                   prefer_source=None):
    """The motion of the vehicle in the axes of :mod:`suspension.car` (x
    forward, y left, z up).

    The accelerations take their signs from the GPS when the speed and the
    bearing are there (see the module docstring); the sign checks go to
    ``notes``.  The gyro is in the axes of the logging device: give the
    device axes along the vehicle x, y, z in ``gyro_axes`` and their signs
    in ``gyro_signs`` (a phone mounted in landscape, for example, has its
    x axis across the car).  The z sign is checked against the GPS yaw
    rate.

    :param prefer_source: a source name (for example ``acc`` or ``calc``)
        to pick among duplicate channels
    :return: dict with ``time`` (s from the first sample), ``ax``, ``ay``
        (m/s^2), and when present ``speed`` (m/s), ``roll_rate``,
        ``pitch_rate``, ``yaw_rate`` (rad/s, about the vehicle x, y, z),
        ``gps_yaw_rate``; and ``notes`` (list of str)
    :raises ValueError: without a time or a lateral acceleration channel
    """
    data = log['data']
    j = _find(log, 'time')
    time = data[j]
    order = np.argsort(time, kind='stable')
    good = np.isfinite(time[order])
    order = order[good]
    time = time[order]
    keep = np.concatenate([[True], np.diff(time) > 0.0])
    order, time = order[keep], time[keep]
    notes = []

    def channel(name):
        j = _find(log, name, prefer_source)
        if j is None:
            return None
        sources = [src for key, _, src in log['columns']
                   if key in ALIASES[name]]
        if len(sources) > 1:
            notes.append(f"{name}: {len(sources)} columns "
                         f"({', '.join(sources)}); using "
                         f"{log['columns'][j][2]}")
        values = data[j][order].copy()
        if 'bearing' == name:
            # Unwrap before the gaps are filled: 359 deg to 1 deg is +2.
            good = np.isfinite(values)
            values[good] = np.unwrap(values[good])
        return _fill(time, values)

    out = {'time': time - time[0], 'notes': notes}
    ay = channel('lateral_acc')
    if ay is None:
        raise ValueError("the log has no lateral acceleration channel")
    ax = channel('longitudinal_acc')
    if ax is None:
        ax = np.zeros_like(ay)
        notes.append("no longitudinal acceleration channel: a_x = 0")
    speed = channel('speed')
    bearing = channel('bearing')
    if speed is not None:
        out['speed'] = speed
        ax = _signed(time, ax, _gps_rate(time, speed), "a_x", "dV/dt",
                     notes)
    if speed is not None and bearing is not None:
        r_gps = -_gps_rate(time, bearing)
        out['gps_yaw_rate'] = r_gps
        ay = _signed(time, ay, speed * r_gps, "a_y",
                     "V r of the GPS bearing", notes)
    else:
        notes.append("no GPS speed and bearing: the a_y sign is taken as "
                     "+ to the left (assumed)")
    out['ax'], out['ay'] = ax, ay
    rates = {}
    for axis in ('x', 'y', 'z'):
        values = channel(f'{axis}_rate')
        if values is not None:
            rates[axis] = values
    names = ('roll_rate', 'pitch_rate', 'yaw_rate')
    for name, axis, sign in zip(names, gyro_axes, gyro_signs):
        if axis in rates:
            out[name] = sign * rates[axis]
    if 'yaw_rate' in out and 'gps_yaw_rate' in out:
        c = _agreement(_lowpass(time, out['yaw_rate']),
                       _lowpass(time, out['gps_yaw_rate']))
        if c < 0.0:
            notes.append(f"the gyro yaw rate opposes the GPS one "
                         f"(correlation {c:.2f}): check gyro_axes and "
                         "gyro_signs")
        else:
            notes.append(f"gyro yaw rate matches the GPS one (correlation "
                         f"{c:.2f})")
    if rates:
        notes.append(f"gyro axes (device axes along vehicle x, y, z): "
                     f"{gyro_axes}, signs {gyro_signs} (assumed)")
    return out

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
