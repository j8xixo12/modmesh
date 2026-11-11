# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The reader of RaceChrono CSV exports (``suspension.racechrono``) on the
synthetic logs of tests/data/multibody/racechrono_synthetic (a slalom at
20 m/s: r = 0.3 sin(pi t) rad/s, a_y = V r).

* CSV v3: the metadata lines, the keys, units, and sources rows; G and
  deg/s to SI; the missing gyro samples filled.
* The older single header with the units in parentheses.
* The signs: a_y follows V r of the GPS bearing (the file has it + to the
  right), a_x follows dV/dt, the gyro yaw rate is checked against the GPS.
* The layout of a real export (RaceChrono Pro 10.2, phone GPS and CAN bus):
  a bearing that updates once a second and is repeated in between, and
  accelerations from ``calc`` and from the car sensors with opposite signs.
"""

import csv
import math
import os
import re
import tempfile

import numpy as np
import pytest

from solvcon.multibody.suspension import racechrono as rc

DATA = os.path.join(os.path.dirname(__file__), 'data', 'multibody',
                    'racechrono_synthetic')


def _slalom(t):
    v = 20.0 + 0.3 * t
    return v, 0.3 * np.sin(math.pi * t)


def test_reads_csv_v3():
    log = rc.read_csv(os.path.join(DATA, 'session_v3.csv'))
    assert 3 == log['format']
    assert 'Synthetic slalom' == log['meta']['Session title']
    keys = [key for key, _, _ in log['columns']]
    assert 'timestamp' == keys[0]
    j = keys.index('lateral_acc')
    assert ('G', 'calc') == log['columns'][j][1:]
    m = rc.vehicle_motion(log)
    t = m['time']
    assert t[0] == 0.0 and t[-1] == pytest.approx(10.0)
    v, r = _slalom(t)
    np.testing.assert_allclose(m['ay'], v * r, atol=1e-3)
    np.testing.assert_allclose(m['ax'], 0.3, atol=1e-3)
    np.testing.assert_allclose(m['speed'], v, atol=1e-3)
    # The gyro misses every fourth sample: filled by interpolation.
    assert np.all(np.isfinite(m['yaw_rate']))
    np.testing.assert_allclose(m['yaw_rate'], r, atol=5e-3)
    # The bearing wraps from 0 to 360 deg: the GPS yaw rate stays smooth.
    np.testing.assert_allclose(m['gps_yaw_rate'][1:-1], r[1:-1], atol=5e-3)
    notes = " ".join(m['notes'])
    assert "a_y sign from V r" in notes and "flipped" in notes
    assert "gyro yaw rate matches" in notes


def test_reads_the_single_header_export():
    log = rc.read_csv(os.path.join(DATA, 'session_v2.csv'))
    assert 2 == log['format']
    m = rc.vehicle_motion(log)
    v, r = _slalom(m['time'])
    np.testing.assert_allclose(m['speed'], v, atol=1e-3)
    np.testing.assert_allclose(m['ay'], v * r, atol=1e-3)
    assert 'roll_rate' not in m


def test_gyro_mount_and_a_wrong_yaw_sign():
    log = rc.read_csv(os.path.join(DATA, 'session_v3.csv'))
    # The device z axis down: the yaw rate opposes the GPS one.
    m = rc.vehicle_motion(log, gyro_signs=(1, 1, -1))
    assert any("opposes the GPS" in n for n in m['notes'])
    flat = rc.vehicle_motion(log, gyro_axes=('y', 'x', 'z'),
                             gyro_signs=(1, -1, 1))
    full = rc.vehicle_motion(log)
    np.testing.assert_allclose(flat['pitch_rate'], -full['roll_rate'])


def _write_phone_and_can_log(path):
    """120 s of a slow slalom at 50 Hz rows: the GPS bearing holds its value
    for 1 s to 3 s between fixes; lateral_acc from ``calc`` (+ right) and
    from ``200: canbus`` (+ left), both with vibration noise; the CAN bus
    yaw rate."""
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 120.0, 0.02)
    v = 25.0 + 0.0 * t
    r = 0.25 * np.sin(2.0 * math.pi * t / 12.0)
    bearing = (40.0 - np.degrees(np.cumsum(r) * 0.02)) % 360.0
    fixes = np.cumsum(rng.uniform(1.0, 3.0, t.size))
    fixes = np.concatenate([[0.0], fixes[fixes < t[-1]]])
    last = np.searchsorted(fixes, t, side='right') - 1
    held = bearing[np.round(fixes[last] / 0.02).astype(int)]
    ay = v * r + rng.normal(0.0, 6.0, t.size)
    calc = -ay / rc.STANDARD_GRAVITY
    rows = [['Format', '3'], [],
            ['timestamp', 'elapsed_time', 'bearing', 'speed',
             'lateral_acc', 'lateral_acc', 'longitudinal_acc', 'yaw_rate'],
            ['unix time', 's', 'deg', 'm/s', 'G', 'G', 'G', 'deg/s'],
            ['', '', '100: gps', '100: gps', 'calc', '200: canbus',
             '200: canbus', '200: canbus']]
    for k in range(t.size):
        rows.append([f'{1.7e9 + t[k]:.2f}', f'{t[k]:.2f}', f'{held[k]:.3f}',
                     f'{v[k]:.3f}', f'{calc[k]:.4f}',
                     f'{ay[k] / rc.STANDARD_GRAVITY:.5f}', '0.0',
                     f'{math.degrees(r[k]):.4f}'])
    with open(path, 'w', newline='') as fobj:
        csv.writer(fobj).writerows(rows)
    return t, ay


def test_a_held_bearing_and_two_sources():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'phone_can.csv')
        t, ay = _write_phone_and_can_log(path)
        m = rc.vehicle_motion(rc.read_csv(path))
    notes = " | ".join(m['notes'])
    # The car sensor wins over the GPS-derived channel, as logged (+ left).
    assert "lateral_acc: 2 columns (calc, 200: canbus); using 200: canbus" \
        in notes
    np.testing.assert_allclose(m['ay'], ay, atol=1e-4)
    # The GPS rates come from the fixes: the sign checks agree.
    found = re.search(r"a_y sign from V r of the GPS bearing \(agreement "
                      r"([0-9.]+)\)", notes)
    assert found and float(found.group(1)) > 0.9, notes
    found = re.search(r"gyro yaw rate matches the GPS one \(correlation "
                      r"([0-9.]+)\)", notes)
    assert found and float(found.group(1)) > 0.9, notes


def test_rejects_a_file_without_a_time_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'x.csv')
        with open(path, 'w', newline='') as fobj:
            csv.writer(fobj).writerows([['a', 'b'], ['1', '2']])
        with pytest.raises(ValueError):
            rc.read_csv(path)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
