# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Full-vehicle multibody step-steer run for the research demonstration."""

import numpy as np
import csv
from .vehicle_candidate import vehicle_model

CORNERS = ('front_left', 'front_right', 'rear_left', 'rear_right')


def run(
    data_dir, speed=12.0, rack=-0.004, duration=3.0, dt=0.002, pose_dt=0.02
):
    from solvcon import multibody
    from solvcon.multibody.suspension import car

    result = car.step_steer(
        multibody,
        vehicle_model(data_dir),
        speed,
        rack,
        start=0.5,
        ramp=0.3,
        duration=duration,
        dt=dt,
        pose_dt=pose_dt,
    )
    if not result['poses'] or not np.isfinite(result['yaw_rate']).all():
        raise RuntimeError('Multibody step-steer run failed')
    return result


def channels(poses):
    """Calculate body angles and wheel vertical travel relative to chassis.

    Coordinates are relative to the first sampled pose, in metres/radians;
    wheel bump is positive upward in the chassis frame.  Pose is provided
    directly by the solvcon multibody solver and is not exaggerated.
    """
    from solvcon.multibody.suspension.metrics import rot

    values = []
    for pose in poses:
        position, quat = pose['chassis']
        matrix = rot(quat)
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2]))
        travel = []
        for corner in CORNERS:
            wheel, _ = pose['corners'][corner]['upright']
            travel.append((matrix.T @ (wheel - position))[2])
        values.append([position[2], roll, pitch, *travel])
    values = np.asarray(values)
    return values - values[0]


def export_csv(result, path):
    time = result['pose_times']
    values = channels(result['poses'])
    speed = np.interp(time, result['time'], result['speed'])
    yaw = np.interp(time, result['time'], result['yaw_rate'])
    steer = np.interp(time, result['time'], result['steer'][:, 0])
    header = [
        'time_s',
        'speed_m_s',
        'yaw_rate_rad_s',
        'front_steer_rad',
        'heave_m',
        'roll_rad',
        'pitch_rad',
        'front_left_travel_m',
        'front_right_travel_m',
        'rear_left_travel_m',
        'rear_right_travel_m',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(np.column_stack((time, speed, yaw, steer, values)))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
