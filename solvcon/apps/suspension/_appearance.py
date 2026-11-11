# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Procedural rubber texture and a smooth, UV-mapped tire surface."""

import functools

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QImage

from . import _scene


@functools.lru_cache(maxsize=1)
def rubber_texture():
    width, height = 2048, 1024
    across, around = np.mgrid[0:height, 0:width]
    across = across / (height - 1)
    around = around / width
    grain = np.random.default_rng(7).normal(0.0, 1.8, (height, width))
    tread = (across > 0.12) & (across < 0.88)
    cuts = np.mod(around * 64 + abs(across - 0.5) * 5, 1.0) < 0.045
    shade = 48 + grain + 3 * np.sin(around * np.pi * 320)
    shade -= 19 * (cuts & tread)
    rgba = np.empty((height, width, 4), dtype='uint8')
    rgba[:, :, :3] = np.clip(shade[:, :, None], 0, 255).astype('uint8')
    rgba[:, :, 3] = 255
    image = QImage(rgba.data, width, height, width * 4,
                   QImage.Format_RGBA8888)
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    if not image.save(buffer, 'PNG'):
        raise RuntimeError('Cannot encode the rubber texture')
    return bytes(buffer.data())


def tire_surface(geometry):
    radius, rim, half = geometry.wheel_sizes()
    columns = _scene.tire_columns(radius, rim, 2 * half, ncol=64)
    profile = np.array([(pos, outer) for pos, _, outer in columns]
                       + [(pos, inner) for pos, inner, _ in reversed(columns)],
                       dtype='float64')
    return _revolved(profile, geometry.model.spin_axis)


def _revolved(profile, direction, segments=192):
    profile = np.asarray(profile, dtype='float64')
    tangent = np.roll(profile, -1, axis=0) - np.roll(profile, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    axis, unit1, unit2 = _scene._frame(direction)
    angle = np.linspace(0, 2 * np.pi, segments + 1, dtype='float64')
    radial = np.cos(angle)[:, None] * unit1 + np.sin(angle)[:, None] * unit2
    positions = (profile[:, 0, None, None] * axis
                 + profile[:, 1, None, None] * radial)
    normals = (tangent[:, 0, None, None] * radial
               - tangent[:, 1, None, None] * axis)
    uv = np.empty((len(profile), segments + 1, 2), dtype='float32')
    uv[:, :, 0] = angle / (2 * np.pi)
    span = np.ptp(profile[:, 0])
    uv[:, :, 1] = (profile[:, 0, None] - profile[:, 0].min()) / span
    rings = len(profile)
    grid = np.arange(rings * (segments + 1), dtype='uint32').reshape(
        rings, segments + 1)
    first = grid[:, :-1].ravel()
    second = np.roll(grid, -1, axis=0)[:, :-1].ravel()
    indices = np.column_stack((first, second, first + 1,
                               first + 1, second, second + 1))
    return (np.asarray(positions.reshape(-1, 3), dtype='float32'),
            np.asarray(normals.reshape(-1, 3), dtype='float32'),
            uv.reshape(-1, 2), indices.reshape(-1, 3))


def rim_surface(geometry):
    radius, rim, half = geometry.wheel_sizes()
    axis, unit1, unit2 = _scene._frame(geometry.model.spin_axis)
    inside = rim - geometry.BARREL_DEPTH
    flange = rim + geometry.FLANGE_HEIGHT
    edge = half + geometry.FLANGE_WIDTH
    profile = [(-edge, inside), (-edge, flange - 3),
               (-edge + 3, flange), (-half, flange),
               (-half + 3, rim - 12), (half - 3, rim - 12),
               (half, flange), (edge - 3, flange),
               (edge, flange - 3), (edge, inside)]
    pieces = [_revolved(profile, axis)]
    face = geometry.outboard() * 0.45 * half
    hub = 0.2 * radius
    pieces.append(_revolved([(face - 20, hub - 3), (face - 17, hub),
                             (face + 15, hub), (face + 18, hub - 3),
                             (face + 18, 0.08 * radius),
                             (face - 20, 0.08 * radius)], axis))
    for index in range(10):
        angle = 2 * np.pi * index / 10
        radial = np.cos(angle) * unit1 + np.sin(angle) * unit2
        start = face * axis + 0.9 * hub * radial
        end = ((face - geometry.outboard() * 0.35 * half) * axis
               + (inside + 3) * radial)
        direction = end - start
        length = np.linalg.norm(direction)
        part = _revolved([(0, 3), (3, 7), (length - 3, 7),
                          (length, 3), (length, 0), (0, 0)], direction, 24)
        pieces.append((part[0] + start, *part[1:]))
    positions, normals, indices = [], [], []
    offset = 0
    for points, normal, _, triangles in pieces:
        positions.append(points)
        normals.append(normal)
        indices.append(triangles + offset)
        offset += len(points)
    return (np.asarray(np.concatenate(positions), dtype='float32'),
            np.asarray(np.concatenate(normals), dtype='float32'),
            np.asarray(np.concatenate(indices), dtype='uint32'))


def link_surface(ends, radius=4.0):
    positions, normals, indices = [], [], []
    offset = 0
    for start, end in zip(ends[::2], ends[1::2]):
        start = np.asarray(start, dtype='float64')
        direction = np.asarray(end, dtype='float64') - start
        length = np.linalg.norm(direction)
        if length < 1.e-6:
            continue
        collar = min(radius * 2, length * 0.2)
        points, normal, _, triangles = _revolved(
            [(0, radius * 0.7), (0, radius * 1.4),
             (collar, radius * 1.4), (collar, radius),
             (length - collar, radius), (length - collar, radius * 1.4),
             (length, radius * 1.4), (length, radius * 0.7)], direction, 16)
        positions.append(points + start)
        normals.append(normal)
        indices.append(triangles + offset)
        offset += len(points)
    if not positions:
        return None
    return (np.asarray(np.concatenate(positions), dtype='float32'),
            np.asarray(np.concatenate(normals), dtype='float32'),
            np.asarray(np.concatenate(indices), dtype='uint32'))


def spring_surface(ends):
    positions, normals, indices = [], [], []
    offset = 0
    for start, end in zip(ends[::2], ends[1::2]):
        start = np.asarray(start, dtype='float64')
        direction = np.asarray(end, dtype='float64') - start
        length = np.linalg.norm(direction)
        if length < 1.e-6:
            continue
        _, u, v = _scene._frame(direction)
        t = np.linspace(0, 1, 145, dtype='float64')
        angle = t * (12 * np.pi)
        radial = np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v
        center = start + t[:, None] * direction + 13 * radial
        tangent = (direction + 13 * 12 * np.pi
                   * (-np.sin(angle)[:, None] * u
                      + np.cos(angle)[:, None] * v))
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        binormal = np.cross(tangent, radial)
        section = np.linspace(0, 2 * np.pi, 9, dtype='float64')
        normal = (np.cos(section)[None, :, None] * radial[:, None, :]
                  + np.sin(section)[None, :, None] * binormal[:, None, :])
        points = center[:, None, :] + 2.5 * normal
        grid = np.arange(len(t) * len(section), dtype='uint32').reshape(
            len(t), len(section))
        a = grid[:-1, :-1].ravel()
        b = grid[1:, :-1].ravel()
        triangles = np.column_stack((a, b, a + 1, a + 1, b, b + 1))
        positions.append(points.reshape(-1, 3))
        normals.append(normal.reshape(-1, 3))
        indices.append(triangles.reshape(-1, 3) + offset)
        offset += points.shape[0] * points.shape[1]
    if not positions:
        return None
    return (np.asarray(np.concatenate(positions), dtype='float32'),
            np.asarray(np.concatenate(normals), dtype='float32'),
            np.asarray(np.concatenate(indices), dtype='uint32'))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
