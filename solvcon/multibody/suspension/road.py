# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Road profiles, tire enveloping, and the road input of the quarter car.

A road profile is the road height z(x) along the travel direction x (m).
A tire does not follow it point by point: over an obstacle shorter than its
contact patch it envelops the obstacle, and the force it feels comes from an
*effective road plane* with a height w and a forward slope beta.  The slope
tilts the force, so a tire climbing a bump is pushed back and one leaving it
is pushed forward.

:class:`TandemCam` is the enveloping model of Schmeitz (2004): two elliptical
cams, a distance l_s apart, follow the profile; each cam center sits at

    Z = max over |x| <= a_e of [z(X + x) + z_e(x)],
    z_e(x) = b_e (1 - (|x| / a_e)^c_e)^(1 / c_e),

and the effective plane is w = (Z_front + Z_rear) / 2 - b_e, beta =
atan((Z_front - Z_rear) / l_s), with l_s = p_ls * 2a and 2a the contact
length.  The default shape is the one fitted for a Goodyear 205/60R15
(r0 = 0.3135 m: a_e = 0.3626 m, b_e = 0.3580 m, c_e = 1.7359, p_ls = 0.8;
Sivaramakrishnan, Virginia Tech MS thesis 2013, Table 5.11, after Schmeitz),
scaled by the free radius of the tire in use; that scaling is an assumption
of this module, not a fitted result.

:func:`road_input` turns a profile and a speed into the time functions h(t)
and b(t) that the engine tire (``create_tire(road_height=, road_slope=)``)
uses as its road plane.  A shaker rig (the whole road moving up and down) is
:func:`rig_input`.
"""

import csv
import math

import numpy as np

__all__ = [  # noqa: F822
    'RoadInput',
    'RoadProfile',
    'TandemCam',
    'rig_input',
    'road_input',
]

#: The tandem-cam shape fitted for a 205/60R15 tire and its free radius.
REFERENCE_RADIUS = 0.3135
REFERENCE_SHAPE = {'a_e': 0.3626, 'b_e': 0.3580, 'c_e': 1.7359, 'p_ls': 0.8}


class RoadProfile(object):
    """Road height z(x) (m) along the travel direction, flat outside the
    feature.

    :ivar name: A short description for plots and reports.
    :ivar x: The sample positions (m), increasing.
    :ivar z: The heights at the samples (m).
    """

    def __init__(self, name, x, z):
        x = np.asarray(x, dtype='float64')
        z = np.asarray(z, dtype='float64')
        if x.ndim != 1 or x.shape != z.shape or x.size < 2:
            raise ValueError("a road profile needs matching x and z "
                             "samples, at least two")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("road profile x must increase strictly")
        self.name = name
        self.x = x
        self.z = z

    def height(self, x):
        """Heights at the positions x; the ends extend flat."""
        return np.interp(np.asarray(x, dtype='float64'), self.x, self.z)

    @property
    def start(self):
        return float(self.x[0])

    @property
    def end(self):
        return float(self.x[-1])

    @classmethod
    def flat(cls):
        return cls("flat road", [0.0, 1.0], [0.0, 0.0])

    @classmethod
    def bump(cls, height, length, n=401):
        """A smooth bump (a stone or a speed hump): z = h (1 - cos) / 2
        over ``length``, starting at x = 0."""
        x = np.linspace(0.0, length, n)
        z = 0.5 * height * (1.0 - np.cos(2.0 * math.pi * x / length))
        return cls(f"bump {height * 1e3:g} mm x {length * 1e3:g} mm", x, z)

    @classmethod
    def pothole(cls, depth, length, edge=0.0):
        """A pothole of ``depth`` (positive, m) and ``length`` at the
        bottom, with sloped edges of horizontal extent ``edge`` (0 for a
        sharp edge), starting at x = 0."""
        if depth <= 0.0 or length <= 0.0:
            raise ValueError("a pothole needs a positive depth and length")
        edge = max(float(edge), 1e-4)
        x = [0.0, edge, edge + length, 2.0 * edge + length]
        z = [0.0, -depth, -depth, 0.0]
        return cls(f"pothole {depth * 1e3:g} mm x {length * 1e3:g} mm",
                   x, z)

    @classmethod
    def step(cls, height, ramp=1e-4):
        """A step of ``height`` at x = 0 (up if positive), over a ramp of
        horizontal extent ``ramp``; the road stays at the new height."""
        return cls(f"step {height * 1e3:+g} mm", [0.0, ramp],
                   [0.0, float(height)])

    @classmethod
    def sine(cls, amplitude, wavelength, cycles=10, n_per_cycle=64):
        """A sine road z = A sin(2 pi x / wavelength) over ``cycles``."""
        x = np.linspace(0.0, cycles * wavelength, cycles * n_per_cycle + 1)
        z = amplitude * np.sin(2.0 * math.pi * x / wavelength)
        return cls(f"sine road {amplitude * 1e3:g} mm, wavelength "
                   f"{wavelength:g} m", x, z)

    @classmethod
    def from_csv(cls, path):
        """Read a profile with the header ``x_mm,z_mm`` (or ``x_m,z_m``)."""
        with open(path, newline='', encoding='utf-8') as fobj:
            rows = [row for row in csv.reader(
                line for line in fobj
                if line.strip() and not line.lstrip().startswith('#'))]
        if not rows:
            raise ValueError(f"{path}: no header row")
        header = [h.strip().lower() for h in rows[0]]
        for xname, zname, scale in (('x_mm', 'z_mm', 1e-3),
                                    ('x_m', 'z_m', 1.0)):
            if xname in header and zname in header:
                ix, iz = header.index(xname), header.index(zname)
                break
        else:
            raise ValueError(f"{path}: the header needs x_mm,z_mm or "
                             f"x_m,z_m, not {','.join(rows[0])}")
        data = []
        for number, row in enumerate(rows[1:], start=2):
            try:
                data.append((float(row[ix]) * scale, float(row[iz]) * scale))
            except (ValueError, IndexError):
                raise ValueError(f"{path}: line {number} is not a pair of "
                                 f"numbers: {row}") from None
        data = np.array(data)
        return cls(f"profile {path}", data[:, 0], data[:, 1])


class TandemCam(object):
    """The tandem elliptical cam enveloping model (Schmeitz 2004).

    :ivar a_e: Cam half length (m).
    :ivar b_e: Cam half height (m).
    :ivar c_e: Cam shape exponent.
    :ivar l_s: Cam spacing (m).
    """

    #: The cam scan step (m); the thesis bound is 2 cm.
    SCAN_STEP = 1e-3

    def __init__(self, a_e, b_e, c_e, l_s):
        if min(a_e, b_e, c_e, l_s) <= 0.0:
            raise ValueError("tandem cam parameters must be positive")
        self.a_e = float(a_e)
        self.b_e = float(b_e)
        self.c_e = float(c_e)
        self.l_s = float(l_s)

    @classmethod
    def for_tire(cls, radius, deflection):
        """The reference shape scaled to a tire of free radius ``radius`` at
        the static ``deflection``: the contact half length is the chord of
        the free circle at that deflection."""
        scale = radius / REFERENCE_RADIUS
        half_contact = math.sqrt(max(2.0 * radius * deflection
                                     - deflection ** 2, 0.0))
        return cls(REFERENCE_SHAPE['a_e'] * scale,
                   REFERENCE_SHAPE['b_e'] * scale, REFERENCE_SHAPE['c_e'],
                   REFERENCE_SHAPE['p_ls'] * 2.0 * half_contact)

    def cam_offset(self, x):
        """z_e(x): the height of the cam center above its contact point at
        the local position x (|x| <= a_e)."""
        r = np.clip(np.abs(np.asarray(x, dtype='float64')) / self.a_e,
                    0.0, 1.0)
        return self.b_e * (1.0 - r ** self.c_e) ** (1.0 / self.c_e)

    def cam_height(self, profile, center):
        """Global height of a cam centered at ``center`` (array)."""
        center = np.atleast_1d(np.asarray(center, dtype='float64'))
        n = int(math.ceil(self.a_e / self.SCAN_STEP))
        local = np.linspace(-self.a_e, self.a_e, 2 * n + 1)
        offsets = self.cam_offset(local)
        heights = profile.height(center[:, None] + local[None, :])
        return np.max(heights + offsets[None, :], axis=1)

    def effective(self, profile, x):
        """The effective plane (w, beta) at wheel positions x (arrays)."""
        x = np.atleast_1d(np.asarray(x, dtype='float64'))
        front = self.cam_height(profile, x + 0.5 * self.l_s)
        rear = self.cam_height(profile, x - 0.5 * self.l_s)
        w = 0.5 * (front + rear) - self.b_e
        beta = np.arctan((front - rear) / self.l_s)
        return w, beta

    @property
    def reach(self):
        """How far ahead of the wheel center the model feels the road."""
        return self.a_e + 0.5 * self.l_s


class RoadInput(object):
    """A road input of the quarter car as time functions.

    :ivar kind: ``road`` (a profile at a speed) or ``rig`` (a moving plate).
    :ivar name: A description for plots.
    :ivar times: Sample times (s).
    :ivar height: Effective road height at the samples (m).
    :ivar slope: Effective forward slope at the samples (rad).
    :ivar position: Wheel position along the profile at the samples (m),
        or None for a rig.
    """

    def __init__(self, kind, name, times, height, slope, position=None,
                 profile=None, cam=None, speed=None):
        self.kind = kind
        self.name = name
        self.times = np.asarray(times, dtype='float64')
        self.height = np.asarray(height, dtype='float64')
        self.slope = np.asarray(slope, dtype='float64')
        self.position = (None if position is None
                         else np.asarray(position, dtype='float64'))
        self.profile = profile
        self.cam = cam
        self.speed = speed

    @property
    def duration(self):
        return float(self.times[-1])

    def drive_functions(self, mbs):
        """(h(t), b(t)) as engine drive functions."""
        return (mbs.DriveFunction.spline(list(self.times),
                                         list(self.height)),
                mbs.DriveFunction.spline(list(self.times),
                                         list(self.slope)))

    def at(self, t):
        """(height, slope, position) at time t, interpolated."""
        h = float(np.interp(t, self.times, self.height))
        b = float(np.interp(t, self.times, self.slope))
        x = (None if self.position is None
             else float(np.interp(t, self.times, self.position)))
        return h, b, x


def road_input(profile, speed, cam, duration=None, lead=None,
               sample_dt=5e-4):
    """The road input of a wheel driving over ``profile`` at ``speed``.

    The wheel starts ``lead`` (m) before the profile start, so the cams
    begin on flat road, and the run lasts until the cams have left the
    profile (or ``duration`` seconds).

    :param speed: Vehicle speed (m/s), positive.
    :param cam: The :class:`TandemCam` of the tire.
    :return: The :class:`RoadInput`.
    """
    if speed <= 0.0:
        raise ValueError("the speed must be positive")
    lead = cam.reach + 0.05 if lead is None else float(lead)
    x0 = profile.start - lead
    if duration is None:
        duration = (profile.end - x0 + cam.reach + 0.05) / speed
    n = max(int(math.ceil(duration / sample_dt)), 4)
    times = np.linspace(0.0, duration, n + 1)
    position = x0 + speed * times
    w, beta = cam.effective(profile, position)
    name = f"{profile.name} at {speed * 3.6:g} km/h"
    return RoadInput('road', name, times, w, beta, position, profile, cam,
                     speed)


def rig_input(amplitude, frequency, duration, sample_dt=5e-4,
              fade=None):
    """A shaker rig: the road plate moves under the tire; it is not a road
    the car drives on.

    The plate starts at rest: h(t) = A r(t) sin(2 pi f t), with r(t) =
    (1 - cos(pi t / T)) / 2 for t < T and 1 after, T = ``fade`` (default
    one period, at most 1 s).  A sine started at full amplitude would jump
    the plate velocity to A 2 pi f at t = 0.
    """
    omega = 2.0 * math.pi * frequency
    fade = min(1.0 / frequency, 1.0) if fade is None else float(fade)
    n = max(int(math.ceil(duration / sample_dt)), 4)
    times = np.linspace(0.0, duration, n + 1)
    phase = math.pi * np.minimum(times / fade, 1.0)
    ramp = 0.5 * (1.0 - np.cos(phase))
    height = amplitude * ramp * np.sin(omega * times)
    name = (f"rig: plate {amplitude * 1e3:g} mm at {frequency:g} Hz (the "
            f"whole road moves)")
    return RoadInput('rig', name, times, height, np.zeros_like(times))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
