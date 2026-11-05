# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Ride comfort and road holding on random roads.

Road roughness (ISO 8608:1995).  A road class is the displacement PSD
coefficient G_d(n0) at the reference spatial frequency n0 = 0.1 cycles/m;
the one-sided PSD is G_d(n) = G_d(n0) (n / n0)^-w with w = 2.  The class
values are the geometric means of the class ranges (A 16, B 64, C 256, D
1024, E 4096, F 16384, G 65536, H 262144, in 1e-6 m^3).  The standard also
states them in angular spatial frequency at Omega0 = 1 rad/m, G_d(Omega) =
G_d(n) / (2 pi): A 1, B 4, C 16, ... (some papers quote these as G_d(n0)).
A profile is a sum of cosines with random phases, z(x) = sum_i a_i cos(2 pi
n_i x + phi_i), a_i = sqrt(2 G_d(n_i) dn), whose variance is the integral of
G_d.  Some papers use a_i = sqrt(G_d(n_i) dn) (half the variance); the
``convention='half'`` option reproduces them.

Human response (ISO 2631-1:1997, Annex A).  The frequency weighting is
H = H_h H_l H_t H_s: the band limits (second-order Butterworth high pass at
f1 and low pass at f2), the acceleration-velocity transition H_t = (1 +
s/w3) / (1 + s/(Q4 w4) + (s/w4)^2), and the upward step H_s = (s^2 +
s w5/Q5 + w5^2) / (s^2 + s w6/Q6 + w6^2) (the form of the erratum of Rimell
and Mansfield, Ind. Health 48, 2010).  Wk (vertical): f1 0.4, f2 100, f3 =
f4 12.5 Hz, Q4 0.63, f5 2.37 Hz, Q5 0.91, f6 3.35 Hz, Q6 0.91.  Wd
(horizontal): f1 0.4, f2 100, f3 = f4 2 Hz, Q4 0.63, no upward step.  The
weighted RMS acceleration a_w reads on the comfort scale of Annex C (the
ranges overlap): < 0.315 m/s^2 not uncomfortable, 0.315-0.63 a little
uncomfortable, 0.5-1 fairly uncomfortable, 0.8-1.6 uncomfortable, 1.25-2.5
very uncomfortable, > 2 extremely uncomfortable.

Road holding is the standard deviation of the tire load over the static
load, R = sigma_F / F_static; R below about 0.3 keeps the tire on the road
(a Gaussian load leaves zero load 3.3 standard deviations away).
"""

import math

import numpy as np

from . import kc
from . import road as road_mod
from . import vehicle

__all__ = [  # noqa: F822
    'COMFORT_SCALE',
    'ISO_8608_CLASSES',
    'LinearCorner',
    'TwoDofQuarterCar',
    'WEIGHTINGS',
    'comfort_reaction',
    'point_road_input',
    'random_profile',
    'road_psd',
    'road_psd_angular',
    'simulate',
    'weighted_rms',
    'weighting',
]

#: ISO 8608 road classes: G_d(n0) (m^3), the geometric mean of the class.
ISO_8608_CLASSES = {
    'A': 16e-6, 'B': 64e-6, 'C': 256e-6, 'D': 1024e-6, 'E': 4096e-6,
    'F': 16384e-6, 'G': 65536e-6, 'H': 262144e-6,
}
#: The reference spatial frequency (cycles/m) and the waviness.
N0 = 0.1
WAVINESS = 2.0
#: The spatial frequency band of ISO 8608 (cycles/m).
N_BAND = (0.011, 2.83)

#: ISO 2631-1 Annex A weighting parameters (Hz and quality factors).
WEIGHTINGS = {
    'k': {'f1': 0.4, 'f2': 100.0, 'f3': 12.5, 'f4': 12.5, 'q4': 0.63,
          'f5': 2.37, 'q5': 0.91, 'f6': 3.35, 'q6': 0.91},
    'd': {'f1': 0.4, 'f2': 100.0, 'f3': 2.0, 'f4': 2.0, 'q4': 0.63,
          'f5': None, 'q5': None, 'f6': None, 'q6': None},
}

#: ISO 2631-1 Annex C comfort reactions: (low, high, label) in m/s^2.
COMFORT_SCALE = (
    (0.0, 0.315, "not uncomfortable"),
    (0.315, 0.63, "a little uncomfortable"),
    (0.5, 1.0, "fairly uncomfortable"),
    (0.8, 1.6, "uncomfortable"),
    (1.25, 2.5, "very uncomfortable"),
    (2.0, math.inf, "extremely uncomfortable"),
)


def _class_value(road_class):
    if isinstance(road_class, str):
        try:
            return ISO_8608_CLASSES[road_class.upper()]
        except KeyError:
            raise ValueError(f"unknown ISO 8608 road class {road_class!r} "
                             "(A to H)") from None
    value = float(road_class)
    if value <= 0.0:
        raise ValueError("G_d(n0) must be positive")
    return value


def road_psd(n, road_class='C'):
    """The one-sided displacement PSD G_d(n) (m^3) of an ISO 8608 road.

    :param n: spatial frequency (cycles/m), positive
    :param road_class: a class letter, or G_d(n0) in m^3
    """
    n = np.asarray(n, dtype='float64')
    return _class_value(road_class) * (n / N0) ** -WAVINESS


def road_psd_angular(omega, road_class='C'):
    """G_d(Omega) (m^3) in angular spatial frequency (rad/m):
    G_d(n) / (2 pi) at n = Omega / (2 pi)."""
    omega = np.asarray(omega, dtype='float64')
    return road_psd(omega / (2.0 * math.pi), road_class) / (2.0 * math.pi)


def random_profile(road_class, length, seed=0, band=N_BAND, count=2000,
                   dx=0.01, convention='iso'):
    """A random road profile of an ISO 8608 class (harmonic superposition).

    :param road_class: a class letter, or G_d(n0) in m^3
    :param length: profile length (m)
    :param seed: random phase seed
    :param band: (n_min, n_max) spatial frequency band (cycles/m)
    :param count: number of harmonics, at the band midpoints
    :param dx: sample spacing (m)
    :param convention: ``iso`` (a_i = sqrt(2 G_d dn), the variance of the
        PSD) or ``half`` (a_i = sqrt(G_d dn), the convention of some papers)
    :return: the profile, starting at x = 0
    :rtype: suspension.road.RoadProfile
    """
    if length <= 0.0 or dx <= 0.0:
        raise ValueError("the length and the spacing must be positive")
    if convention not in ('iso', 'half'):
        raise ValueError("convention is 'iso' or 'half'")
    n_lo, n_hi = float(band[0]), float(band[1])
    if not 0.0 < n_lo < n_hi:
        raise ValueError("the band needs 0 < n_min < n_max")
    dn = (n_hi - n_lo) / count
    n = n_lo + dn * (np.arange(count) + 0.5)
    factor = 2.0 if 'iso' == convention else 1.0
    amp = np.sqrt(factor * road_psd(n, road_class) * dn)
    phase = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, count)
    x = np.arange(0.0, length + 0.5 * dx, dx)
    z = np.zeros_like(x)
    # Chunks keep the cosine table small.
    for lo in range(0, count, 64):
        hi = min(lo + 64, count)
        z += amp[lo:hi] @ np.cos(2.0 * math.pi * np.outer(n[lo:hi], x)
                                 + phase[lo:hi, None])
    name = (f"ISO 8608 class {road_class}" if isinstance(road_class, str)
            else f"G_d(n0) {road_class:g} m^3")
    return road_mod.RoadProfile(f"{name}, {length:g} m", x, z)


def weighting(f, kind='k'):
    """The complex ISO 2631-1 frequency weighting at frequencies ``f``
    (Hz)."""
    try:
        p = WEIGHTINGS[kind]
    except KeyError:
        raise ValueError(f"unknown weighting {kind!r} ('k' or 'd')") \
            from None
    s = 2j * math.pi * np.asarray(f, dtype='float64')
    w1, w2, w3, w4 = (2.0 * math.pi * p[key] for key in
                      ('f1', 'f2', 'f3', 'f4'))
    q = 1.0 / math.sqrt(2.0)
    out = s ** 2 / (s ** 2 + w1 * s / q + w1 ** 2)
    out = out * w2 ** 2 / (s ** 2 + w2 * s / q + w2 ** 2)
    out = out * (1.0 + s / w3) / (1.0 + s / (p['q4'] * w4) + (s / w4) ** 2)
    if p['f5'] is not None:
        w5, w6 = 2.0 * math.pi * p['f5'], 2.0 * math.pi * p['f6']
        out = out * ((s ** 2 + w5 * s / p['q5'] + w5 ** 2)
                     / (s ** 2 + w6 * s / p['q6'] + w6 ** 2))
    return out


def weighted_rms(times, signal, kind='k', settle=0.0):
    """The RMS of a uniformly sampled acceleration after the ISO 2631-1
    weighting (applied in the frequency domain), and the unweighted RMS.

    :param settle: time (s) dropped at the start (the weighting is applied
        to the whole record first)
    :return: (weighted RMS, unweighted RMS) of the signal after ``settle``
    """
    t = np.asarray(times, dtype='float64')
    a = np.asarray(signal, dtype='float64')
    if t.size < 8 or t.size != a.size:
        raise ValueError("weighted_rms needs matching samples, 8 or more")
    dt = (t[-1] - t[0]) / (t.size - 1)
    if np.max(np.abs(np.diff(t) - dt)) > 1e-6 * dt:
        raise ValueError("weighted_rms needs uniform samples")
    keep = t >= t[0] + settle
    a = a - np.mean(a[keep])
    spec = np.fft.rfft(a)
    freq = np.fft.rfftfreq(a.size, dt)
    filtered = np.fft.irfft(spec * weighting(freq, kind), a.size)
    return (float(np.sqrt(np.mean(filtered[keep] ** 2))),
            float(np.sqrt(np.mean(a[keep] ** 2))))


def comfort_reaction(a_w):
    """The ISO 2631-1 comfort reactions whose range holds ``a_w`` (m/s^2),
    joined with ' to ' where the ranges overlap."""
    labels = [label for lo, hi, label in COMFORT_SCALE if lo <= a_w < hi]
    return " to ".join(labels)


def _frequency_grid(speed, band, count):
    lo, hi = band[0] * speed, band[1] * speed
    return np.geomspace(lo, hi, count)


class _LinearRide(object):
    """A linear vertical model M q'' + C q' + K q = b (k_t y + c_t y') on
    the road height y; subclasses define the outputs."""

    def __init__(self, mm, cc, kk, b, tire_k, tire_c, static_load):
        self.M = np.asarray(mm, dtype='float64')
        self.C = np.asarray(cc, dtype='float64')
        self.K = np.asarray(kk, dtype='float64')
        self.b = np.asarray(b, dtype='float64')
        self.tire_k = float(tire_k)
        self.tire_c = float(tire_c)
        self.static_load = float(static_load)

    def response(self, f):
        """Frequency responses per unit road height at ``f`` (Hz): dict of
        complex arrays ``body_acc`` (m/s^2 per m), ``tire_force`` (N per
        m), ``travel`` (m per m, + bump)."""
        w = 2.0 * math.pi * np.asarray(f, dtype='float64')
        out = {'body_acc': [], 'tire_force': [], 'travel': []}
        for wi in w:
            dyn = self.K - wi * wi * self.M + 1j * wi * self.C
            road = self.tire_k + 1j * wi * self.tire_c
            q = np.linalg.solve(dyn, self.b * road)
            out['body_acc'].append(-wi * wi * q[0])
            out['tire_force'].append(road * (1.0 - self.b @ q))
            out['travel'].append(q[1])
        return {key: np.array(val) for key, val in out.items()}

    def random_response(self, road_class, speed, band=N_BAND, count=4000,
                        convention='iso'):
        """Standard deviations on a random road at ``speed`` (m/s).

        The temporal road PSD is G(f) = G_d(f / v) / v over the spatial
        band; ``convention='half'`` halves it (see :func:`random_profile`).

        :return: dict with ``body_acc`` and ``body_acc_w`` (Wk weighted)
            in m/s^2, ``tire_force`` (N), ``tire_load_ratio`` (sigma_F over
            the static load), and ``travel`` (m)
        """
        if speed <= 0.0:
            raise ValueError("the speed must be positive")
        f = _frequency_grid(speed, band, count)
        g = road_psd(f / speed, road_class) / speed
        if 'half' == convention:
            g = 0.5 * g
        h = self.response(f)

        def sigma(x):
            return float(math.sqrt(np.trapezoid(np.abs(x) ** 2 * g, f)))

        out = {'body_acc': sigma(h['body_acc']),
               'body_acc_w': sigma(h['body_acc'] * weighting(f, 'k')),
               'tire_force': sigma(h['tire_force']),
               'travel': sigma(h['travel'])}
        out['tire_load_ratio'] = out['tire_force'] / self.static_load
        out['comfort'] = comfort_reaction(out['body_acc_w'])
        return out


class TwoDofQuarterCar(_LinearRide):
    """The classic 2-DOF quarter car: sprung mass on the suspension
    spring-damper, unsprung mass on the tire spring-damper; coordinates
    (z_s, z_u - z_s) so the second one is the travel (+ bump)."""

    def __init__(self, ms, mu, ks, cs, kt, ct=0.0, gravity=9.81):
        # z_u = z_s + z: kinetic energy ms zs'^2 + mu (zs' + z')^2.
        mm = [[ms + mu, mu], [mu, mu]]
        cc = [[ct, ct], [ct, cs + ct]]
        kk = [[kt, kt], [kt, ks + kt]]
        super().__init__(mm, cc, kk, [1.0, 1.0], kt, ct,
                         (ms + mu) * gravity)
        self.ms, self.mu, self.ks, self.cs = ms, mu, ks, cs


def point_road_input(profile, speed, duration, sample_dt=5e-4):
    """The road input of a point follower: the tire sees the profile height
    under the wheel, with no enveloping (the input the linear models
    assume).

    :rtype: suspension.road.RoadInput
    """
    if speed <= 0.0 or duration <= 0.0:
        raise ValueError("the speed and the duration must be positive")
    n = max(int(math.ceil(duration / sample_dt)), 4)
    times = np.linspace(0.0, duration, n + 1)
    position = profile.start + speed * times
    height = profile.height(position)
    return road_mod.RoadInput('road', f"{profile.name} at "
                              f"{speed * 3.6:g} km/h (point follower)",
                              times, height, np.zeros_like(times), position,
                              profile, None, speed)


def simulate(mbs, model, sprung_mass, road_input, dt=1e-3, settle=1.0,
             gravity=9.81):
    """March the quarter car of a corner model over a road input and
    measure its ride.

    :param settle: time (s) dropped at the start of the statistics
    :return: dict with the sampled ``time`` (s), ``body_acc`` (m/s^2),
        ``tire_force`` (N), ``travel`` (m, + bump), ``road`` (m), and the
        statistics after ``settle``: ``body_acc_rms``, ``body_acc_w``
        (Wk), ``tire_force_std``, ``tire_load_ratio``, ``travel_std``,
        ``airborne`` (s), and ``comfort``
    """
    s, h = vehicle.quarter_car(mbs, model, sprung_mass, road=road_input,
                               dt=dt, gravity=gravity)
    s.set_integrator('rk45')
    s.set_tolerances(1e-7, 1e-10)
    nums = dict(h['corner'].nums)
    chassis = vehicle.CHASSIS
    z_design = float(model.hp['wheel_center'][2])
    steps = int(math.floor(road_input.duration / dt)) - 1
    rec = {'time': [], 'body_acc': [], 'tire_force': [], 'travel': [],
           'road': []}
    for _ in range(steps):
        s.solve()
        t = s.get_sim_time()
        pos = np.asarray(s.get_body_positions())
        acc = np.asarray(s.get_body_accelerations())
        rec['time'].append(t)
        rec['body_acc'].append(float(acc[chassis][2]))
        rec['tire_force'].append(float(
            mbs.get_tire_state(h['tire'])['normal_force']))
        rec['travel'].append(float(pos[nums['upright']][2]
                                   - pos[chassis][2] - z_design))
        rec['road'].append(road_input.at(t)[0])
    out = {key: np.array(val) for key, val in rec.items()}
    keep = out['time'] >= settle
    a_w, a_rms = weighted_rms(out['time'], out['body_acc'], 'k', settle)
    force = out['tire_force'][keep]
    out.update({
        'body_acc_rms': a_rms, 'body_acc_w': a_w,
        'tire_force_std': float(np.std(force)),
        'tire_load_ratio': float(np.std(force) / h['weight']),
        'travel_std': float(np.std(out['travel'][keep])),
        'airborne': float(dt * np.count_nonzero(force <= 0.0)),
        'comfort': comfort_reaction(a_w),
        'static_load': float(h['weight'])})
    return out


class LinearCorner(_LinearRide):
    """The linear quarter car of a corner model: the exact K&C
    linearization of :func:`vehicle.linear_quarter_car` in (chassis z,
    travel), with the suspension damping c MR^2 on the travel (the damper
    acts on the spring line) and the tire damping on the contact point, at
    the static travel of the model (its ride height).

    :ivar motion_ratio: spring compression per wheel travel at design
    :ivar wheel_rate: wheel rate at design (N/m)
    """

    def __init__(self, mbs, model, sprung_mass, gravity=9.81):
        # The wheel rate under the static spring preload: the spring force
        # times the change of the motion ratio adds to k MR^2.  The quarter
        # car finds the free length that holds the design pose.
        _, handles = vehicle.quarter_car(mbs, model, sprung_mass,
                                         gravity=gravity)
        s0 = float(model.static_travel)
        sweep = kc.bump_sweep(mbs, handles['model'],
                              [s0 - 1e-3, s0, s0 + 1e-3])
        if not sweep['completed']:
            raise RuntimeError(f"ride: the K&C sweep failed: "
                               f"{sweep['message']}")
        self.motion_ratio = float(sweep['motion_ratio'][1])
        self.wheel_rate = float(sweep['wheel_rate'][1])
        _, weight = vehicle.static_tire_deflection(model, sprung_mass,
                                                   gravity)
        lin = vehicle.linear_quarter_car(mbs, model, sprung_mass,
                                         self.wheel_rate, model.tire_k,
                                         tire_load=weight, travel=s0)
        a = lin['a']
        ct = model.tire_c
        c_w = model.spring_c * self.motion_ratio ** 2
        cc = [[ct, ct * a], [ct * a, c_w + ct * a * a]]
        super().__init__(lin['M'], cc, lin['K'], [1.0, a], model.tire_k,
                         ct, weight)
        self.wheel_damping = c_w
        self.patch_rate = a

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
