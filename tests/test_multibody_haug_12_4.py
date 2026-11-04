# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Haug Section 12.4: inverse dynamics of the air compressor (BK, AN, IV).

Book data: ``tests/data/multibody/haug_12_4.json`` (Table 12.4.1, Fig. 12.4.1,
and Figs. 12.4.2 to 12.4.5, pp. 462-468). Model:
``solvcon/multibody/models/haug/air_compressor.py`` with the geometry of
Section 10.4.

Rules under test:

* The air force of Fig. 12.4.1 is continuous, zero at A and B, and its
  stroke work equals the closed-form integral (AN).
* If the rotor turns at 20 pi rad/s, then the mean actuator torque equals
  6 W / (2 pi), the air work per revolution divided by 2 pi (IV; this is
  also the constant torque definition of Sec. 12.4.3).
* The driving torque and the x position constraint force agree with
  Figs. 12.4.2 and 12.4.3 (30 degrees; 0, 250, 500 psi) and with the 0 psi
  curves of Figs. 12.4.4 and 12.4.5 (40 degrees) within the read
  uncertainty of each figure.

Sign convention. The book plots the driving torque as the constraint
reaction on the rotor, which is minus the actuator torque (solver driver
``effort``). The x constraint force is the global x reaction on the disk.

Not compared (see ``not_compared`` and ``pressure_curves`` in the JSON):
the 40 degree pressure curves and the forward dynamics of Sec. 12.4.3.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody

DATA_DIR = Path(__file__).parent / "data" / "multibody"
DATA = json.loads((DATA_DIR / "haug_12_4.json").read_text())
INV = DATA["inverse_dynamics_12_4_2"]
PRESSURES = (0.0, 250.0, 500.0)


@pytest.fixture(scope="module")
def ac():
    """Import the compressor builder, or skip if the solver is not built.

    :return: builder module ``air_compressor``
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    from solvcon.multibody.models.haug import air_compressor
    return air_compressor


@pytest.fixture(scope="module")
def inv30(ac):
    """Inverse dynamics over one revolution, 30 degree offset.

    The output step 2.5e-4 s is 3.5 px of the figures' time axis.

    :return: dict pressure -> dict of numpy arrays
    :rtype: dict
    """
    res = ac.inverse_dynamics(list(np.linspace(0.0, 0.1, 401)), 30.0,
                              PRESSURES)
    return {p: {k: np.array(v) for k, v in r.items()} for p, r in res.items()}


@pytest.fixture(scope="module")
def inv40(ac):
    """Inverse dynamics over one revolution, 40 degree offset, 0 psi.

    :return: dict of numpy arrays
    :rtype: dict
    """
    res = ac.inverse_dynamics(list(np.linspace(0.0, 0.1, 201)), 40.0, (0.0,))
    return {k: np.array(v) for k, v in res[0.0].items()}


def pixel_distance(points, t, y, upp):
    """Distance in pixels from each book point to a simulated curve.

    :param points: book points [[t, y], ...]
    :type points: list
    :param t: simulated times
    :type t: numpy.ndarray
    :param y: simulated values
    :type y: numpy.ndarray
    :param upp: data units per pixel [t, y]
    :type upp: list
    :return: distance of each point [px]
    :rtype: numpy.ndarray
    """
    tt = np.linspace(t[0], t[-1], 20 * len(t))
    yy = np.interp(tt, t, y)
    return np.array([np.min(np.hypot((tt - a) / upp[0], (yy - b) / upp[1]))
                     for a, b in points])


class TestAirForce:

    def test_curve_shape(self, ac):
        """F(A) = 0, F(B) = 0, F continuous at C, no force on the return."""
        p = 250.0
        f_c = ac.air_force(ac.Y_C, -1.0, p)
        assert ac.air_force(ac.Y_A + 1e-12, -1.0, p) == pytest.approx(0.0,
                                                                      abs=1e-6)
        assert ac.air_force(ac.Y_B - 1e-12, -1.0, p) == pytest.approx(0.0,
                                                                      abs=1e-6)
        # 1e-9 m from C the slope F_max / u_C changes F by 3e-8 F_C.
        assert ac.air_force(ac.Y_C - 1e-9, -1.0, p) == pytest.approx(f_c,
                                                                     rel=1e-7)
        assert ac.air_force(ac.Y_C + 1e-9, -1.0, p) == pytest.approx(f_c,
                                                                     rel=1e-7)
        assert ac.air_force(-1.0, +1.0, p) == 0.0

    @pytest.mark.parametrize("p", [250.0, 500.0])
    def test_stroke_work_closed_form(self, ac, p):
        """W = F_max u_C ln(u_B/u_C) - F_min (u_B - u_C) + F_C u_C / 2."""
        f_max = p * ac.PSI * math.pi * ac.BORE**2 / 4.0
        u_c, u_b = ac.Y_C - ac.Y_A, ac.Y_B - ac.Y_A
        f_min = f_max * u_c / u_b
        w = (f_max * u_c * math.log(u_b / u_c) - f_min * (u_b - u_c)
             + (f_max - f_min) * u_c / 2)
        # Midpoint rule with 20000 intervals: relative error below 1e-8.
        assert ac.cycle_work(p) == pytest.approx(w, rel=1e-8)


class TestInverseDynamics30:

    @pytest.mark.parametrize("p", [250.0, 500.0])
    def test_mean_torque_energy_balance(self, ac, inv30, p):
        """Mean actuator torque = 6 W / (2 pi) at constant speed (IV)."""
        torque = inv30[p]["torque"][:-1]
        # Uniform samples of a continuous periodic signal; the force is zero
        # where it switches (A and B). 1e-3 covers the sampling error.
        assert torque.mean() == pytest.approx(
            6.0 * ac.cycle_work(p) / (2.0 * math.pi), rel=1e-3)

    def test_zero_psi_mean_torque_is_zero(self, inv30):
        """Without air force no energy leaves the system (IV)."""
        torque = inv30[0.0]["torque"]
        assert abs(torque[:-1].mean()) < 1e-6 * np.abs(torque).max()

    def test_fig_12_4_2_zero_psi(self, inv30):
        f = INV["fig_12_4_2_torque_30deg"]
        r = inv30[0.0]
        d = pixel_distance(f["zero_psi"]["points_t_y"], r["t"], -r["torque"],
                           f["units_per_px"])
        assert d.max() <= f["zero_psi"]["read_uncertainty_px"], d.max()

    def test_fig_12_4_3_zero_psi(self, inv30):
        f = INV["fig_12_4_3_xforce_30deg"]
        r = inv30[0.0]
        d = pixel_distance(f["zero_psi"]["points_t_y"], r["t"],
                           r["slot_force"], f["units_per_px"])
        assert d.max() <= f["zero_psi"]["read_uncertainty_px"], d.max()

    @pytest.mark.parametrize("p", [250.0, 500.0])
    def test_fig_12_4_2_extremes(self, inv30, p):
        """Every peak and dip of the pressure curves (6 + 6 per curve)."""
        f = INV["fig_12_4_2_torque_30deg"]
        pts = [v for k, v in f["extremes_read_t_y"].items()
               if k.startswith(f"{int(p)}_")]
        assert len(pts) == 12
        r = inv30[p]
        d = pixel_distance(pts, r["t"], -r["torque"], f["units_per_px"])
        assert d.max() <= f["extremes_read_uncertainty_px"], d.max()

    @pytest.mark.parametrize("p", [250.0, 500.0])
    def test_fig_12_4_3_extremes(self, inv30, p):
        f = INV["fig_12_4_3_xforce_30deg"]
        pts = [v for k, v in f["extremes_read_t_y"].items()
               if k.startswith(f"{int(p)}_")]
        assert len(pts) == 12
        r = inv30[p]
        d = pixel_distance(pts, r["t"], r["slot_force"], f["units_per_px"])
        assert d.max() <= f["extremes_read_uncertainty_px"], d.max()


class TestInverseDynamics40:

    def test_fig_12_4_4_zero_psi(self, inv40):
        f = INV["fig_12_4_4_torque_40deg"]
        d = pixel_distance(f["zero_psi"]["points_t_y"], inv40["t"],
                           -inv40["torque"], f["units_per_px"])
        assert d.max() <= f["zero_psi"]["read_uncertainty_px"], d.max()

    def test_fig_12_4_5_zero_psi(self, inv40):
        f = INV["fig_12_4_5_xforce_40deg"]
        d = pixel_distance(f["zero_psi"]["points_t_y"], inv40["t"],
                           inv40["slot_force"], f["units_per_px"])
        assert d.max() <= f["zero_psi"]["read_uncertainty_px"], d.max()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
