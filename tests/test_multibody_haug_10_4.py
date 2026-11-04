# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Haug Section 10.4: kinematic analysis of an air compressor (BK, FD).

Book data: ``tests/data/multibody/haug_10_4.json`` (Tables 10.4.1 to 10.4.5,
Figs. 10.4.2 to 10.4.4, pp. 408-414). Model:
``solvcon/multibody/models/haug/air_compressor.py``.

Rules under test:

* If the bodies start at the estimates of Table 10.4.4, then ``init()``
  assembles the compressor with zero DOF and a residual below 1e-10.
* Every body origin of the assembled configuration equals Table 10.4.5
  within the precision of the printed table.
* The extreme values of piston 1 position (30 degree offset), stroke (20 and
  40 degrees), velocity, and acceleration equal the values read from
  Figs. 10.4.2 to 10.4.4 within the read uncertainty.
* The piston 1 velocity and acceleration equal finite differences of the
  position and velocity histories (FD).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

DATA_DIR = Path(__file__).parent / "data" / "multibody"
DATA = json.loads((DATA_DIR / "haug_10_4.json").read_text())
OFFSETS = (20.0, 30.0, 40.0)
# One rotor revolution (0.1 s). The sampled extremes differ from the true
# extremes by at most (DT/2)^2/2 |f''|, below 1 percent of each tolerance.
DT = 5e-4
T_END = 0.1


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
def assembled(ac):
    """Assemble the 30 degree model from the Table 10.4.4 estimates.

    :return: (positions, quaternions, dof, max |C|)
    :rtype: tuple
    """
    s, _ = ac.build(offset_deg=30.0)
    s.init()
    pos = [np.array(p) for p in np.asarray(s.get_body_positions())]
    quat = [np.array(q) for q in np.asarray(s.get_body_quaternions())]
    return (pos, quat, s.get_dof(),
            float(np.abs(
                result_arrays(s.evaluate_constraints(0.0))["C"]).max()))


@pytest.fixture(scope="module")
def histories(ac):
    """Piston 1 kinematics for one revolution at each offset angle.

    :return: dict offset -> dict of numpy arrays t, y, ydot, yddot
    :rtype: dict
    """
    out = {}
    for off in OFFSETS:
        h = ac.kinematics(offset_deg=off, t_end=T_END, dt=DT)
        out[off] = {k: np.array(v) for k, v in h.items()}
    return out


TABLE = DATA["results"]["table_10_4_5_assembled"]
ROWS = ["ground_1", "rotor_2", "disk_3"] + [f"piston_{k}" for k in range(1, 7)]

# Table 10.4.5 tolerance. Revolutes A and B fix the rotor origin at 0 and
# the disk origin at (0, -0.5, 0) exactly; the table prints them with errors
# of 9.4e-4 and 2.0e-3 (test_book_table_precision). A piston origin depends
# on the disk origin through one more joint, so its printed error is the
# disk error plus the distance and orientation residuals (about 1e-3 more).
# 5e-3 is about 2.5 times the largest error of an exactly known coordinate.
TOL_POS = 5.0e-3
# Euler parameters carry half the rotation angle; the printed disk e1
# (-0.25844) differs from -sin(15 deg) by 3.8e-4. 1e-3 is 2.5 times that.
TOL_E = 1.0e-3


class TestAssembly:

    def test_zero_dof_and_residual(self, assembled):
        _, _, dof, c0 = assembled
        assert dof == 0
        assert c0 < 1e-10

    def test_book_table_precision(self):
        """The printed table carries errors of about 2e-3 (basis of
        TOL_POS)."""
        rotor = np.array(TABLE["rotor_2"][:3])
        disk = np.array(TABLE["disk_3"][:3])
        err_rotor = np.linalg.norm(rotor)
        err_disk = np.linalg.norm(disk - np.array([0.0, -0.5, 0.0]))
        assert 5e-4 < err_rotor < 1e-3
        assert 1.5e-3 < err_disk < 2.5e-3
        assert 2.5 * max(err_rotor, err_disk) == pytest.approx(TOL_POS,
                                                               rel=0.1)

    @pytest.mark.parametrize("num", range(1, 9))
    def test_table_10_4_5_positions(self, assembled, num):
        pos = assembled[0]
        np.testing.assert_allclose(pos[num], TABLE[ROWS[num]][:3], rtol=0,
                                   atol=TOL_POS)

    @pytest.mark.parametrize("num", range(1, 9))
    def test_table_10_4_5_euler_parameters(self, assembled, num):
        quat = assembled[1]
        e_book = np.array(TABLE[ROWS[num]][3:])
        if num == 2:
            # Printed e0 = 1.96597 is a misprint of 0.96597 (see the JSON
            # note).
            e_book[0] -= 1.0
        e = quat[num] if quat[num][0] >= 0 else -quat[num]
        np.testing.assert_allclose(e, e_book, rtol=0, atol=TOL_E)


def extremes(data, prefix):
    """Return the book extremes whose key starts with ``prefix``.

    :param data: dict label -> [t, y]
    :type data: dict
    :param prefix: key prefix, for example "30_max"
    :type prefix: str
    :return: list of [t, y]
    :rtype: list
    """
    return [v for k, v in data.items() if k.startswith(prefix)]


class TestFigures:

    def test_fig_10_4_2_position_30deg(self, histories):
        f = DATA["results"]["fig_10_4_2"]
        # Tolerance: the read uncertainty of the figure (8 px).
        tol = f["read_uncertainty_px"] * f["units_per_px"][1]
        h = histories[30.0]
        assert h["y"].max() == pytest.approx(
            f["extremes_read_t_y"]["max_t0.1"][1], abs=tol)
        for _, yb in extremes(f["extremes_read_t_y"], "30_min"):
            assert h["y"].min() == pytest.approx(yb, abs=tol)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_fig_10_4_2_stroke(self, histories, offset):
        """Stroke = max - min. For 20 and 40 degrees the book shifts the
        curves so that all three share one maximum (see the JSON note);
        the stroke does not depend on that shift."""
        f = DATA["results"]["fig_10_4_2"]
        ex = f["extremes_read_t_y"]
        # Two readings, each with the read uncertainty: root sum square.
        tol = np.sqrt(2.0) * f["read_uncertainty_px"] * f["units_per_px"][1]
        h = histories[offset]
        for _, ymin in extremes(ex, f"{int(offset)}_min"):
            assert h["y"].max() - h["y"].min() == pytest.approx(
                ex["max_t0.1"][1] - ymin, abs=tol)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_fig_10_4_3_velocity(self, histories, offset):
        f = DATA["results"]["fig_10_4_3"]
        tol = f["read_uncertainty_px"] * f["units_per_px"][1]
        h = histories[offset]
        for _, yb in extremes(f["extremes_read_t_y"], f"{int(offset)}_max"):
            assert h["ydot"].max() == pytest.approx(yb, abs=tol)
        for _, yb in extremes(f["extremes_read_t_y"], f"{int(offset)}_min"):
            assert h["ydot"].min() == pytest.approx(yb, abs=tol)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_fig_10_4_4_acceleration(self, histories, offset):
        f = DATA["results"]["fig_10_4_4"]
        tol = f["read_uncertainty_px"] * f["units_per_px"][1]
        h = histories[offset]
        ex = f["extremes_read_t_y"]
        assert (extremes(ex, f"{int(offset)}_max")
                and extremes(ex, f"{int(offset)}_min"))
        for _, yb in extremes(ex, f"{int(offset)}_max"):
            assert h["yddot"].max() == pytest.approx(yb, abs=tol)
        for _, yb in extremes(ex, f"{int(offset)}_min"):
            assert h["yddot"].min() == pytest.approx(yb, abs=tol)


class TestConsistency:

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_velocity_and_acceleration_fd(self, histories, offset):
        """Central differences of y and ydot match ydot and yddot (FD).

        The central difference error is dt^2/6 times the third derivative
        of the differenced signal. The bound uses the largest third
        derivative estimated from the yddot history (jerk for the velocity,
        its derivative for the acceleration) with a factor 1.5 for the
        estimate.
        """
        h = histories[offset]
        jerk = np.gradient(h["yddot"], DT)
        snap = np.gradient(jerk, DT)
        tol_v = 1.5 * DT**2 / 6.0 * np.abs(jerk).max()
        tol_a = 1.5 * DT**2 / 6.0 * np.abs(snap).max()
        v_fd = (h["y"][2:] - h["y"][:-2]) / (2 * DT)
        a_fd = (h["ydot"][2:] - h["ydot"][:-2]) / (2 * DT)
        np.testing.assert_allclose(v_fd, h["ydot"][1:-1], rtol=0, atol=tol_v)
        np.testing.assert_allclose(a_fd, h["yddot"][1:-1], rtol=0, atol=tol_a)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
