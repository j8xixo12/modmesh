# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for compliance K&C (roadmap P9.8).

Rules under test:

* With very stiff bushings, the compliant rig reproduces the rigid K&C
  alignment.
* The linear contact patch compliance (central differences) does not depend
  on the load level.
* Maxwell-Betti reciprocity: the linear compliance matrix is symmetric.
"""

import copy

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import compliance, kc
from solvcon.multibody.suspension.model import CornerModel


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


@pytest.fixture
def model():
    m = CornerModel.load("race_front_corner")
    # non-zero static toe: a non-trivial alignment
    m.set_adjusters(toe_link=0.001)
    return m


def test_stiff_bushings_reproduce_rigid_alignment(mbs, model):
    rigid = kc.bump_sweep(mbs, model, [0.0])
    s, _ = compliance.compliant_rig(mbs, model, k_trans=1e12, k_rot=1e6)
    out = compliance.apply_patch_load(mbs, s, model, [0.0, 0.0, 0.0])
    # The rigid K&C rig solves positions to 1e-10 m; over the 90 mm tie-rod
    # lever that is ~1e-9 rad, so 2e-8 rad bounds both solves.
    assert out["toe"] == pytest.approx(rigid["toe"][0], abs=2e-8)
    assert out["camber"] == pytest.approx(rigid["camber"][0], abs=2e-8)


def compliance_matrix(mbs, model, force):
    """Contact patch compliance C[i, j] = d u_i / d F_j by central differences
    (+/- F).

    Central differences remove the second-order (geometric) terms that a
    one-sided load would add to the linear compliance.

    :return: 3x3 matrix (m/N) and the toe compliance row (rad/N)
    :rtype: tuple
    """
    s, _ = compliance.compliant_rig(mbs, model)
    base = compliance.apply_patch_load(mbs, s, model, [0.0, 0.0, 0.0])
    cp_local = base["a"].T @ (base["contact_patch"] - base["r"])
    c = np.zeros((3, 3))
    toe = np.zeros(3)
    for j in range(3):
        pts, toes = [], []
        for sign in (1.0, -1.0):
            f = [0.0, 0.0, 0.0]
            f[j] = sign * force
            s2, _ = compliance.compliant_rig(mbs, model)
            out = compliance.apply_patch_load(mbs, s2, model, f)
            pts.append(out["r"] + out["a"] @ cp_local)
            toes.append(out["toe"])
        c[:, j] = (pts[0] - pts[1]) / (2 * force)
        toe[j] = (toes[0] - toes[1]) / (2 * force)
    return c, toe


def test_linearity_and_reciprocity(mbs, model):
    c1, toe1 = compliance_matrix(mbs, model, 200.0)
    c2, toe2 = compliance_matrix(mbs, model, 400.0)
    assert abs(toe1[1]) > 1e-9  # the bushings give measurable compliance steer
    # Maxwell-Betti: the linear compliance of a conservative system is
    # symmetric. Measured asymmetry is ~7e-9 of max |C| (Newton tolerance).
    scale = np.max(np.abs(c1))
    np.testing.assert_allclose(c1, c1.T, atol=1e-6 * scale)
    # Linearity: the central-difference compliance does not depend on the load
    # level to first order; the remaining (cubic) term is below 1e-3 here.
    np.testing.assert_allclose(c2, c1, atol=1e-3 * scale)
    assert toe2[1] == pytest.approx(toe1[1], rel=1e-3)


def test_softer_bushings_give_more_compliance_steer(mbs, model):
    toes = []
    for k in (4e7, 1e7):
        m = copy.deepcopy(model)
        s, _ = compliance.compliant_rig(mbs, m, k_trans=k)
        base = compliance.apply_patch_load(mbs, s, m, [0.0, 0.0, 0.0])
        s2, _ = compliance.compliant_rig(mbs, m, k_trans=k)
        loaded = compliance.apply_patch_load(mbs, s2, m, [0.0, 500.0, 0.0])
        toes.append(loaded["toe"] - base["toe"])
    # Bushings in series with rigid links: compliance scales with 1 / k.
    assert toes[1] == pytest.approx(4 * toes[0], rel=2e-2)
    assert np.sign(toes[0]) == np.sign(toes[1])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
