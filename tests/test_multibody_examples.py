# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Regression tests for the example mechanisms in
solvcon.multibody.models.gallery.

Rules under test:

* Every example assembles, runs 300 steps, and keeps |C| below 1e-8.
* The scotch yoke follows its analytic motion y = R sin(omega t).
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

GALLERY = Path(importlib.util.find_spec(
    "solvcon.multibody.models.gallery").origin)

CLOSED_LOOP = {"build_slider_crank", "build_crank_rocker", "build_scotch_yoke",
               "build_peaucellier", "build_klann_leg"}


@pytest.fixture(scope="module")
def ga():
    """Import the example module or skip if the solver is not built.

    :return: solvcon.multibody.models.gallery module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    from solvcon.multibody.models import gallery
    return gallery


def builder_names():
    """Return the example builder names without importing the solver.

    :return: sorted builder function names
    :rtype: list
    """
    text = GALLERY.read_text()
    return sorted(line.split("(")[0][4:] for line in text.splitlines()
                  if line.startswith("def build_"))


@pytest.mark.parametrize("name", builder_names())
def test_example_runs_and_keeps_constraints(ga, name):
    out = getattr(ga, name)()
    s = out[0] if isinstance(out, tuple) else out
    if name not in CLOSED_LOOP:
        s.Assembly()
    s.init()
    for _ in range(300):
        s.solve()
    c = result_arrays(s.evaluate_constraints(s.get_sim_time()))["C"]
    assert np.max(np.abs(c)) < 1e-8
    assert np.all(np.isfinite(np.asarray(s.get_body_positions())))


def test_scotch_yoke_follows_analytic_motion(ga):
    s, dt = ga.build_scotch_yoke()
    s.init()
    for k in range(1, 1001):
        s.solve()
        y = np.asarray(s.get_body_positions())[2][1]
        assert y == pytest.approx(1.5 * math.sin(2 * math.pi * k * dt),
                                  abs=1e-12)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
