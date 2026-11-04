# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The K&C plot: the previous curve is the dashed one, and the cursor reads
the curves where the 3D pose is."""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.suspension import _plots
except ImportError:
    pilot = None


@testing.requires_pilot
class KCPlotTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def _plot(self):
        plot = _plots.KCPlot("Camber", "Wheel travel [mm]", "Camber [deg]")
        self.addCleanup(plot.deleteLater)
        plot.set_curves([
            ("previous", [-10.0, 0.0, 10.0], [1.0, 0.0, -1.0],
             _plots.PREVIOUS_COLOR, True),
            ("current", [-10.0, 0.0, 10.0], [0.5, -0.5, -1.5],
             _plots.CURRENT_COLOR, False)])
        return plot

    def test_only_the_previous_curve_is_dashed(self):
        plot = self._plot()
        self.assertEqual(["previous", "current"], plot.labels())
        self.assertTrue(plot.dashed("previous"))
        self.assertFalse(plot.dashed("current"))
        self.assertEqual(([-10.0, 0.0, 10.0], [0.5, -0.5, -1.5]),
                         plot.data("current"))

    def test_new_curves_replace_the_old(self):
        # A sweep over another variable must not leave the curves of the
        # last one behind.
        plot = self._plot()
        plot.set_curves([("current", [0.0, 1.0], [2.0, 3.0],
                          _plots.CURRENT_COLOR, False)])
        self.assertEqual(["current"], plot.labels())
        self.assertFalse(plot.dashed("previous"))

    def test_cursor_reads_each_curve(self):
        plot = self._plot()
        self.assertIsNone(plot.marker("current"))
        plot.set_cursor(5.0)
        self.assertAlmostEqual(-0.5, plot.marker("previous"))
        self.assertAlmostEqual(-1.0, plot.marker("current"))
        plot.set_cursor(50.0)
        self.assertIsNone(plot.marker("current"))
        # Painting with the cursor outside the curves draws no marker and
        # does not fail.
        plot.resize(320, 200)
        plot.grab()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
