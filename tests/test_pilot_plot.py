# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the native xy-plot core: PlotColor and the matplotlib C0-C9 cycle.

The core is pure C++ with no Qt widget, so everything here is exercised
through the pybind11 surface registered into ``solvcon.pilot``.
"""

import unittest

import solvcon

try:
    from solvcon import pilot
except ImportError:
    pilot = None


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PlotColorTC(unittest.TestCase):
    """The color vocabulary a plot is styled with."""

    def test_cycle_is_matplotlib_c0_to_c9(self):
        """The cycle must be matplotlib's C0-C9, not an invented palette.

        Users read matplotlib plots every day; reusing the same ten colors
        in the same order means the pilot's plot of the same data carries
        the same meaning without relearning anything.
        """
        cycle = pilot.plot_color_cycle()
        self.assertEqual(10, len(cycle))
        self.assertEqual(31, cycle[0].r)
        self.assertEqual(119, cycle[0].g)
        self.assertEqual(180, cycle[0].b)
        self.assertEqual(255, cycle[0].a)

    def test_cycle_wraps_instead_of_running_out(self):
        """An eleventh request must get C0 again, not an error or black.

        The cycle is a modulo, so a plot with any number of series always
        has a color for every one of them.
        """
        self.assertEqual(pilot.plot_cycle_color(0),
                         pilot.plot_cycle_color(10))
        self.assertEqual(pilot.plot_cycle_color(3),
                         pilot.plot_cycle_color(23))

    def test_color_is_an_immutable_value(self):
        """Alpha defaults to opaque and a channel assignment must fail.

        Every function that yields a color yields a copy, so
        ``plot_cycle_color(0).a = 128`` would paint nothing different while
        looking like it had. Refusing the assignment turns that into an
        error the caller can see.
        """
        color = pilot.PlotColor(1, 2, 3)
        self.assertEqual((1, 2, 3, 255),
                         (color.r, color.g, color.b, color.a))
        self.assertEqual(pilot.PlotColor(1, 2, 3, 255), color)
        self.assertNotEqual(pilot.PlotColor(1, 2, 3, 128), color)
        for channel in ('r', 'a'):
            with self.subTest(channel=channel):
                with self.assertRaises(AttributeError):
                    setattr(color, channel, 9)
        self.assertEqual(pilot.PlotColor(1, 2, 3), color)

    def test_color_follows_the_python_data_model(self):
        """A color compares against anything and works as a dict key.

        ``==`` against a non-color must be False rather than a TypeError,
        which every ``color in [...]`` downstream depends on. Defining
        ``__eq__`` also drops the inherited ``__hash__``, so the value
        hash has to be put back or the cycle cannot go in a set.
        """
        color = pilot.PlotColor(1, 2, 3)
        self.assertFalse(color == None)  # noqa: E711
        self.assertTrue(color != None)  # noqa: E711
        self.assertNotIn(color, [1, 'a'])
        self.assertEqual(10, len(set(pilot.plot_color_cycle())))
        self.assertEqual('custom', {color: 'custom'}[pilot.PlotColor(1, 2, 3)])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
