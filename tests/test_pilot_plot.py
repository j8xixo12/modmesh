# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the native xy-plot core: PlotColor and the C0-C9 cycle,
RPlotSeries, the nonsingular range guard, RPlotModel, and RPlotView.

The core is pure C++ math with no Qt widget, so everything here is exercised
through the pybind11 surface registered into ``solvcon.pilot``.
"""

import math
import re
import unittest

import numpy as np

import solvcon

try:
    from solvcon import pilot
except ImportError:
    pilot = None

EPS = 1e-9


def _array(values):
    """Wrap a sequence of numbers as a float64 SimpleArray.

    :param values: The sample values.
    :type values: list or numpy.ndarray
    :return: A zero-copy SimpleArray over a fresh float64 ndarray.
    :rtype: solvcon.SimpleArrayFloat64
    """
    return solvcon.SimpleArrayFloat64(array=np.array(values, dtype='float64'))


def _series(x_values, y_values):
    """Build an RPlotSeries holding the given samples.

    :param x_values: The x samples.
    :type x_values: list or numpy.ndarray
    :param y_values: The y samples.
    :type y_values: list or numpy.ndarray
    :return: The populated series.
    :rtype: solvcon.pilot.RPlotSeries
    """
    ser = pilot.RPlotSeries()
    ser.set_data(_array(x_values), _array(y_values))
    return ser


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PlotColorTC(unittest.TestCase):
    """The color vocabulary the model hands to series."""

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
        """An eleventh series must get C0 again, not an error or black.

        The cycle is a modulo, so a plot with any number of series always
        has a color for every one of them.
        """
        self.assertEqual(pilot.plot_cycle_color(0),
                         pilot.plot_cycle_color(10))
        self.assertEqual(pilot.plot_cycle_color(3),
                         pilot.plot_cycle_color(23))

    def test_color_fields_round_trip(self):
        """Alpha defaults to opaque and every channel is readable.

        The widget converts a PlotColor to a QColor field by field, so a
        channel that does not round-trip would silently mis-paint.
        """
        color = pilot.PlotColor(1, 2, 3)
        self.assertEqual(1, color.r)
        self.assertEqual(2, color.g)
        self.assertEqual(3, color.b)
        self.assertEqual(255, color.a)
        self.assertEqual(pilot.PlotColor(1, 2, 3, 255), color)
        self.assertNotEqual(pilot.PlotColor(1, 2, 3, 128), color)

    def test_color_channels_are_read_only(self):
        """A channel assignment must fail loudly, not vanish.

        Every getter that yields a color -- RPlotSeries.color above all --
        yields a copy, so ``series.color.a = 128`` would paint nothing
        different while looking like it had. Refusing the assignment is
        what turns that into an error the caller can see.
        """
        color = pilot.PlotColor(1, 2, 3)
        with self.assertRaises(AttributeError):
            color.r = 9
        with self.assertRaises(AttributeError):
            color.a = 128
        self.assertEqual(pilot.PlotColor(1, 2, 3), color)

    def test_color_compares_with_unrelated_objects(self):
        """``==`` against a non-color is False, never a TypeError.

        Python's data model says so, and every ``assertNotEqual(color,
        None)`` or ``color in [...]`` in downstream code depends on it.
        """
        color = pilot.PlotColor(1, 2, 3)
        self.assertFalse(color == None)  # noqa: E711
        self.assertTrue(color != None)  # noqa: E711
        self.assertFalse(color == 5)
        self.assertNotIn(color, [1, 'a'])

    def test_color_is_hashable(self):
        """A color must work as a set member and a dict key.

        Defining ``__eq__`` drops the inherited ``__hash__``, so the value
        hash has to be put back; deduplicating the cycle is the obvious
        thing a caller does with ten colors.
        """
        cycle = pilot.plot_color_cycle()
        self.assertEqual(10, len(set(cycle)))
        self.assertEqual(hash(pilot.PlotColor(1, 2, 3)),
                         hash(pilot.PlotColor(1, 2, 3)))
        named = {pilot.PlotColor(1, 2, 3): 'custom'}
        self.assertEqual('custom', named[pilot.PlotColor(1, 2, 3)])


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RPlotSeriesTC(unittest.TestCase):
    """The series: its samples, its revision, and its raw limits."""

    def test_fresh_series_is_empty_and_unstyled(self):
        """A default series must be legal to query before it has data.

        The widget may paint a series between its creation and its first
        set_data, so every accessor has to answer on an empty series
        instead of faulting.
        """
        ser = pilot.RPlotSeries()
        self.assertEqual(0, ser.size)
        self.assertEqual(0, len(ser))
        self.assertEqual(0, ser.revision)
        self.assertIsNone(ser.data_limits())
        self.assertFalse(ser.color_is_set)
        self.assertEqual('', ser.label)
        self.assertEqual(1.5, ser.line_width)

    def test_set_data_stores_the_samples(self):
        """The stored samples must be the ones handed in, at their index.

        This is the whole point of the C++ store: the widget reads the
        samples back by index without a Python round trip.
        """
        ser = _series([0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(4, ser.size)
        self.assertEqual(0.0, ser.x_at(0))
        self.assertEqual(2.0, ser.x_at(2))
        self.assertEqual(3.0, ser.x_at(3))
        self.assertEqual(10.0, ser.y_at(0))
        self.assertEqual(12.0, ser.y_at(2))
        self.assertEqual(13.0, ser.y_at(3))

    def test_revision_tracks_the_event_not_the_value(self):
        """Rewriting identical data must still bump the revision.

        Downstream caches key off the revision alone. If a rewrite with
        equal values did not bump it, a cache would keep a result computed
        over storage that has since been replaced.
        """
        ser = pilot.RPlotSeries()
        first = ser.revision
        ser.set_data(_array([0.0, 1.0]), _array([2.0, 3.0]))
        second = ser.revision
        self.assertGreater(second, first)
        ser.set_data(_array([0.0, 1.0]), _array([2.0, 3.0]))
        self.assertGreater(ser.revision, second)

    def test_clear_data_empties_and_bumps(self):
        """Clearing must empty the series and invalidate every cache.

        A cleared series that kept its old limits would keep the view
        scaled to data that is no longer there.
        """
        ser = _series([0.0, 1.0], [2.0, 3.0])
        before = ser.revision
        ser.clear_data()
        self.assertEqual(0, ser.size)
        self.assertIsNone(ser.data_limits())
        self.assertGreater(ser.revision, before)

    def test_index_out_of_range_raises_index_error(self):
        """Both an over-range and a negative index must raise IndexError.

        A negative index is the natural Python typo; it has to fail the
        same way as an over-range one instead of wrapping or raising
        TypeError from an unsigned parameter.
        """
        ser = _series([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
        with self.assertRaisesRegex(
                IndexError,
                re.escape('index 3 is out of bounds with size 3')):
            ser.x_at(3)
        with self.assertRaisesRegex(
                IndexError,
                re.escape('index -1 is out of bounds with size 3')):
            ser.x_at(-1)
        with self.assertRaisesRegex(
                IndexError,
                re.escape('index 3 is out of bounds with size 3')):
            ser.y_at(3)

    def test_mismatched_lengths_are_rejected(self):
        """An x and a y of different lengths is not a plottable series.

        Accepting it would mean drawing pairs that do not exist.
        """
        ser = pilot.RPlotSeries()
        with self.assertRaisesRegex(
                ValueError,
                re.escape('x and y must have the same length, '
                          'but they are 10 and 9')):
            ser.set_data(_array(np.arange(10, dtype='float64')),
                         _array(np.arange(9, dtype='float64')))

    def test_multidimensional_array_is_rejected(self):
        """A 2-D array has no single sample sequence to plot.

        The span accessors describe one contiguous run of samples, which a
        2-D array does not have.
        """
        ser = pilot.RPlotSeries()
        with self.assertRaisesRegex(
                ValueError,
                re.escape('must be 1-dimensional, but ndim is 2')):
            ser.set_data(_array(np.zeros((2, 3), dtype='float64')),
                         _array(np.zeros((2, 3), dtype='float64')))

    def test_strided_array_is_rejected(self):
        """A strided array must be refused rather than read as contiguous.

        This is the load-bearing validation: SimpleArray's body pointer
        walks elements, not strides, so plot(x[::2], y[::2]) would silently
        draw the wrong values and nothing would report an error.
        """
        ser = pilot.RPlotSeries()
        # Wrap the view itself: SimpleArrayFloat64 keeps the NumPy stride,
        # which is exactly the input that must not reach the span accessors.
        strided = solvcon.SimpleArrayFloat64(
            array=np.arange(10, dtype='float64')[::2])
        with self.assertRaisesRegex(
                ValueError,
                re.escape('must be contiguous with unit stride, '
                          'but stride is 2')):
            ser.set_data(strided, strided)

    def test_empty_arrays_make_an_empty_series(self):
        """Plotting nothing yet must work, whatever produced the arrays.

        A solver before its first output, a cleared plot being refilled,
        and ``plot([], [])`` from the console all land here. NumPy reports
        a stride of 0 for a zero-length array, so a stride check that ran
        unconditionally would reject the empty case for a reason that does
        not exist -- and reject it only for arrays built one way, since a
        zero-length slice of a real array reports a stride of 1.
        """
        for values in ([], np.zeros(0, dtype='float64'),
                       np.arange(10, dtype='float64')[5:5]):
            ser = _series(values, values)
            self.assertEqual(0, ser.size)
            self.assertIsNone(ser.data_limits())

    def test_single_sample_ignores_the_stride(self):
        """One sample cannot be discontiguous, whatever its stride says.

        A one-element view of a strided array carries the parent's stride
        and no step to take with it; refusing it would drop a legitimate
        series over an unobservable number.
        """
        one = solvcon.SimpleArrayFloat64(
            array=np.arange(4, dtype='float64')[::2][:1])
        ser = pilot.RPlotSeries()
        ser.set_data(one, one)
        self.assertEqual(1, ser.size)
        self.assertEqual(0.0, ser.x_at(0))

    def test_ghosted_array_is_rejected(self):
        """A ghosted array's body is not its buffer origin.

        Reading it as if it were would draw the ghost cells, so the series
        refuses the array instead of guessing.
        """
        ser = pilot.RPlotSeries()
        ghosted = _array(np.arange(10, dtype='float64'))
        ghosted.nghost = 1
        with self.assertRaisesRegex(
                ValueError,
                re.escape('must have no ghost, but nghost is 1')):
            ser.set_data(ghosted, _array(np.arange(9, dtype='float64')))

    def test_rejected_set_data_leaves_the_series_untouched(self):
        """Validation must finish before any mutation begins.

        Without the strong guarantee a bad call could half-replace a
        series -- a new x against an old y -- which is worse than the
        error it was trying to report.
        """
        ser = _series([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
        size = ser.size
        revision = ser.revision
        limits = ser.data_limits()
        with self.assertRaises(ValueError):
            ser.set_data(_array(np.arange(10, dtype='float64')),
                         _array(np.arange(9, dtype='float64')))
        self.assertEqual(size, ser.size)
        self.assertEqual(revision, ser.revision)
        self.assertEqual(limits, ser.data_limits())

    def test_data_limits_on_clean_data(self):
        """The raw extent is the exact min and max of each axis.

        Everything the view does is built on this box, so it must be the
        data's own numbers with nothing added.
        """
        ser = _series([3.0, -1.0, 2.0], [7.0, 9.0, -4.0])
        self.assertEqual((-1.0, 3.0, -4.0, 9.0), ser.data_limits())

    def test_nan_in_one_axis_drops_the_whole_sample(self):
        """A NaN y must remove the point from the x limits as well.

        A point with an unknown y is not a point at a known x either;
        letting its x stretch the axis would scale the plot to a sample
        that is never drawn.
        """
        ser = _series([0.0, 1.0, 2.0, 3.0],
                      [10.0, 11.0, 12.0, float('nan')])
        self.assertEqual((0.0, 2.0, 10.0, 12.0), ser.data_limits())

    def test_infinities_are_skipped_not_clamped(self):
        """An infinite sample must be skipped exactly like a NaN.

        Clamping it to the finite extent would invent a point; keeping it
        would make the view infinite. Skipping is the only honest answer.
        """
        ser = _series([0.0, 1.0, 2.0, 3.0],
                      [10.0, 11.0, 12.0, float('inf')])
        self.assertEqual((0.0, 2.0, 10.0, 12.0), ser.data_limits())
        ser = _series([float('-inf'), 1.0, 2.0, 3.0],
                      [10.0, 11.0, 12.0, 13.0])
        self.assertEqual((1.0, 3.0, 11.0, 13.0), ser.data_limits())

    def test_all_nan_series_has_no_limits(self):
        """A series with no finite pair reports no limits at all.

        There is no sentinel box that could be mistaken for real data;
        the caller has to handle the empty case, and the model does.
        """
        nans = [float('nan')] * 4
        ser = _series(nans, nans)
        self.assertIsNone(ser.data_limits())

    def test_single_sample_limits_are_degenerate(self):
        """One sample yields a zero-width box, and that is correct here.

        data_limits() reports the raw extent; opening a degenerate axis is
        the model's job, so the two concerns stay testable apart.
        """
        ser = _series([5.0], [7.0])
        self.assertEqual((5.0, 5.0, 7.0, 7.0), ser.data_limits())

    def test_limits_cache_is_stable_and_invalidates(self):
        """The cache must neither recompute needlessly nor go stale.

        The revision is the only invalidation signal, so a data change has
        to move the answer and a no-op must not.
        """
        ser = _series([0.0, 1.0], [2.0, 3.0])
        first = ser.data_limits()
        self.assertEqual(first, ser.data_limits())
        ser.set_data(_array([0.0, 4.0]), _array([2.0, 8.0]))
        self.assertEqual((0.0, 4.0, 2.0, 8.0), ser.data_limits())

    def test_style_changes_do_not_bump_the_revision(self):
        """The revision tracks data, not appearance.

        A recolor triggers a repaint, not a limits recompute or a
        re-decimation; bumping the revision for it would throw away work
        for nothing.
        """
        ser = _series([0.0, 1.0], [2.0, 3.0])
        revision = ser.revision
        ser.label = 'pressure'
        ser.color = pilot.PlotColor(1, 2, 3, 4)
        ser.line_width = 2.5
        self.assertEqual('pressure', ser.label)
        self.assertEqual(pilot.PlotColor(1, 2, 3, 4), ser.color)
        self.assertEqual(2.5, ser.line_width)
        self.assertEqual(revision, ser.revision)

    def test_bad_line_width_is_rejected(self):
        """A non-positive or non-finite stroke width is not paintable.

        QPainter would silently draw nothing (or a cosmetic hairline), so
        the error belongs at the setter where the caller can see it.
        """
        ser = pilot.RPlotSeries()
        for width in (0.0, -1.0, float('nan')):
            with self.assertRaises(ValueError):
                ser.line_width = width
        self.assertEqual(1.5, ser.line_width)

    def test_set_color_marks_the_color_resolved(self):
        """An explicit color must stop the model from overwriting it.

        color_is_set is the flag add_series() reads; without it a
        hand-colored series would be recolored by the cycle.
        """
        ser = pilot.RPlotSeries()
        self.assertFalse(ser.color_is_set)
        ser.color = pilot.PlotColor(10, 20, 30)
        self.assertTrue(ser.color_is_set)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PlotNonsingularTC(unittest.TestCase):
    """The margin and the degenerate-axis guard, tested directly."""

    def test_margin_is_a_fraction_of_the_span(self):
        """The pad is relative, so it looks the same at any data scale.

        This is matplotlib's axes.xmargin rule; a fixed absolute pad would
        be invisible on a large span and dominant on a small one.
        """
        self.assertEqual((-0.5, 10.5),
                         pilot.plot_nonsingular_range(0.0, 10.0, 0.05))

    def test_zero_span_opens_relative_to_its_center(self):
        """A constant series must still get a visible, in-scale window.

        Opening by a fraction of the value keeps the window sensible
        whether the constant is 5 or 5e6.
        """
        self.assertEqual((4.75, 5.25),
                         pilot.plot_nonsingular_range(5.0, 5.0, 0.05))

    def test_zero_span_at_zero_uses_the_absolute_half_width(self):
        """A constant of zero has no scale to be relative to.

        The relative half-width is zero there, so the guard must fall back
        to an absolute one or the axis would stay degenerate.
        """
        self.assertEqual((-0.5, 0.5),
                         pilot.plot_nonsingular_range(0.0, 0.0, 0.05))

    def test_reversed_bounds_are_swapped(self):
        """Bounds handed in backwards describe the same interval.

        Rejecting them would turn a harmless argument order into a failed
        plot; swapping is what the caller meant.
        """
        self.assertEqual((-0.5, 10.5),
                         pilot.plot_nonsingular_range(10.0, 0.0, 0.05))

    def test_non_finite_bounds_fall_back_to_the_unit_box(self):
        """A NaN or infinite bound must not poison the transform.

        The zoom is a division by the span; a non-finite span would make
        every mapped coordinate non-finite and nothing would draw.
        """
        self.assertEqual((0.0, 1.0),
                         pilot.plot_nonsingular_range(
                             float('nan'), 1.0, 0.05))
        self.assertEqual((0.0, 1.0),
                         pilot.plot_nonsingular_range(
                             float('inf'), 1.0, 0.05))
        self.assertEqual((0.0, 1.0),
                         pilot.plot_nonsingular_range(
                             0.0, float('-inf'), 0.05))

    def test_zero_margin_is_a_no_op_on_a_healthy_span(self):
        """A zero margin must leave a good range exactly alone.

        set_view_limits() pins explicit limits through this path, so any
        drift here would move a view the user asked for.
        """
        self.assertEqual((0.0, 10.0),
                         pilot.plot_nonsingular_range(0.0, 10.0, 0.0))

    def test_result_is_always_a_usable_interval(self):
        """Every input, however broken, yields a usable interval.

        This is the invariant the view derivation depends on: it divides
        by the span without checking. Finite bounds are not enough for
        that, so the sampled bounds reach both ends of the double range:
        1e308 pairs make the width overflow, and the subnormals make it
        so much smaller than the bounds themselves that the reciprocal
        would overflow instead.
        """
        bounds = [0.0, 1.0, -1.0, 5.0, -5.0, 1.0e-30, 1.0e30,
                  1.0e308, -1.0e308, 5.0e-324, 1.0e-310, 3.0e-310,
                  1.0 + 1.0e-15,
                  float('nan'), float('inf'), float('-inf')]
        margins = [0.0, 0.05, 1.0, 1.0e307, -1.0, float('nan')]
        for lo in bounds:
            for hi in bounds:
                for margin in margins:
                    low, high = pilot.plot_nonsingular_range(lo, hi, margin)
                    self.assertTrue(math.isfinite(low))
                    self.assertTrue(math.isfinite(high))
                    self.assertGreater(high, low)
                    self.assertTrue(math.isfinite(high - low))
                    # The width has to stay significant against the
                    # magnitude it sits at, or pixels / span overflows.
                    scale = max(abs(low), abs(high))
                    self.assertGreater(high - low, scale * 1e-15)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RPlotModelTC(unittest.TestCase):
    """The model: series list, color cycle, limits, and autoscale."""

    def test_fresh_model_is_drawable(self):
        """An empty model must already describe a paintable frame.

        The widget paints before anything is plotted, so the default view
        has to be a real box rather than an error or an empty optional.
        """
        model = pilot.RPlotModel()
        self.assertEqual(0, len(model))
        self.assertEqual(0, model.series_count)
        self.assertIsNone(model.data_limits())
        self.assertEqual((0.0, 1.0, 0.0, 1.0), model.view_limits())
        self.assertTrue(model.autoscale_enabled)
        self.assertEqual(0.05, model.margin)

    def test_add_series_returns_a_live_reference(self):
        """The returned series must be the stored one, not a copy.

        The console's plot(x, y) mutates the series it just added; a copy
        here would drop the data on the floor with no error.
        """
        model = pilot.RPlotModel()
        ser = model.add_series()
        ser.label = 'residual'
        ser.set_data(_array([0.0, 1.0]), _array([2.0, 3.0]))
        self.assertEqual('residual', model.series(0).label)
        self.assertEqual(2, model.series(0).size)

    def test_cycle_is_assigned_in_order(self):
        """Consecutive series must get C0, C1, C2 without repeating.

        Two series in one color is a plot the user cannot read.
        """
        model = pilot.RPlotModel()
        for _ in range(3):
            model.add_series()
        for index in range(3):
            self.assertEqual(pilot.plot_cycle_color(index),
                             model.series(index).color)

    def test_explicit_color_does_not_consume_a_cycle_slot(self):
        """A hand-colored series must not push the cycle forward.

        Otherwise a plot with one custom color would skip C0 for its
        automatic series, leaving a gap for no reason the user can see.
        """
        model = pilot.RPlotModel()
        custom = pilot.RPlotSeries()
        custom.color = pilot.PlotColor(1, 2, 3)
        model.add_series(custom)
        model.add_series()
        self.assertEqual(pilot.PlotColor(1, 2, 3), model.series(0).color)
        self.assertEqual(pilot.plot_cycle_color(0), model.series(1).color)

    def test_series_index_is_bounds_checked(self):
        """Out-of-range access must raise, never hand out garbage.

        A negative index is the natural Python typo and must fail the same
        way as an over-range one, rather than wrapping through an unsigned
        parameter into an in-range-looking element.
        """
        model = pilot.RPlotModel()
        model.add_series()
        with self.assertRaises(IndexError):
            model.series(1)
        with self.assertRaises(IndexError):
            model.series(-1)

    def test_series_handle_survives_clear_series(self):
        """A handle taken from the model must not become a dangling read.

        The model shares the ownership of its series with Python, so a
        script that kept a series and then cleared the plot reads its own
        object. Handing out a borrowed pointer instead would leave that
        script reading freed memory -- silently, because a freed and
        reused block answers with another series' samples.
        """
        model = pilot.RPlotModel()
        ser = model.add_series()
        ser.set_data(_array([0.0, 1.0]), _array([2.0, 3.0]))
        model.clear_series()
        self.assertEqual(0, model.series_count)
        self.assertEqual(2, ser.size)
        self.assertEqual(1.0, ser.x_at(1))
        self.assertEqual((0.0, 1.0, 2.0, 3.0), ser.data_limits())
        # Still writable, and no longer connected to the model.
        ser.set_data(_array([4.0]), _array([5.0]))
        self.assertEqual(1, ser.size)
        self.assertIsNone(model.data_limits())

    def test_series_accessor_returns_the_stored_object(self):
        """model.series(i) must be the series the model holds, not a copy.

        The console and the widget both mutate through it; a copy would
        drop every change with no error, and two calls returning two
        different objects would make the identity meaningless.
        """
        model = pilot.RPlotModel()
        added = model.add_series()
        self.assertIs(added, model.series(0))
        self.assertIs(model.series(0), model.series(0))
        model.series(0).label = 'residual'
        self.assertEqual('residual', added.label)

    def test_clear_series_restarts_the_cycle(self):
        """Clearing must reset the color counter, not keep counting.

        A cleared and refilled plot should look like a fresh one; keeping
        the counter would give its first series an arbitrary color.
        """
        model = pilot.RPlotModel()
        model.add_series()
        model.add_series()
        model.clear_series()
        self.assertEqual(0, model.series_count)
        ser = model.add_series()
        self.assertEqual(pilot.plot_cycle_color(0), ser.color)

    def test_data_limits_union_over_series(self):
        """The model's box must cover every series component-wise.

        Autoscale has to show all the data; taking one series' box would
        clip the others out of the frame.
        """
        model = pilot.RPlotModel()
        model.add_series(_series([0.0, 1.0], [0.0, 1.0]))
        model.add_series(_series([10.0, 20.0], [-5.0, -1.0]))
        self.assertEqual((0.0, 20.0, -5.0, 1.0), model.data_limits())

    def test_empty_series_are_skipped_in_the_union(self):
        """A series with no finite sample must not affect the union.

        A solver that has produced nothing yet is a normal state; it must
        not blank out the limits of the series that do have data.
        """
        nans = [float('nan')] * 3
        model = pilot.RPlotModel()
        model.add_series(_series([0.0, 1.0], [2.0, 3.0]))
        model.add_series(_series(nans, nans))
        self.assertEqual((0.0, 1.0, 2.0, 3.0), model.data_limits())

        blank = pilot.RPlotModel()
        blank.add_series(_series(nans, nans))
        blank.add_series(_series(nans, nans))
        self.assertIsNone(blank.data_limits())

    def test_data_revision_tracks_data_only(self):
        """The fold must move on data and structure, not on style.

        This is the seam a variable-size source will hang on: the model
        learns that a series changed without being told, and nothing
        recomputes when only the appearance moved.
        """
        model = pilot.RPlotModel()
        model.add_series()
        after_add = model.data_revision
        model.series(0).set_data(_array([0.0, 1.0]), _array([2.0, 3.0]))
        after_data = model.data_revision
        self.assertNotEqual(after_add, after_data)
        model.add_series()
        after_second_add = model.data_revision
        self.assertNotEqual(after_data, after_second_add)
        model.series(0).label = 'renamed'
        model.margin = 0.1
        self.assertEqual(after_second_add, model.data_revision)

    def test_autoscale_applies_the_margin(self):
        """Autoscaled limits must pad the data instead of touching it.

        A polyline drawn hard against the frame is unreadable; the margin
        is what keeps the extreme points visible.
        """
        model = pilot.RPlotModel()
        ramp = np.linspace(0.0, 10.0, 11, dtype='float64')
        model.add_series(_series(ramp, ramp))
        limits = model.view_limits()
        self.assertAlmostEqual(-0.5, limits[0], delta=EPS)
        self.assertAlmostEqual(10.5, limits[1], delta=EPS)

    def test_margin_change_takes_effect_without_invalidation(self):
        """A new margin must show up on the next view_limits() call.

        The margin is applied on top of the cached data limits every time,
        so there is no second cache to forget to invalidate.
        """
        model = pilot.RPlotModel()
        model.add_series(_series([0.0, 10.0], [0.0, 10.0]))
        self.assertAlmostEqual(-0.5, model.view_limits()[0], delta=EPS)
        model.margin = 0.0
        self.assertEqual((0.0, 10.0, 0.0, 10.0), model.view_limits())

    def test_bad_margin_is_rejected(self):
        """A negative or non-finite margin is not a padding fraction.

        A negative one would shrink the view inside the data; a NaN one
        would make every limit non-finite.
        """
        model = pilot.RPlotModel()
        for margin in (-0.1, float('nan')):
            with self.assertRaises(ValueError):
                model.margin = margin
        self.assertEqual(0.05, model.margin)

    def test_single_point_model_still_has_an_open_view(self):
        """One sample must still produce a box with a positive span.

        The view divides the pixel width by the span; a zero span would
        make the zoom infinite and the plot would vanish.
        """
        model = pilot.RPlotModel()
        model.add_series(_series([5.0], [7.0]))
        limits = model.view_limits()
        self.assertGreater(limits[1], limits[0])
        self.assertGreater(limits[3], limits[2])

    def test_constant_series_opens_its_flat_axis(self):
        """A flat y must be opened about its own value.

        A constant residual is a normal, meaningful plot; it has to be
        drawable rather than degenerate.
        """
        model = pilot.RPlotModel()
        model.add_series(_series([0.0, 1.0, 2.0], [5.0, 5.0, 5.0]))
        limits = model.view_limits()
        self.assertAlmostEqual(4.75, limits[2], delta=EPS)
        self.assertAlmostEqual(5.25, limits[3], delta=EPS)

        zero = pilot.RPlotModel()
        zero.add_series(_series([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]))
        limits = zero.view_limits()
        self.assertAlmostEqual(-0.5, limits[2], delta=EPS)
        self.assertAlmostEqual(0.5, limits[3], delta=EPS)

    def test_explicit_limits_pin_the_view(self):
        """Pinned limits must survive a data change until autoscale.

        A user who zoomed in does not want the next solver step to yank
        the view away; re-enabling autoscale is the explicit way back.
        """
        model = pilot.RPlotModel()
        ser = model.add_series()
        ser.set_data(_array([0.0, 1.0]), _array([0.0, 1.0]))
        model.set_view_limits(0.0, 2.0, 0.0, 2.0)
        self.assertFalse(model.autoscale_enabled)
        self.assertEqual((0.0, 2.0, 0.0, 2.0), model.view_limits())
        ser.set_data(_array([0.0, 100.0]), _array([0.0, 100.0]))
        self.assertEqual((0.0, 2.0, 0.0, 2.0), model.view_limits())
        model.autoscale()
        self.assertTrue(model.autoscale_enabled)
        self.assertAlmostEqual(105.0, model.view_limits()[1], delta=EPS)

    def test_explicit_limits_are_guarded_too(self):
        """A degenerate explicit box must be opened, not accepted.

        Every path into view_limits() goes through the same guard, so a
        caller cannot pin a zero-span view that would divide by zero.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(1.0, 1.0, 2.0, 2.0)
        limits = model.view_limits()
        self.assertGreater(limits[1], limits[0])
        self.assertGreater(limits[3], limits[2])

    def test_non_finite_explicit_limits_are_rejected(self):
        """A NaN bound the caller asked for is an error, not a default box.

        The guard's silent fallback is right for limits the model derived
        from data it did not choose; for limits a caller passed in it
        would pin a view nobody asked for. set_margin() already raises on
        a NaN, and the two setters must not disagree.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(0.0, 2.0, 0.0, 2.0)
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                model.set_view_limits(bad, 1.0, 0.0, 1.0)
            with self.assertRaises(ValueError):
                model.set_view_limits(0.0, 1.0, 0.0, bad)
        self.assertEqual((0.0, 2.0, 0.0, 2.0), model.view_limits())

    def test_model_without_finite_samples_uses_the_default_box(self):
        """Series full of NaN must fall back to the unit box.

        The widget still has to paint a frame; falling back keeps the
        axes and the grid sensible until real data arrives.
        """
        nans = [float('nan')] * 3
        model = pilot.RPlotModel()
        model.add_series(_series(nans, nans))
        self.assertEqual((0.0, 1.0, 0.0, 1.0), model.view_limits())


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RPlotViewTC(unittest.TestCase):
    """The derived data-to-screen mapping."""

    def test_corners_map_to_the_pixel_rect(self):
        """The view limits must land exactly on the widget corners.

        Any offset here shifts every drawn point, and an off-by-one at the
        corner is what makes a plot look mis-registered against its frame.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(0.0, 200.0, 0.0, 100.0)
        view = model.view(400, 200)
        self.assertEqual((0.0, 0.0), view.screen_from_data(0.0, 100.0))
        self.assertEqual((400.0, 200.0), view.screen_from_data(200.0, 0.0))

    def test_axes_scale_independently(self):
        """x and y must carry their own scale, not one shared zoom.

        Time against pressure spans 100 on x and 1e4 on y. A single
        ViewTransform2d has one zoom, so an isotropic fit would set both
        scales to 0.06 and squeeze the entire x extent into six pixels.
        This test fails loudly if the view is ever "simplified" back to a
        single transform.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(0.0, 100.0, 1.0e5, 1.1e5)
        view = model.view(800, 600)
        self.assertAlmostEqual(8.0, view.x_scale, delta=EPS)
        self.assertAlmostEqual(0.06, view.y_scale, delta=EPS)
        self.assertEqual(800.0, view.screen_from_data(100.0, 1.0e5)[0])

    def test_round_trip_through_the_inverse(self):
        """data_from_screen must undo screen_from_data.

        Pan, zoom, and hit-testing all go from a cursor position back to a
        data coordinate; a broken inverse breaks every interaction.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(-3.0, 7.0, 100.0, 900.0)
        view = model.view(640, 480)
        for data_x, data_y in ((-3.0, 100.0), (0.0, 500.0),
                               (2.5, 733.25), (7.0, 900.0)):
            screen_x, screen_y = view.screen_from_data(data_x, data_y)
            back_x, back_y = view.data_from_screen(screen_x, screen_y)
            self.assertAlmostEqual(data_x, back_x, delta=EPS)
            self.assertAlmostEqual(data_y, back_y, delta=EPS)

    def test_y_axis_is_flipped(self):
        """A larger data y must land higher on the screen.

        Screen y grows downward while data y grows upward; without the
        flip every plot would be drawn upside down.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(0.0, 1.0, 0.0, 10.0)
        view = model.view(100, 100)
        low = view.screen_from_data(0.5, 1.0)[1]
        high = view.screen_from_data(0.5, 9.0)[1]
        self.assertLess(high, low)

    def test_degenerate_widget_size_is_clamped(self):
        """A zero or negative pixel size must not poison the transform.

        A widget is zero-sized before its first layout; producing a NaN
        zoom there would leave the plot blank even after it is resized.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(0.0, 10.0, 0.0, 10.0)
        for width, height in ((0, 0), (-5, -5)):
            view = model.view(width, height)
            self.assertTrue(math.isfinite(view.x_scale))
            self.assertTrue(math.isfinite(view.y_scale))
            self.assertGreater(view.x_scale, 0.0)
            self.assertGreater(view.y_scale, 0.0)

    def test_degenerate_data_still_yields_a_usable_transform(self):
        """No data or view limits may produce a zero or infinite scale.

        The scale is a pixel extent over a data span, and both ends can
        break it: limits wide enough to overflow the span collapse every
        point onto one column, and a span too small against its own
        magnitude sends the reciprocal to infinity. Either way the widget
        paints NaNs and nothing appears.
        """
        cases = []
        wide = pilot.RPlotModel()
        wide.set_view_limits(-1e308, 1e308, 0.0, 1.0)
        cases.append(wide)
        for values in ([1e-310, 3e-310], [1.0, 1.0000000000000002], [5.0]):
            tiny = pilot.RPlotModel()
            tiny.add_series(_series(values, values))
            cases.append(tiny)
        huge_margin = pilot.RPlotModel()
        huge_margin.add_series(_series([0.0, 10.0], [0.0, 10.0]))
        huge_margin.margin = 1e307
        cases.append(huge_margin)

        for model in cases:
            limits = model.view_limits()
            self.assertTrue(math.isfinite(limits[1] - limits[0]))
            self.assertTrue(math.isfinite(limits[3] - limits[2]))
            view = model.view(800, 600)
            for scale in (view.x_scale, view.y_scale):
                self.assertTrue(math.isfinite(scale))
                self.assertGreater(scale, 0.0)
            screen_x, screen_y = view.screen_from_data(
                0.5 * (limits[0] + limits[1]), 0.5 * (limits[2] + limits[3]))
            self.assertTrue(math.isfinite(screen_x))
            self.assertTrue(math.isfinite(screen_y))
            data_x, data_y = view.data_from_screen(400.0, 300.0)
            self.assertTrue(math.isfinite(data_x))
            self.assertTrue(math.isfinite(data_y))

    def test_absurd_widget_size_is_clamped(self):
        """An implausible pixel extent must not overflow the scale.

        The widget size is an int32 at the binding, so a caller can pass
        one no monitor has. The scale is that extent over the data span,
        and the guard gives the span a floor rather than a fixed value, so
        a large enough numerator overflows it anyway. Clamping the extent
        is what lets view() divide by the span with no further check.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(1e-285, 1e-285 + 5e-300, 0.0, 1.0)
        for extent in (1, 800, 2 ** 20, 2 ** 31 - 1):
            view = model.view(extent, extent)
            self.assertTrue(math.isfinite(view.x_scale))
            self.assertGreater(view.x_scale, 0.0)
            self.assertTrue(
                math.isfinite(view.screen_from_data(1e-285, 0.5)[0]))

    def test_the_axes_are_real_view_transforms(self):
        """Each axis must be a ViewTransform2dFp64 driving its own channel.

        Reusing the existing transform, rather than adding a new affine,
        is what lets Step 7's pan and zoom helpers work unchanged. A
        freshly derived view starts with the unused channel of each member
        at zero; that is a starting value and not an invariant, because
        ViewTransform2d.zoom_at writes both pans.
        """
        model = pilot.RPlotModel()
        model.set_view_limits(0.0, 10.0, 0.0, 100.0)
        view = model.view(200, 400)
        self.assertIsInstance(view.x_axis, solvcon.ViewTransform2dFp64)
        self.assertIsInstance(view.y_axis, solvcon.ViewTransform2dFp64)
        data_x = 3.0
        data_y = 40.0
        screen_x, screen_y = view.screen_from_data(data_x, data_y)
        self.assertEqual(view.x_axis.screen_from_world(data_x, 0.0)[0],
                         screen_x)
        self.assertEqual(view.y_axis.screen_from_world(0.0, data_y)[1],
                         screen_y)
        self.assertEqual(0.0, view.x_axis.pan_y)
        self.assertEqual(0.0, view.y_axis.pan_x)

    def test_view_is_derived_not_cached(self):
        """A view taken after a data change must reflect the new data.

        The model stores limits, not a transform, so autoscale and resize
        are fresh derivations; a cached transform would go stale silently.
        """
        model = pilot.RPlotModel()
        ser = model.add_series()
        ser.set_data(_array([0.0, 1.0]), _array([0.0, 1.0]))
        before = model.view(100, 100)
        ser.set_data(_array([0.0, 100.0]), _array([0.0, 100.0]))
        after = model.view(100, 100)
        self.assertNotAlmostEqual(before.x_scale, after.x_scale,
                                  delta=EPS)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
