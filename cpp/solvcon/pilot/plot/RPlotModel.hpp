#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The native plot's model: the series list, the color cycle, the cached
 * NaN-safe data limits, the view limits with an autoscale margin and a
 * nonsingular guard, and the data-to-screen mapping derived from them.
 *
 * Qt-free on purpose, so the plot arithmetic compiles into the no-GUI test
 * target and a later GPU backend reuses it untouched.
 *
 * @ingroup group_domain
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <solvcon/universe/ViewTransform2d.hpp>

#include <solvcon/pilot/plot/RPlotSeries.hpp>

namespace solvcon
{

/// Autoscale padding as a fraction of the data span, added on each side;
/// matplotlib's axes.xmargin / axes.ymargin default.
inline constexpr double PLOT_DEFAULT_MARGIN = 0.05;

/// Half-width used to open a degenerate axis, relative to its center.
inline constexpr double PLOT_DEGENERATE_RELATIVE_HALF_WIDTH = 0.05;

/// Half-width used to open a degenerate axis whose center is zero.
inline constexpr double PLOT_DEGENERATE_ABSOLUTE_HALF_WIDTH = 0.5;

/// A range counts as degenerate once its width falls to this fraction of its
/// own magnitude, even though the width is still strictly positive;
/// matplotlib's nonsingular() "tiny".
inline constexpr double PLOT_DEGENERATE_RELATIVE_SPAN = 1.0e-15;

/// A range whose bounds are all smaller than this carries no usable scale to
/// measure a width against, so the guard falls back to a fixed box instead;
/// matplotlib's nonsingular() (1e6 / tiny) * smallest-normal test.
inline constexpr double PLOT_MIN_MAGNITUDE =
    (1.0e6 / PLOT_DEGENERATE_RELATIVE_SPAN) * std::numeric_limits<double>::min();

/// The largest pixel extent view() maps onto. No widget is a million pixels
/// across; clamping is what lets the derived zoom stay finite for every
/// int32_t a caller can hand in.
inline constexpr double PLOT_MAX_PIXEL_EXTENT = 1.0e6;

/// The view limits a model with no finite data falls back to.
inline constexpr std::array<double, 4> PLOT_DEFAULT_LIMITS = {0.0, 1.0, 0.0, 1.0};

/**
 * Pad [lo, hi] by @p margin of its span on each side and open it if it is
 * degenerate. The result always satisfies hi > lo, both bounds finite, and
 * hi - lo finite and large enough against max(|lo|, |hi|) that the view's
 * zoom (pixels / span) and pan (zoom * bound) both stay finite.
 *
 * This is the single normative definition of both the margin and the guard.
 * Every RPlotModel::view_limits() answer either comes from it or is the
 * PLOT_DEFAULT_LIMITS constant, which already satisfies the same invariant,
 * so the nondegeneracy holds for autoscaled and explicit limits alike.
 *
 * The margin is applied first and the guard second, which compose without
 * double-padding because the margin is a no-op on a zero span (0.05 * 0 == 0).
 * [0, 10] with margin 0.05 becomes [-0.5, 10.5]; [5, 5] becomes [4.75, 5.25];
 * [0, 0] becomes [-0.5, 0.5]; a non-finite bound falls back to [0, 1]; a
 * bound pair whose width overflows, such as [-1e308, 1e308], is halved about
 * its center until the width is representable.
 */
std::array<double, 2> nonsingular_range(double lo, double hi, double margin);

/**
 * @brief The data-to-screen mapping of one plot frame, as a pair of
 * ViewTransform2dFp64 -- one per axis.
 *
 * A single ViewTransform2d has one zoom, so it cannot carry independent x
 * and y scales, which every xy plot with unlike units needs. It is however
 * two independent 1-D channels sharing that zoom, so one instance per axis
 * gives the four degrees of freedom while reusing the existing transform
 * whole: its formulas, its pan / zoom_at / zoom_at_clamped helpers, and its
 * +Y-up flip.
 *
 * The unused channel of each member must never be read. A freshly derived
 * view leaves it at zero, but that is a starting value and not an invariant:
 * ViewTransform2d::zoom_at writes both pans, so the first interactive zoom
 * puts a number in it.
 *
 * @ingroup group_domain
 */
struct RPlotView
{
    /// Carries the x axis: screen_x = x_axis.zoom() * data_x + x_axis.pan_x().
    /// Its pan_y is unused.
    ViewTransform2dFp64 x_axis;

    /// Carries the y axis: screen_y = y_axis.pan_y() - y_axis.zoom() * data_y.
    /// Its pan_x is unused.
    ViewTransform2dFp64 y_axis;

    /// Map a data point onto the pixel rect, each axis through its own
    /// channel. Screen y grows downward, which the y axis' +Y-up flip does.
    void screen_from_data(double data_x, double data_y, double & screen_x, double & screen_y) const;

    /// The inverse of screen_from_data: the data point under a pixel.
    void data_from_screen(double screen_x, double screen_y, double & data_x, double & data_y) const;

    /// Screen pixels per data unit on x.
    double x_scale() const { return x_axis.zoom(); }

    /// Screen pixels per data unit on y.
    double y_scale() const { return y_axis.zoom(); }

}; /* end struct RPlotView */

/**
 * @brief The plot's series list, color cycle, data limits, and view limits.
 *
 * The model owns the limits; the transform is derived, never stored, so a
 * resize or an autoscale is a fresh derivation rather than a mutation. The
 * data limits are cached and keyed on data_revision(), a fold over every
 * series' revision plus a structural counter, so a series that grows or is
 * replaced invalidates the cache without the model being told.
 *
 * Not thread-safe.
 *
 * @ingroup group_domain
 */
class RPlotModel
{

public:

    RPlotModel() = default;
    RPlotModel(RPlotModel const &) = delete;
    RPlotModel & operator=(RPlotModel const &) = delete;
    RPlotModel(RPlotModel &&) noexcept = default;
    RPlotModel & operator=(RPlotModel &&) noexcept = default;
    ~RPlotModel() = default;

    /// Append an empty series and return the shared handle to it. The handle
    /// is returned by value: it shares the ownership rather than pointing
    /// into the series list, so it survives anything the list does next.
    std::shared_ptr<RPlotSeries> add_series();

    /// Append @p series (moved in) and return the shared handle to the stored
    /// copy, not to the argument.
    std::shared_ptr<RPlotSeries> add_series(RPlotSeries series);

    std::size_t series_count() const { return m_series.size(); }

    /// Throws std::out_of_range when @p index >= series_count(). Non-const
    /// only: the handle carries write access to the series, which a const
    /// model must not hand out.
    std::shared_ptr<RPlotSeries> series(std::size_t index);

    /// Drop every series and restart the color cycle at C0. A handle taken
    /// before the call keeps its series alive and usable; it is simply no
    /// longer in the model.
    void clear_series();

    double margin() const { return m_margin; }

    /// Throws std::invalid_argument unless @p margin is finite and >= 0.
    void set_margin(double margin);

    bool autoscale_enabled() const { return m_autoscale; }

    /// Re-enable autoscale: view_limits() follows the data again. Nothing
    /// else to do, because view_limits() derives from the data on demand, so
    /// a later data change is picked up without another call.
    void autoscale() { m_autoscale = true; }

    /// Pin the view to explicit limits and disable autoscale. Throws
    /// std::invalid_argument on a non-finite bound, the way set_margin()
    /// rejects a non-finite margin. Bounds handed in the other way round
    /// describe the same interval and are swapped. The limits then pass
    /// through nonsingular_range with a zero margin, so the nondegeneracy
    /// invariant holds for explicit limits too.
    void set_view_limits(double xmin, double xmax, double ymin, double ymax);

    /// Changes whenever any series' data changes or the series list changes.
    /// A fold, not a counter: two different histories could in principle
    /// collide, but only across a 64-bit revision gap no monotonic counter
    /// reaches.
    std::uint64_t data_revision() const;

    /// The union of every series' data_limits(); empty when no series has
    /// any finite sample.
    std::optional<std::array<double, 4>> data_limits() const;

    /// {xmin, xmax, ymin, ymax} of the drawn region. Always finite with
    /// xmax > xmin and ymax > ymin.
    std::array<double, 4> view_limits() const;

    /// The mapping that puts view_limits() on a width x height pixel rect.
    /// The rect is clamped into [1, PLOT_MAX_PIXEL_EXTENT] on each axis, so
    /// both derived scales are finite and strictly positive whatever the
    /// caller passes.
    RPlotView view(std::int32_t width, std::int32_t height) const;

private:

    // The series are held behind shared_ptr because add_series() and series()
    // hand the handle to Python; a std::vector<RPlotSeries> would invalidate
    // every raw reference on reallocation, and a unique_ptr would leave the
    // Python objects dangling on clear_series(). Sharing the ownership is
    // what makes both a use-after-free the script cannot cause.
    std::vector<std::shared_ptr<RPlotSeries>> m_series;
    std::uint64_t m_structure_revision = 0;
    std::size_t m_color_index = 0;
    double m_margin = PLOT_DEFAULT_MARGIN;
    bool m_autoscale = true;
    std::array<double, 4> m_view_limits = PLOT_DEFAULT_LIMITS;

    mutable bool m_limits_valid = false;
    mutable std::uint64_t m_limits_revision = 0;
    mutable std::optional<std::array<double, 4>> m_limits;

}; /* end class RPlotModel */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
