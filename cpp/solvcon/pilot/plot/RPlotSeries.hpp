#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * One xy data series of the native plot: the SimpleArray pair that holds the
 * samples, the style used to stroke them, the revision counter every
 * downstream cache keys off, and the NaN-safe raw data limits.
 *
 * Qt-free on purpose, so the series math compiles into the no-GUI test target
 * and the eventual GPU backend can reuse it untouched.
 *
 * @ingroup group_domain
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <solvcon/buffer/SimpleArray.hpp>

#include <solvcon/pilot/plot/plot_style.hpp>

namespace solvcon
{

/**
 * @brief One xy data series: a contiguous SimpleArray<double> pair plus the
 * style used to stroke it.
 *
 * Samples are read only through size() / x() / y() / x_at() / y_at(), which
 * report the current extent at the moment they are called. Nothing
 * downstream keeps the SimpleArray, so a variable-size source can replace
 * the storage later without reworking the model, the ticker, the decimator,
 * or the widget. Every data change bumps revision(), which is the key every
 * downstream cache -- the series' own data_limits(), the model's data
 * limits, and the decimated point set -- tests for staleness.
 *
 * Not thread-safe: the lazy limits cache is mutable and the pilot paints on
 * the GUI thread.
 *
 * @ingroup group_domain
 */
class RPlotSeries
{

public:

    RPlotSeries() = default;
    RPlotSeries(RPlotSeries const &) = default;

    /// Moving hands the samples over and empties the source, whose revision
    /// advances so that anything keyed on it recomputes. The defaulted move
    /// cannot do this: SimpleArray's moved-from shape survives the move (the
    /// small_vector move copies its inline storage), so the source would go
    /// on reporting its old sample count over a body pointer that now aims
    /// into the destination's buffer. Emptying it builds a zero-length array
    /// and therefore allocates, which is why these two are not noexcept;
    /// nothing here stores an RPlotSeries in a container by value.
    RPlotSeries(RPlotSeries && other);

    /// Copy-and-move rather than the defaulted member-wise assignment:
    /// SimpleArray's copy assignment writes through ConcreteBuffer::operator=,
    /// which throws std::out_of_range when the buffers differ in size, so the
    /// defaulted form would fail on exactly the assignment that changes the
    /// sample count.
    RPlotSeries & operator=(RPlotSeries const & other);

    RPlotSeries & operator=(RPlotSeries && other);
    ~RPlotSeries() = default;

    /// Replace the samples. Both arrays must be 1-dimensional, ghost-free,
    /// unit-stride, and the same length; anything else throws
    /// std::invalid_argument and leaves the series untouched. Copies both
    /// buffers (SimpleArray's copy constructor clones the ConcreteBuffer);
    /// this is once per data change, never per frame. The "zero-copy" the
    /// plan claims is the NumPy-to-SimpleArray hand-off, not this store: a
    /// series that aliased the caller's buffer would change under the caches
    /// without bumping revision(), which is the one signal they key off.
    void set_data(SimpleArray<double> x, SimpleArray<double> y);

    /// Drop the samples. size() becomes 0 and data_limits() becomes empty.
    void clear_data();

    /// The number of samples right now.
    std::size_t size() const { return static_cast<std::size_t>(m_x.nbody()); }

    /// The current x samples; empty before the first set_data. Fetch it where
    /// it is used and do not store it: it is valid until the next set_data /
    /// clear_data today, and until the next append once a variable-size
    /// source is accepted, because growing one reallocates its storage.
    std::span<double const> x() const { return std::span<double const>(m_x.body(), size()); }

    /// The current y samples, with the same lifetime as x().
    std::span<double const> y() const { return std::span<double const>(m_y.body(), size()); }

    /// The x sample at @p it; throws std::out_of_range when it >= size().
    double x_at(std::size_t it) const;

    /// The y sample at @p it; throws std::out_of_range when it >= size().
    double y_at(std::size_t it) const;

    /// Strictly increasing across every data change. Caches key off it.
    std::uint64_t revision() const { return m_revision; }

    /// {xmin, xmax, ymin, ymax} over the samples whose x and y are both
    /// finite; empty when no such sample exists. May be degenerate.
    std::optional<std::array<double, 4>> data_limits() const;

    std::string const & label() const { return m_label; }
    void set_label(std::string label) { m_label = std::move(label); }

    PlotColor color() const { return m_color; }

    void set_color(PlotColor color)
    {
        m_color = color;
        m_color_is_set = true;
    }

    /// Whether a color has been resolved, either explicitly or by the
    /// model's cycle. RPlotModel::add_series assigns one when this is false.
    bool color_is_set() const { return m_color_is_set; }

    double line_width() const { return m_line_width; }

    /// Throws std::invalid_argument unless @p width is finite and positive.
    void set_line_width(double width);

private:

    /// Record that the samples are not the ones the caches were built over.
    /// The caches self-invalidate on the revision mismatch; nothing else has
    /// to be told.
    void note_data_change() { ++m_revision; }

    SimpleArray<double> m_x;
    SimpleArray<double> m_y;
    std::string m_label;
    PlotColor m_color;
    bool m_color_is_set = false;
    double m_line_width = PLOT_DEFAULT_LINE_WIDTH;
    std::uint64_t m_revision = 0;

    mutable bool m_limits_valid = false;
    mutable std::uint64_t m_limits_revision = 0;
    mutable std::optional<std::array<double, 4>> m_limits;

}; /* end class RPlotSeries */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
