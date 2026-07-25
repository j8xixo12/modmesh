/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/plot/RPlotModel.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace
{

/// Clamp a widget extent into the pixel range view() can map onto. A widget
/// reports a zero or negative size before its first layout, and no widget is
/// PLOT_MAX_PIXEL_EXTENT across; both ends are clamped so that dividing the
/// extent by the span nonsingular_range() guarantees can never overflow.
double clamp_pixel_extent(std::int32_t extent)
{
    if (extent < 1)
    {
        return 1.0;
    }
    double const pixels = static_cast<double>(extent);
    return pixels < PLOT_MAX_PIXEL_EXTENT ? pixels : PLOT_MAX_PIXEL_EXTENT;
}

} /* end namespace */

std::array<double, 2> nonsingular_range(double lo, double hi, double margin)
{
    if (!std::isfinite(lo) || !std::isfinite(hi))
    {
        return {PLOT_DEFAULT_LIMITS[0], PLOT_DEFAULT_LIMITS[1]};
    }
    if (hi < lo)
    {
        std::swap(lo, hi);
    }
    if (!std::isfinite(margin) || margin < 0.0)
    {
        margin = 0.0;
    }

    // Finite bounds do not imply a finite width: [-1e308, 1e308] overflows to
    // an infinite span, and the view's zoom is pixels / span, so the whole
    // axis would collapse onto one pixel column. Halving about the center
    // once always suffices, because each bound is already finite.
    if (!std::isfinite(hi - lo))
    {
        lo *= 0.5;
        hi *= 0.5;
    }

    double const scale = std::max(std::abs(lo), std::abs(hi));
    if (scale < PLOT_MIN_MAGNITUDE)
    {
        // Every bound sits within a hair of zero, so there is no magnitude to
        // measure a width against and no finite zoom that could show one.
        return {-PLOT_DEGENERATE_ABSOLUTE_HALF_WIDTH, PLOT_DEGENERATE_ABSOLUTE_HALF_WIDTH};
    }

    double const span = hi - lo;
    if (span > scale * PLOT_DEGENERATE_RELATIVE_SPAN)
    {
        double const padded_lo = lo - margin * span;
        double const padded_hi = hi + margin * span;
        if (std::isfinite(padded_lo) && std::isfinite(padded_hi) && std::isfinite(padded_hi - padded_lo))
        {
            return {padded_lo, padded_hi};
        }
        // The margin itself overflowed. The unpadded range is already a
        // usable interval, and dropping the padding beats returning one that
        // is not.
        return {lo, hi};
    }

    // Degenerate: open the range about its center. A strictly positive width
    // is not enough to land here -- [1, 1 + 1e-310] has one, and it would give
    // an infinite zoom -- so the test above is relative to the magnitude of
    // the bounds. The relative half-width keeps the opened box in scale with
    // the value, and the absolute one covers the center-at-zero case where the
    // relative one is also zero.
    double const center = 0.5 * lo + 0.5 * hi; // Halved first: lo + hi can overflow.
    double half_width = std::abs(center) * PLOT_DEGENERATE_RELATIVE_HALF_WIDTH;
    if (!(half_width > 0.0) || !std::isfinite(half_width))
    {
        half_width = PLOT_DEGENERATE_ABSOLUTE_HALF_WIDTH;
    }
    lo = center - half_width;
    hi = center + half_width;
    if (std::isfinite(lo) && std::isfinite(hi) && hi > lo)
    {
        return {lo, hi};
    }
    return {PLOT_DEFAULT_LIMITS[0], PLOT_DEFAULT_LIMITS[1]};
}

void RPlotView::screen_from_data(double data_x, double data_y, double & screen_x, double & screen_y) const
{
    double unused = 0.0;
    x_axis.screen_from_world(data_x, 0.0, screen_x, unused);
    y_axis.screen_from_world(0.0, data_y, unused, screen_y);
}

void RPlotView::data_from_screen(double screen_x, double screen_y, double & data_x, double & data_y) const
{
    double unused = 0.0;
    x_axis.world_from_screen(screen_x, 0.0, data_x, unused);
    y_axis.world_from_screen(0.0, screen_y, unused, data_y);
}

std::shared_ptr<RPlotSeries> RPlotModel::add_series()
{
    return add_series(RPlotSeries());
}

std::shared_ptr<RPlotSeries> RPlotModel::add_series(RPlotSeries series)
{
    std::shared_ptr<RPlotSeries> const stored = std::make_shared<RPlotSeries>(std::move(series));
    m_series.push_back(stored);
    // A series that arrived with an explicit color keeps it and leaves the
    // cycle counter alone, so hand-colored series do not punch gaps in the
    // sequence the automatic ones walk through.
    if (!stored->color_is_set())
    {
        stored->set_color(plot_cycle_color(m_color_index));
        ++m_color_index;
    }
    ++m_structure_revision;
    return stored;
}

std::shared_ptr<RPlotSeries> RPlotModel::series(std::size_t index)
{
    if (index >= m_series.size())
    {
        throw std::out_of_range(
            std::format(
                "RPlotModel::series: index {} is out of bounds with series count {}",
                index,
                m_series.size()));
    }
    return m_series[index];
}

void RPlotModel::clear_series()
{
    m_series.clear();
    m_color_index = 0;
    ++m_structure_revision;
}

void RPlotModel::set_margin(double margin)
{
    if (!std::isfinite(margin) || margin < 0.0)
    {
        throw std::invalid_argument(
            std::format("RPlotModel::set_margin: margin must be finite and non-negative, but it is {}", margin));
    }
    m_margin = margin;
}

void RPlotModel::set_view_limits(double xmin, double xmax, double ymin, double ymax)
{
    // nonsingular_range() falls back to the unit box on a non-finite bound,
    // which is the right answer for data the model derived but the wrong one
    // for limits a caller asked for: a silent unit box is a wrong view where
    // the exception is a diagnosis.
    if (!std::isfinite(xmin) || !std::isfinite(xmax) || !std::isfinite(ymin) || !std::isfinite(ymax))
    {
        throw std::invalid_argument(
            std::format(
                "RPlotModel::set_view_limits: every bound must be finite, but they are {}, {}, {}, {}",
                xmin,
                xmax,
                ymin,
                ymax));
    }
    std::array<double, 2> const xr = nonsingular_range(xmin, xmax, 0.0);
    std::array<double, 2> const yr = nonsingular_range(ymin, ymax, 0.0);
    m_view_limits = {xr[0], xr[1], yr[0], yr[1]};
    m_autoscale = false;
}

std::uint64_t RPlotModel::data_revision() const
{
    std::uint64_t sig = 1469598103934665603ULL; // FNV-1a offset basis
    sig = (sig ^ m_structure_revision) * 1099511628211ULL; // FNV-1a prime
    for (auto const & sptr : m_series)
    {
        sig = (sig ^ sptr->revision()) * 1099511628211ULL;
    }
    return sig;
}

std::optional<std::array<double, 4>> RPlotModel::data_limits() const
{
    std::uint64_t const rev = data_revision();
    if (m_limits_valid && m_limits_revision == rev)
    {
        return m_limits;
    }

    std::optional<std::array<double, 4>> limits;
    for (auto const & sptr : m_series)
    {
        std::optional<std::array<double, 4>> const slim = sptr->data_limits();
        if (!slim.has_value())
        {
            continue;
        }
        if (!limits.has_value())
        {
            limits = *slim;
            continue;
        }
        std::array<double, 4> & lim = *limits;
        std::array<double, 4> const & sub = *slim;
        if (sub[0] < lim[0])
        {
            lim[0] = sub[0];
        }
        if (sub[1] > lim[1])
        {
            lim[1] = sub[1];
        }
        if (sub[2] < lim[2])
        {
            lim[2] = sub[2];
        }
        if (sub[3] > lim[3])
        {
            lim[3] = sub[3];
        }
    }

    m_limits = limits;
    m_limits_revision = rev;
    m_limits_valid = true;
    return m_limits;
}

std::array<double, 4> RPlotModel::view_limits() const
{
    if (!m_autoscale)
    {
        return m_view_limits;
    }
    std::optional<std::array<double, 4>> const dlim = data_limits();
    if (!dlim.has_value())
    {
        return PLOT_DEFAULT_LIMITS;
    }
    std::array<double, 2> const xr = nonsingular_range((*dlim)[0], (*dlim)[1], m_margin);
    std::array<double, 2> const yr = nonsingular_range((*dlim)[2], (*dlim)[3], m_margin);
    return {xr[0], xr[1], yr[0], yr[1]};
}

RPlotView RPlotModel::view(std::int32_t width, std::int32_t height) const
{
    // The division below is unguarded on purpose: nonsingular_range() bounds
    // the span away from zero relative to its own magnitude, and
    // clamp_pixel_extent() bounds the numerator, so neither the zoom nor the
    // pan it scales can reach an infinity or a NaN.
    double const wpx = clamp_pixel_extent(width);
    double const hpx = clamp_pixel_extent(height);
    std::array<double, 4> const lim = view_limits();

    double const zoom_x = wpx / (lim[1] - lim[0]);
    double const zoom_y = hpx / (lim[3] - lim[2]);

    RPlotView out;
    out.x_axis.set_zoom(zoom_x);
    out.x_axis.set_pan_x(-zoom_x * lim[0]);
    out.x_axis.set_pan_y(0.0);
    out.y_axis.set_zoom(zoom_y);
    out.y_axis.set_pan_x(0.0);
    out.y_axis.set_pan_y(zoom_y * lim[3]);
    return out;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
