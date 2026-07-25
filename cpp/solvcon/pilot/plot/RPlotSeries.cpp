/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/plot/RPlotSeries.hpp>

#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace
{

/**
 * Reject anything a std::span cannot describe. A strided or negative-stride
 * SimpleArray has no contiguous logical run, and a ghosted one starts its body
 * before the logical origin, so accepting either would draw the wrong values
 * with no error to notice. Fewer than two samples cannot be discontiguous, and
 * NumPy reports a stride of 0 for a zero-length array, so the stride is only
 * meaningful once there is a step to take. The caller validates every array
 * before it mutates anything, which is what makes a rejected set_data leave
 * the series exactly as it was.
 */
void validate_operand(SimpleArray<double> const & arr, char const * name)
{
    if (arr.ndim() != 1)
    {
        throw std::invalid_argument(
            std::format("RPlotSeries::set_data: {} must be 1-dimensional, but ndim is {}", name, arr.ndim()));
    }
    if (arr.nghost() != 0)
    {
        throw std::invalid_argument(
            std::format("RPlotSeries::set_data: {} must have no ghost, but nghost is {}", name, arr.nghost()));
    }
    if (arr.nbody() > 1 && arr.stride(0) != 1)
    {
        throw std::invalid_argument(
            std::format(
                "RPlotSeries::set_data: {} must be contiguous with unit stride, but stride is {}",
                name,
                arr.stride(0)));
    }
}

} /* end namespace */

RPlotSeries::RPlotSeries(RPlotSeries && other)
    : m_x(std::move(other.m_x))
    , m_y(std::move(other.m_y))
    , m_label(std::move(other.m_label))
    , m_color(other.m_color)
    , m_color_is_set(other.m_color_is_set)
    , m_line_width(other.m_line_width)
    , m_revision(other.m_revision)
{
    // The limits cache is not carried over; this series recomputes on first
    // use. clear_data() is what actually empties the source, and it bumps the
    // source's revision so its own cache recomputes too.
    other.clear_data();
}

RPlotSeries & RPlotSeries::operator=(RPlotSeries const & other)
{
    if (this != &other)
    {
        *this = RPlotSeries(other);
    }
    return *this;
}

RPlotSeries & RPlotSeries::operator=(RPlotSeries && other)
{
    if (this != &other)
    {
        m_x = std::move(other.m_x);
        m_y = std::move(other.m_y);
        m_label = std::move(other.m_label);
        m_color = other.m_color;
        m_color_is_set = other.m_color_is_set;
        m_line_width = other.m_line_width;
        m_revision = other.m_revision;
        m_limits_valid = false;
        other.clear_data();
    }
    return *this;
}

void RPlotSeries::set_data(SimpleArray<double> x, SimpleArray<double> y)
{
    validate_operand(x, "x");
    validate_operand(y, "y");
    if (x.nbody() != y.nbody())
    {
        throw std::invalid_argument(
            std::format(
                "RPlotSeries::set_data: x and y must have the same length, but they are {} and {}",
                x.nbody(),
                y.nbody()));
    }

    m_x = std::move(x);
    m_y = std::move(y);
    // The limits cache is left alone; it self-invalidates on the revision
    // mismatch, which is the same test a growing source would trip.
    note_data_change();
}

void RPlotSeries::clear_data()
{
    m_x = SimpleArray<double>();
    m_y = SimpleArray<double>();
    note_data_change();
}

double RPlotSeries::x_at(std::size_t it) const
{
    if (it >= size())
    {
        throw std::out_of_range(
            std::format("RPlotSeries::x_at: index {} is out of bounds with size {}", it, size()));
    }
    return x()[it];
}

double RPlotSeries::y_at(std::size_t it) const
{
    if (it >= size())
    {
        throw std::out_of_range(
            std::format("RPlotSeries::y_at: index {} is out of bounds with size {}", it, size()));
    }
    return y()[it];
}

std::optional<std::array<double, 4>> RPlotSeries::data_limits() const
{
    if (m_limits_valid && m_limits_revision == m_revision)
    {
        return m_limits;
    }

    std::optional<std::array<double, 4>> limits;
    std::span<double const> const xs = x();
    std::span<double const> const ys = y();
    for (std::size_t it = 0; it < xs.size(); ++it)
    {
        double const xv = xs[it];
        double const yv = ys[it];
        // A sample participates only when both of its coordinates are finite.
        // Skipping the whole sample -- rather than only the offending
        // coordinate -- keeps the reported box a box over real points: a
        // point with a NaN y is not at a known x either.
        if (!std::isfinite(xv) || !std::isfinite(yv))
        {
            continue;
        }
        if (!limits.has_value())
        {
            limits = std::array<double, 4>{xv, xv, yv, yv};
            continue;
        }
        std::array<double, 4> & lim = *limits;
        if (xv < lim[0])
        {
            lim[0] = xv;
        }
        if (xv > lim[1])
        {
            lim[1] = xv;
        }
        if (yv < lim[2])
        {
            lim[2] = yv;
        }
        if (yv > lim[3])
        {
            lim[3] = yv;
        }
    }

    m_limits = limits;
    m_limits_revision = m_revision;
    m_limits_valid = true;
    return m_limits;
}

void RPlotSeries::set_line_width(double width)
{
    if (!std::isfinite(width) || !(width > 0.0))
    {
        throw std::invalid_argument(
            std::format("RPlotSeries::set_line_width: width must be finite and positive, but it is {}", width));
    }
    m_line_width = width;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
