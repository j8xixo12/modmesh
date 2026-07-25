#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Qt-free plot styling vocabulary: the color a series is stroked with, the
 * default stroke width, and the matplotlib C0-C9 categorical cycle the model
 * hands out to every series that carries no explicit color.
 *
 * Nothing here mentions Qt, so the whole vocabulary compiles into the no-GUI
 * test target and is checked on every runner. The widget converts a PlotColor
 * to a QColor at the boundary and nowhere else.
 *
 * @ingroup group_domain
 */

#include <cstddef>
#include <cstdint>
#include <span>

namespace solvcon
{

/**
 * @brief One sRGB color with alpha, a byte per channel.
 *
 * Deliberately free of Qt so the color cycle compiles and tests without
 * QtGui; the widget converts at the boundary with QColor(r, g, b, a).
 *
 * @ingroup group_domain
 */
struct PlotColor
{
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;

    constexpr PlotColor() = default;

    constexpr PlotColor(std::uint8_t red, std::uint8_t green, std::uint8_t blue, std::uint8_t alpha = 255)
        : r(red)
        , g(green)
        , b(blue)
        , a(alpha)
    {
    }

    friend constexpr bool operator==(PlotColor const & lhs, PlotColor const & rhs) = default;
}; /* end struct PlotColor */

/// Default stroke width in screen pixels; matplotlib's lines.linewidth.
inline constexpr double PLOT_DEFAULT_LINE_WIDTH = 1.5;

/// The matplotlib C0-C9 categorical cycle, in order. Exactly ten entries.
std::span<PlotColor const> plot_color_cycle();

/// The cycle entry for @p index, wrapping with index % plot_color_cycle().size().
PlotColor plot_cycle_color(std::size_t index);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
