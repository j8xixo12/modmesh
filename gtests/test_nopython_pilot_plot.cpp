/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/*
 * The plot core is covered by tests/test_pilot_plot.py. This file exists for
 * what that suite structurally cannot do: its presence in the test_nopython
 * target makes `make gtest BUILD_QT=OFF` the mechanical enforcer of the plot
 * core's Qt-freedom, and the span accessor is never bound to Python.
 */

#include <solvcon/pilot/plot/plot_style.hpp>

#include <span>

#include <gtest/gtest.h>

using solvcon::plot_color_cycle;
using solvcon::plot_cycle_color;
using solvcon::PlotColor;

TEST(PilotPlotStyle, CycleSpansStaticStorage)
{
    // Python only ever sees a list of copies, so the lifetime the header
    // promises (and the wrap landing on the same table) is checkable here
    // alone. A per-call table would hand out a span that dangles at once.
    std::span<PlotColor const> const first = plot_color_cycle();
    std::span<PlotColor const> const second = plot_color_cycle();

    EXPECT_EQ(first.size(), 10U);
    EXPECT_EQ(first.data(), second.data());
    EXPECT_EQ(plot_cycle_color(13), first[3]);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
