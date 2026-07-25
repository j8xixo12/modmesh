/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/*
 * The plot core has a pytest suite that covers its behavior; this file exists
 * for the two things that suite structurally cannot do. First, its mere
 * presence in the test_nopython target makes `make gtest BUILD_QT=OFF` the
 * mechanical enforcer of the plot core's Qt-freedom: a Qt include added to
 * RPlotSeries, RPlotModel, or plot_style breaks this build. Second, the
 * std::span accessors are deliberately not bound to Python (a Python view
 * would dangle on the next set_data), so C++ is the only place they can be
 * exercised at all.
 */

#include <solvcon/pilot/plot/RPlotModel.hpp>
#include <solvcon/pilot/plot/RPlotSeries.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <utility>

using solvcon::RPlotModel;
using solvcon::RPlotSeries;
using solvcon::RPlotView;
using solvcon::SimpleArray;

TEST(PilotPlotSeries, SpanMatchesTheCurrentExtent)
{
    RPlotSeries series;
    series.set_data(SimpleArray<double>({0.0, 1.0, 2.0, 3.0}), SimpleArray<double>({10.0, 11.0, 12.0, 13.0}));

    EXPECT_EQ(series.x().size(), 4U);
    EXPECT_EQ(series.y().size(), 4U);
    EXPECT_DOUBLE_EQ(series.x()[0], 0.0);
    EXPECT_DOUBLE_EQ(series.x()[3], 3.0);
    EXPECT_DOUBLE_EQ(series.y()[0], 10.0);
    EXPECT_DOUBLE_EQ(series.y()[3], 13.0);
    // The x and y spans are two separate runs, not two windows on one buffer.
    EXPECT_NE(series.x().data(), series.y().data());
}

TEST(PilotPlotSeries, EmptySeriesHasAnEmptySpan)
{
    RPlotSeries series;
    EXPECT_TRUE(series.x().empty());
    EXPECT_TRUE(series.y().empty());

    series.set_data(SimpleArray<double>({0.0, 1.0}), SimpleArray<double>({2.0, 3.0}));
    series.clear_data();
    EXPECT_TRUE(series.x().empty());
    EXPECT_TRUE(series.y().empty());
}

TEST(PilotPlotSeries, MovedFromSeriesReportsNoStaleLimits)
{
    // A moved-from SimpleArray keeps its shape, so a defaulted move would
    // leave the source reporting its old sample count over a body pointer
    // aimed into the destination's buffer -- right until the destination
    // dies. Only C++ can reach a move, which is why the check lives here.
    RPlotSeries source;
    source.set_data(SimpleArray<double>({0.0, 1.0}), SimpleArray<double>({2.0, 3.0}));
    ASSERT_TRUE(source.data_limits().has_value());

    RPlotSeries const taken(std::move(source));
    EXPECT_EQ(taken.size(), 2U);
    ASSERT_TRUE(taken.data_limits().has_value());
    EXPECT_DOUBLE_EQ((*taken.data_limits())[1], 1.0);

    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_EQ(source.size(), 0U);
    EXPECT_TRUE(source.x().empty());
    EXPECT_TRUE(source.y().empty());
    EXPECT_FALSE(source.data_limits().has_value());
}

TEST(PilotPlotSeries, CopyAssignmentSurvivesALengthChange)
{
    // SimpleArray's copy assignment writes through the buffer and throws when
    // the sizes differ, so the defaulted member-wise form would fail on
    // exactly the assignment that changes the sample count.
    RPlotSeries source;
    source.set_data(SimpleArray<double>({0.0, 1.0, 2.0}), SimpleArray<double>({3.0, 4.0, 5.0}));
    RPlotSeries target;
    target.set_data(SimpleArray<double>({9.0}), SimpleArray<double>({9.0}));

    EXPECT_NO_THROW(target = source);
    EXPECT_EQ(target.size(), 3U);
    EXPECT_DOUBLE_EQ(target.x_at(2), 2.0);
}

TEST(PilotPlotSeries, SpanIsNotRebuiltPerCall)
{
    RPlotSeries series;
    series.set_data(SimpleArray<double>({0.0, 1.0, 2.0}), SimpleArray<double>({0.0, 1.0, 4.0}));

    double const * const first = series.x().data();
    double const * const second = series.x().data();
    // The accessor reports the current extent; it neither copies on read nor
    // hands out a snapshot that would go stale under a growing source.
    EXPECT_EQ(first, second);

    series.set_data(SimpleArray<double>({5.0, 6.0, 7.0}), SimpleArray<double>({0.0, 1.0, 4.0}));
    EXPECT_NE(series.x().data(), first);
    EXPECT_DOUBLE_EQ(series.x()[0], 5.0);
}

TEST(PilotPlotModel, ViewScalesTheAxesIndependently)
{
    // Time against pressure: the x and y units are unlike and their spans
    // differ by three orders of magnitude. One shared zoom would squeeze the
    // whole x extent into a handful of pixels, so the view must carry a scale
    // per axis. This is the C++ statement of the same guard the pytest suite
    // pins, so the invariant is checked in the no-Qt configuration too.
    RPlotModel model;
    model.set_view_limits(0.0, 100.0, 1.0e5, 1.1e5);
    RPlotView const view = model.view(800, 600);

    EXPECT_DOUBLE_EQ(view.x_scale(), 8.0);
    EXPECT_DOUBLE_EQ(view.y_scale(), 0.06);

    double screen_x = 0.0;
    double screen_y = 0.0;
    view.screen_from_data(100.0, 1.0e5, screen_x, screen_y);
    EXPECT_DOUBLE_EQ(screen_x, 800.0);
    EXPECT_DOUBLE_EQ(screen_y, 600.0);

    view.screen_from_data(0.0, 1.1e5, screen_x, screen_y);
    EXPECT_DOUBLE_EQ(screen_x, 0.0);
    EXPECT_DOUBLE_EQ(screen_y, 0.0);
}

TEST(PilotPlotModel, ClearedSeriesOutliveTheModelListing)
{
    // The model shares the ownership of its series, so a handle taken before
    // clear_series() keeps working instead of becoming a dangling read.
    RPlotModel model;
    std::shared_ptr<RPlotSeries> const handle = model.add_series();
    handle->set_data(SimpleArray<double>({0.0, 1.0}), SimpleArray<double>({2.0, 3.0}));
    ASSERT_EQ(handle.use_count(), 2L);

    model.clear_series();
    EXPECT_EQ(model.series_count(), 0U);
    EXPECT_EQ(handle.use_count(), 1L);
    EXPECT_EQ(handle->size(), 2U);
    EXPECT_DOUBLE_EQ(handle->x_at(1), 1.0);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
