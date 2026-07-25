/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/stl.h> // Must be the first include.

#include <solvcon/pilot/wrap_pilot.hpp> // Must be the first include but give way to above.
#include <solvcon/python/common.hpp>

#include <pybind11/operators.h>

#include <solvcon/buffer/pymod/SimpleArrayCaster.hpp>

#include <solvcon/pilot/plot/plot_style.hpp>
#include <solvcon/pilot/plot/RPlotModel.hpp>
#include <solvcon/pilot/plot/RPlotSeries.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace solvcon
{

namespace python
{

namespace
{

/// A 4-tuple is the documented shape of a limits value; letting
/// std::array<double, 4> auto-cast would hand out a list instead.
pybind11::object limits_to_python(std::optional<std::array<double, 4>> const & limits)
{
    namespace py = pybind11;
    if (!limits.has_value())
    {
        return py::none();
    }
    std::array<double, 4> const & lim = *limits;
    return py::make_tuple(lim[0], lim[1], lim[2], lim[3]);
}

/// Turn a Python index into the C++ one, raising IndexError for a negative
/// value. A std::size_t parameter would raise TypeError instead, and the
/// unsigned wrap-around would turn -1 into a huge in-range-looking number.
std::size_t checked_index(std::int64_t index, std::size_t count, char const * what, char const * noun)
{
    if (index < 0 || static_cast<std::size_t>(index) >= count)
    {
        throw std::out_of_range(
            std::format("{}: index {} is out of bounds with {} {}", what, index, noun, count));
    }
    return static_cast<std::size_t>(index);
}

} /* end namespace */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapPlotColor
    : public WrapBase<WrapPlotColor, PlotColor>
{

    friend root_base_type;

    WrapPlotColor(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        // Constructors.
        (*this)
            .def(py::init<>())
            .def(
                py::init<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>(),
                py::arg("r"),
                py::arg("g"),
                py::arg("b"),
                py::arg("a") = 255)
            //
            ;

        // Properties and methods. The channels are read-only on purpose: a
        // color is a value, and every getter that returns one -- notably
        // RPlotSeries.color -- returns a copy, so `series.color.a = 128` would
        // be a silent no-op if the channel were writable. Rebuild the color
        // instead, and the assignment goes where it is meant to.
        (*this)
            .def_readonly("r", &wrapped_type::r)
            .def_readonly("g", &wrapped_type::g)
            .def_readonly("b", &wrapped_type::b)
            .def_readonly("a", &wrapped_type::a)
            // py::self, not a lambda over the wrapped type: the operator form
            // carries py::is_operator(), which is what turns a comparison
            // against an unrelated object into NotImplemented instead of a
            // TypeError. `color == None` must be False, not an exception.
            .def(py::self == py::self) // NOLINT(misc-redundant-expression)
            .def(py::self != py::self) // NOLINT(misc-redundant-expression)
            // Defining __eq__ makes pybind11 drop __hash__, and a value type
            // that cannot go in a set or a dict key is a needless limit.
            .def(
                "__hash__",
                [](wrapped_type const & self)
                {
                    return static_cast<py::ssize_t>(
                        (static_cast<std::uint32_t>(self.a) << 24U) | (static_cast<std::uint32_t>(self.b) << 16U) |
                        (static_cast<std::uint32_t>(self.g) << 8U) | static_cast<std::uint32_t>(self.r));
                })
            .def(
                "__repr__",
                [](wrapped_type const & self)
                {
                    return std::format(
                        "PlotColor(r={}, g={}, b={}, a={})",
                        static_cast<int>(self.r),
                        static_cast<int>(self.g),
                        static_cast<int>(self.b),
                        static_cast<int>(self.a));
                })
            //
            ;
    }

}; /* end class WrapPlotColor */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRPlotSeries
    : public WrapBase<WrapRPlotSeries, RPlotSeries, std::shared_ptr<RPlotSeries>>
{

    friend root_base_type;

    WrapRPlotSeries(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        // Constructors.
        (*this)
            .def(py::init<>())
            //
            ;

        // Properties and methods.
        (*this)
            .def("set_data", &wrapped_type::set_data, py::arg("x"), py::arg("y"))
            .def("clear_data", &wrapped_type::clear_data)
            .def_property_readonly("size", &wrapped_type::size)
            .def(
                "__len__",
                [](wrapped_type const & self)
                { return self.size(); })
            .def(
                "x_at",
                [](wrapped_type const & self, std::int64_t index)
                { return self.x_at(checked_index(index, self.size(), "RPlotSeries::x_at", "size")); },
                py::arg("index"))
            .def(
                "y_at",
                [](wrapped_type const & self, std::int64_t index)
                { return self.y_at(checked_index(index, self.size(), "RPlotSeries::y_at", "size")); },
                py::arg("index"))
            .def_property_readonly("revision", &wrapped_type::revision)
            .def(
                "data_limits",
                [](wrapped_type const & self)
                { return limits_to_python(self.data_limits()); })
            .def_property("label", &wrapped_type::label, &wrapped_type::set_label)
            // A copy, not a reference into the series: PlotColor is a value
            // and its channels are read-only, so `series.color.a = 128` fails
            // loudly instead of writing to a temporary.
            .def_property(
                "color",
                &wrapped_type::color,
                &wrapped_type::set_color,
                py::return_value_policy::copy)
            .def_property_readonly("color_is_set", &wrapped_type::color_is_set)
            .def_property("line_width", &wrapped_type::line_width, &wrapped_type::set_line_width)
            //
            ;
    }

}; /* end class WrapRPlotSeries */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRPlotView
    : public WrapBase<WrapRPlotView, RPlotView>
{

    friend root_base_type;

    WrapRPlotView(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        // Constructors.
        (*this)
            .def(py::init<>())
            //
            ;

        // Properties and methods.
        (*this)
            .def_readwrite("x_axis", &wrapped_type::x_axis)
            .def_readwrite("y_axis", &wrapped_type::y_axis)
            .def(
                "screen_from_data",
                [](wrapped_type const & self, double data_x, double data_y)
                {
                    double screen_x = 0.0;
                    double screen_y = 0.0;
                    self.screen_from_data(data_x, data_y, screen_x, screen_y);
                    return py::make_tuple(screen_x, screen_y);
                },
                py::arg("data_x"),
                py::arg("data_y"))
            .def(
                "data_from_screen",
                [](wrapped_type const & self, double screen_x, double screen_y)
                {
                    double data_x = 0.0;
                    double data_y = 0.0;
                    self.data_from_screen(screen_x, screen_y, data_x, data_y);
                    return py::make_tuple(data_x, data_y);
                },
                py::arg("screen_x"),
                py::arg("screen_y"))
            .def_property_readonly("x_scale", &wrapped_type::x_scale)
            .def_property_readonly("y_scale", &wrapped_type::y_scale)
            //
            ;
    }

}; /* end class WrapRPlotView */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRPlotModel
    : public WrapBase<WrapRPlotModel, RPlotModel>
{

    friend root_base_type;

    WrapRPlotModel(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        // Constructors.
        (*this)
            .def(py::init<>())
            //
            ;

        // Properties and methods. The series accessors hand out the model's
        // own shared_ptr, so the Python object is the stored series -- every
        // in-place mutation the widget and the console make lands in the
        // model -- and it outlives clear_series() instead of dangling.
        (*this)
            .def(
                "add_series",
                static_cast<std::shared_ptr<RPlotSeries> (wrapped_type::*)()>(&wrapped_type::add_series))
            .def(
                "add_series",
                static_cast<std::shared_ptr<RPlotSeries> (wrapped_type::*)(RPlotSeries)>(&wrapped_type::add_series),
                py::arg("series"))
            .def(
                "series",
                [](wrapped_type & self, std::int64_t index)
                {
                    char const * const what = "RPlotModel::series";
                    return self.series(checked_index(index, self.series_count(), what, "series count"));
                },
                py::arg("index"))
            .def_property_readonly("series_count", &wrapped_type::series_count)
            .def(
                "__len__",
                [](wrapped_type const & self)
                { return self.series_count(); })
            .def("clear_series", &wrapped_type::clear_series)
            .def_property("margin", &wrapped_type::margin, &wrapped_type::set_margin)
            .def_property_readonly("autoscale_enabled", &wrapped_type::autoscale_enabled)
            .def("autoscale", &wrapped_type::autoscale)
            .def(
                "set_view_limits",
                &wrapped_type::set_view_limits,
                py::arg("xmin"),
                py::arg("xmax"),
                py::arg("ymin"),
                py::arg("ymax"))
            .def_property_readonly("data_revision", &wrapped_type::data_revision)
            .def(
                "data_limits",
                [](wrapped_type const & self)
                { return limits_to_python(self.data_limits()); })
            .def(
                "view_limits",
                [](wrapped_type const & self)
                {
                    std::array<double, 4> const lim = self.view_limits();
                    return py::make_tuple(lim[0], lim[1], lim[2], lim[3]);
                })
            .def("view", &wrapped_type::view, py::arg("width"), py::arg("height"))
            //
            ;
    }

}; /* end class WrapRPlotModel */

void wrap_plot(pybind11::module & mod)
{
    namespace py = pybind11;

    WrapPlotColor::commit(
        mod,
        "PlotColor",
        "One sRGB color with alpha, a byte per channel. An immutable value: "
        "the channels are read-only, so rebuild the color to change one. "
        "Qt-free so the color cycle tests without QtGui.");
    WrapRPlotSeries::commit(
        mod,
        "RPlotSeries",
        "One xy data series: a contiguous SimpleArrayFloat64 pair plus the "
        "style used to stroke it. Samples are read through size / x_at / y_at, "
        "which report the current extent, and every data change bumps "
        "revision.");
    WrapRPlotView::commit(
        mod,
        "RPlotView",
        "The data-to-screen mapping of one plot frame, as a pair of "
        "ViewTransform2dFp64 -- one per axis, because a single transform has "
        "one zoom and an xy plot needs independent x and y scales.");
    WrapRPlotModel::commit(
        mod,
        "RPlotModel",
        "The plot's series list, color cycle, data limits, and view limits. "
        "The transform is derived from the limits, never stored.");

    mod.def(
        "plot_color_cycle",
        []()
        {
            std::span<PlotColor const> const cycle = plot_color_cycle();
            return std::vector<PlotColor>(cycle.begin(), cycle.end());
        });
    mod.def(
        "plot_cycle_color",
        [](std::size_t index)
        { return plot_cycle_color(index); },
        py::arg("index"));
    mod.def(
        "plot_nonsingular_range",
        [](double lo, double hi, double margin)
        {
            std::array<double, 2> const out = nonsingular_range(lo, hi, margin);
            return py::make_tuple(out[0], out[1]);
        },
        py::arg("lo"),
        py::arg("hi"),
        py::arg("margin") = PLOT_DEFAULT_MARGIN);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
