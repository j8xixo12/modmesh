/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/stl.h> // Must be the first include.

#include <solvcon/pilot/wrap_pilot.hpp> // Must be the first include but give way to above.
#include <solvcon/python/common.hpp>

#include <pybind11/operators.h>

#include <solvcon/pilot/plot/plot_style.hpp>

#include <cstdint>
#include <format>
#include <span>
#include <vector>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapPlotColor
    : public WrapBase<WrapPlotColor, PlotColor>
{

    friend root_base_type;

    WrapPlotColor(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

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

        // A PlotColor is a value and every function yielding one yields a
        // copy, so a writable channel would let `plot_cycle_color(0).a = 128`
        // look like it recolored the cycle while recoloring nothing. Keeping
        // the channels read-only turns that into an error the caller sees.
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

void wrap_plot(pybind11::module & mod)
{
    namespace py = pybind11;

    WrapPlotColor::commit(
        mod,
        "PlotColor",
        "One sRGB color with alpha, a byte per channel. An immutable value: "
        "the channels are read-only, so rebuild the color to change one.");

    mod.def(
        "plot_color_cycle",
        []()
        {
            std::span<PlotColor const> const cycle = plot_color_cycle();
            return std::vector<PlotColor>(cycle.begin(), cycle.end());
        });
    mod.def("plot_cycle_color", &plot_cycle_color, py::arg("index"));
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
