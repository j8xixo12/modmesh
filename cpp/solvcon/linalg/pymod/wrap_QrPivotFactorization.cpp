/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>

#include <solvcon/linalg/pymod/linalg_pymod.hpp>

namespace solvcon
{

namespace python
{

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapQrPivotFactorization
    : public WrapBase<WrapQrPivotFactorization<T>, QrPivotFactorization<T>>
{

    using root_base_type = WrapBase<WrapQrPivotFactorization<T>, QrPivotFactorization<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using array_type = SimpleArray<T>;

    friend root_base_type;

    WrapQrPivotFactorization(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapQrPivotFactorization */

template <typename T>
WrapQrPivotFactorization<T>::WrapQrPivotFactorization(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](array_type const & a)
                {
                    return std::make_unique<wrapped_type>(a);
                }),
            py::arg("a"));

    (*this)
        .def_property_readonly("m", &wrapped_type::m)
        .def_property_readonly("n", &wrapped_type::n)
        .def_property_readonly("k", &wrapped_type::k)
        .def_property_readonly(
            "perm",
            &wrapped_type::perm,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "tau",
            &wrapped_type::tau,
            py::return_value_policy::reference_internal)
        .def("r", &wrapped_type::r, "The k-by-n upper trapezoidal factor R.")
        .def("q", &wrapped_type::q, "The m-by-k thin factor Q.")
        .def(
            "apply_q",
            &wrapped_type::apply_q,
            py::arg("b"),
            "Return Q b with the full m-by-m Q.")
        .def(
            "apply_qt",
            &wrapped_type::apply_qt,
            py::arg("b"),
            "Return Q^T b with the full m-by-m Q.")
        .def(
            "rank",
            &wrapped_type::rank,
            py::arg("rtol") = T(-1),
            "Number of |R_ii| > rtol * |R_00|; rtol < 0 selects max(m, n) * eps.");
}

void wrap_QrPivotFactorization(pybind11::module & mod)
{
    WrapQrPivotFactorization<float>::commit(
        mod, "QrPivotFactorizationFloat32", "QR with column pivoting (float32)");
    WrapQrPivotFactorization<double>::commit(
        mod, "QrPivotFactorizationFloat64", "QR with column pivoting (float64)");

    // clang-format off
#define SC_DECL_QRDEF(T)                                                     \
    mod.def(                                                                 \
        "qr_pivot",                                                          \
        [](SimpleArray<T> const & a)                                         \
        {                                                                    \
            auto [q, r, perm] = qr_pivot(a);                                 \
            return pybind11::make_tuple(q, r, perm);                         \
        },                                                                   \
        pybind11::arg("a"),                                                  \
        "QR with column pivoting: return (Q, R, perm), A[:, perm] = Q R.");
    // clang-format on

    SC_DECL_QRDEF(float)
    SC_DECL_QRDEF(double)

#undef SC_DECL_QRDEF
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
