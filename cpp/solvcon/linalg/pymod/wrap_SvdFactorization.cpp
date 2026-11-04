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
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapSvdFactorization
    : public WrapBase<WrapSvdFactorization<T>, SvdFactorization<T>>
{

    using root_base_type = WrapBase<WrapSvdFactorization<T>, SvdFactorization<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using array_type = SimpleArray<T>;

    friend root_base_type;

    WrapSvdFactorization(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapSvdFactorization */

template <typename T>
WrapSvdFactorization<T>::WrapSvdFactorization(pybind11::module & mod, char const * pyname, char const * pydoc)
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
        .def_property_readonly("sweeps", &wrapped_type::sweeps)
        .def_property_readonly("u", &wrapped_type::u, py::return_value_policy::reference_internal)
        .def_property_readonly("s", &wrapped_type::s, py::return_value_policy::reference_internal)
        .def_property_readonly("v", &wrapped_type::v, py::return_value_policy::reference_internal)
        .def(
            "rank",
            &wrapped_type::rank,
            py::arg("rtol") = T(-1),
            "Number of s_i > rtol * s_0; rtol < 0 selects max(m, n) * eps.")
        .def(
            "pinv",
            &wrapped_type::pinv,
            py::arg("rcond") = T(-1),
            "Moore-Penrose pseudo-inverse dropping s_i <= rcond * s_0.")
        .def(
            "solve",
            &wrapped_type::solve,
            py::arg("b"),
            py::arg("rcond") = T(-1),
            "Minimum-norm least-squares solution of A x = b.")
        .def(
            "null_space",
            &wrapped_type::null_space,
            py::arg("rcond") = T(-1),
            "Orthonormal null-space basis, n-by-(n - rank).");
}

void wrap_SvdFactorization(pybind11::module & mod)
{
    WrapSvdFactorization<float>::commit(
        mod, "SvdFactorizationFloat32", "One-sided Jacobi SVD (float32)");
    WrapSvdFactorization<double>::commit(
        mod, "SvdFactorizationFloat64", "One-sided Jacobi SVD (float64)");

    // clang-format off
#define SC_DECL_SVDDEF(T)                                                    \
    mod.def(                                                                 \
        "svd",                                                               \
        [](SimpleArray<T> const & a)                                         \
        {                                                                    \
            auto [u, s, v] = svd(a);                                         \
            return pybind11::make_tuple(u, s, v);                            \
        },                                                                   \
        pybind11::arg("a"),                                                  \
        "Thin SVD: return (U, s, V) with A = U diag(s) V^T.");               \
    mod.def(                                                                 \
        "pinv",                                                              \
        [](SimpleArray<T> const & a, T rcond)                                \
        { return pinv(a, rcond); },                                          \
        pybind11::arg("a"),                                                  \
        pybind11::arg("rcond") = T(-1));                                     \
    mod.def(                                                                 \
        "null_space",                                                        \
        [](SimpleArray<T> const & a, T rcond)                                \
        { return null_space(a, rcond); },                                    \
        pybind11::arg("a"),                                                  \
        pybind11::arg("rcond") = T(-1));                                     \
    mod.def(                                                                 \
        "lstsq",                                                             \
        [](SimpleArray<T> const & a, SimpleArray<T> const & b, T rcond)      \
        { return lstsq(a, b, rcond); },                                      \
        pybind11::arg("a"),                                                  \
        pybind11::arg("b"),                                                  \
        pybind11::arg("rcond") = T(-1));
    // clang-format on

    SC_DECL_SVDDEF(float)
    SC_DECL_SVDDEF(double)

#undef SC_DECL_SVDDEF
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
