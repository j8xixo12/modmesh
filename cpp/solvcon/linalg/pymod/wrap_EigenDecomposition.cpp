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
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapEigenDecomposition
    : public WrapBase<WrapEigenDecomposition<T>, EigenDecomposition<T>>
{

    using root_base_type = WrapBase<WrapEigenDecomposition<T>, EigenDecomposition<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using array_type = SimpleArray<T>;

    friend root_base_type;

    WrapEigenDecomposition(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapEigenDecomposition */

template <typename T>
WrapEigenDecomposition<T>::WrapEigenDecomposition(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](array_type const & a, bool compute_vectors, bool balance)
                {
                    return std::make_unique<wrapped_type>(a, compute_vectors, balance);
                }),
            py::arg("a"),
            py::arg("compute_vectors") = true,
            py::arg("balance") = true);

    (*this)
        .def_property_readonly("n", &wrapped_type::n)
        .def_property_readonly("has_vectors", &wrapped_type::has_vectors)
        .def_property_readonly("iterations", &wrapped_type::iterations)
        .def_property_readonly(
            "eigenvalues",
            &wrapped_type::eigenvalues,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "eigenvectors",
            &wrapped_type::eigenvectors,
            py::return_value_policy::reference_internal);
}

void wrap_EigenDecomposition(pybind11::module & mod)
{
    WrapEigenDecomposition<float>::commit(
        mod, "EigenDecompositionFloat32", "Native nonsymmetric eigensolver (float32)");
    WrapEigenDecomposition<double>::commit(
        mod, "EigenDecompositionFloat64", "Native nonsymmetric eigensolver (float64)");

    // clang-format off
#define SC_DECL_EIGDEF(T)                                                    \
    mod.def(                                                                 \
        "eig",                                                               \
        [](SimpleArray<T> const & a, bool balance)                           \
        {                                                                    \
            auto [w, v] = eig(a, balance);                                   \
            return pybind11::make_tuple(w, v);                               \
        },                                                                   \
        pybind11::arg("a"),                                                  \
        pybind11::arg("balance") = true,                                     \
        "Return (w, V): eigenvalues and unit right eigenvectors.");          \
    mod.def(                                                                 \
        "eigvals",                                                           \
        [](SimpleArray<T> const & a, bool balance)                           \
        { return eigvals(a, balance); },                                     \
        pybind11::arg("a"),                                                  \
        pybind11::arg("balance") = true,                                     \
        "Return the eigenvalues.");
    // clang-format on

    SC_DECL_EIGDEF(float)
    SC_DECL_EIGDEF(double)

#undef SC_DECL_EIGDEF
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
