/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/pymod/multibody_pymod.hpp> // Must be the first include.

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <solvcon/buffer/pymod/SimpleArrayCaster.hpp>
#include <solvcon/multibody/multibody.hpp>

namespace solvcon
{

namespace python
{

namespace
{

namespace py = pybind11;
namespace mb = solvcon::multibody;

mb::Vector3 to_vector3(std::vector<double> const & v, char const * name)
{
    if (v.size() != 3)
    {
        throw std::invalid_argument(std::string(name) + " must have 3 components, got " + std::to_string(v.size()));
    }
    return mb::Vector3(v[0], v[1], v[2]);
}

mb::Quaternion to_quaternion(std::vector<double> const & v, char const * name)
{
    if (v.size() != 4)
    {
        throw std::invalid_argument(std::string(name) + " must have 4 components, got " + std::to_string(v.size()));
    }
    return mb::Quaternion(v[0], v[1], v[2], v[3]);
}

template <typename V>
SimpleArray<double> to_array(V const & v)
{
    SimpleArray<double> out(small_vector<ssize_t>{static_cast<ssize_t>(v.size())});
    for (size_t k = 0; k < v.size(); ++k)
    {
        out[k] = v[k];
    }
    return out;
}

template <typename V>
std::vector<double> to_list(V const & v)
{
    return std::vector<double>(v.data(), v.data() + v.size());
}

SimpleArray<double> to_matrix6(std::vector<std::vector<double>> const & rows, char const * name)
{
    if (rows.size() != 6)
    {
        throw std::invalid_argument(std::string(name) + " must be 6x6");
    }
    SimpleArray<double> out(small_vector<ssize_t>{6, 6});
    for (ssize_t r = 0; r < 6; ++r)
    {
        if (rows[r].size() != 6)
        {
            throw std::invalid_argument(std::string(name) + " must be 6x6");
        }
        for (ssize_t c = 0; c < 6; ++c)
        {
            out(r, c) = rows[r][c];
        }
    }
    return out;
}

/**
 * Force element that calls a Python function (state-dependent loads).
 * fn(t, state_i, state_j) -> (force_i, torque_i, force_j, torque_j), global,
 * the forces at the center of mass and the torques about the center of mass;
 * state = {r, v, p, w} as lists.
 */
class PythonForce
    : public mb::ForceElement
{

public:

    PythonForce(mb::BodyPtr body_i, mb::BodyPtr body_j, py::function fn)
        : mb::ForceElement(std::move(body_i), std::move(body_j), mb::Vector3(), mb::Vector3(), 0.0, 0.0, 0.0, 0.0)
        , m_callback(std::move(fn))
    {
    }

    void compute() override
    {
        py::gil_scoped_acquire gil;
        auto state = [](mb::BodyPtr const & b)
        {
            py::dict d;
            d["r"] = to_list(b->position());
            d["v"] = to_list(b->velocity());
            d["p"] = to_list(b->quaternion());
            d["w"] = to_list(b->angular_velocity());
            return d;
        };
        py::tuple out = m_callback(m_time, state(m_body_i), state(m_body_j));
        if (out.size() != 4)
        {
            throw std::runtime_error("PythonForce: callback must return 4 vectors");
        }
        auto v3 = [](py::handle h)
        {
            auto v = h.cast<std::vector<double>>();
            if (v.size() != 3)
            {
                throw std::runtime_error("PythonForce: each vector needs 3 components");
            }
            return mb::Vector3(v[0], v[1], v[2]);
        };
        m_force_i = v3(out[0]);
        m_torque_i = v3(out[1]);
        m_force_j = v3(out[2]);
        m_torque_j = v3(out[3]);
    }

private:

    py::function m_callback;

}; /* end class PythonForce */

/// Wrapper of a class that Python only passes around.
template <typename T, typename Base = T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapHandle
    : public WrapBase<WrapHandle<T, Base>, T, std::shared_ptr<T>, Base>
{

    using base_type = WrapBase<WrapHandle<T, Base>, T, std::shared_ptr<T>, Base>;
    friend base_type;

    WrapHandle(py::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
    }

}; /* end class WrapHandle */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapTireForce
    : public WrapBase<WrapTireForce, mb::TireForce, std::shared_ptr<mb::TireForce>, mb::ForceElement>
{

    using base_type = WrapBase<WrapTireForce, mb::TireForce, std::shared_ptr<mb::TireForce>, mb::ForceElement>;
    using wrapped_type = typename base_type::wrapped_type;
    friend base_type;

    WrapTireForce(py::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        (*this)
            .def("get_vertical_force", &wrapped_type::vertical_force)
            .def("get_slip_angle", &wrapped_type::slip_angle)
            .def("get_contact_point", [](wrapped_type const & self)
                 { return to_array(self.point_j()); })
            //
            ;
    }

}; /* end class WrapTireForce */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapBushing
    : public WrapBase<WrapBushing, mb::Bushing, std::shared_ptr<mb::Bushing>, mb::ForceElement>
{

    using base_type = WrapBase<WrapBushing, mb::Bushing, std::shared_ptr<mb::Bushing>, mb::ForceElement>;
    using wrapped_type = typename base_type::wrapped_type;
    friend base_type;

    WrapBushing(py::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        (*this)
            .def(
                "get_deformation",
                [](wrapped_type const & self)
                {
                    mb::Vector3 const & d = self.translation();
                    mb::Vector3 const & r = self.rotation();
                    SimpleArray<double> out(small_vector<ssize_t>{6});
                    for (ssize_t k = 0; k < 3; ++k)
                    {
                        out[k] = d[static_cast<size_t>(k)];
                        out[k + 3] = r[static_cast<size_t>(k)];
                    }
                    return out;
                },
                "[dx, dy, dz, rx, ry, rz] in the bushing frame")
            //
            ;
    }

}; /* end class WrapBushing */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapCamProfile
    : public WrapBase<WrapCamProfile, mb::CamProfile>
{

    using base_type = WrapBase<WrapCamProfile, mb::CamProfile>;
    using wrapped_type = typename base_type::wrapped_type;
    friend base_type;

    WrapCamProfile(py::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        (*this)
            .def(py::init<double, std::vector<double>, std::vector<double>>(), py::arg("a0"), py::arg("a") = std::vector<double>(), py::arg("b") = std::vector<double>(), "R(phi) = a0 + sum_k a_k cos(k phi) + b_k sin(k phi)")
            //
            ;

        (*this)
            .def_static("fit", &wrapped_type::fit, py::arg("phi"), py::arg("r"), py::arg("n_harmonics"), "Least-squares Fourier fit of samples")
            .def(
                "eval",
                [](wrapped_type const & self, double phi)
                {
                    double r, dr, ddr;
                    self.eval(phi, r, dr, ddr);
                    return py::make_tuple(r, dr, ddr);
                },
                py::arg("phi"))
            //
            ;
    }

}; /* end class WrapCamProfile */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapMagicFormulaTire
    : public WrapBase<WrapMagicFormulaTire, mb::MagicFormulaTire>
{

    using base_type = WrapBase<WrapMagicFormulaTire, mb::MagicFormulaTire>;
    using wrapped_type = typename base_type::wrapped_type;
    friend base_type;

    static mb::MagicFormulaTire::Curve to_curve(py::dict const & d)
    {
        static char const * const keys[] = {"mu", "load_exponent", "stiffness", "shape", "curvature"};
        for (auto const & item : d)
        {
            std::string const key = py::str(item.first);
            bool known = false;
            for (char const * k : keys)
            {
                known = known || key == k;
            }
            if (!known)
            {
                throw py::key_error("MagicFormulaTire: unknown curve key '" + key + "'");
            }
        }
        mb::MagicFormulaTire::Curve c;
        auto get = [&](char const * key, double fallback)
        { return d.contains(key) ? d[key].cast<double>() : fallback; };
        c.mu = get("mu", c.mu);
        c.load_exponent = get("load_exponent", c.load_exponent);
        c.stiffness = get("stiffness", c.stiffness);
        c.shape = get("shape", c.shape);
        c.curvature = get("curvature", c.curvature);
        return c;
    }

    static py::dict to_dict(mb::MagicFormulaTire::Curve const & c)
    {
        py::dict d;
        d["mu"] = c.mu;
        d["load_exponent"] = c.load_exponent;
        d["stiffness"] = c.stiffness;
        d["shape"] = c.shape;
        d["curvature"] = c.curvature;
        return d;
    }

    WrapMagicFormulaTire(py::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        (*this)
            .def(
                py::init(
                    [](double fz0, py::dict const & longitudinal, py::dict const & lateral)
                    { return wrapped_type(fz0, to_curve(longitudinal), to_curve(lateral)); }),
                py::arg("fz0"),
                py::arg("longitudinal"),
                py::arg("lateral"),
                "fz0: the reference load (N); longitudinal and lateral: dicts of mu, load_exponent, "
                "stiffness (B), shape (C), curvature (E)")
            .def_property_readonly("fz0", &wrapped_type::fz0)
            .def_property_readonly("longitudinal", [](wrapped_type const & self)
                                   { return to_dict(self.longitudinal()); })
            .def_property_readonly("lateral", [](wrapped_type const & self)
                                   { return to_dict(self.lateral()); })
            .def(
                "force",
                [](wrapped_type const & self, double fz, double slip_ratio, double slip_angle)
                {
                    auto const f = self.force(fz, slip_ratio, slip_angle);
                    return py::make_tuple(f.fx, f.fy);
                },
                py::arg("fz"),
                py::arg("slip_ratio"),
                py::arg("slip_angle"),
                "(F_x, F_y) in N at the load, the slip ratio, and the slip angle (rad); F_y opposes + slip")
            .def(
                "force_for_demand",
                [](wrapped_type const & self, double fz, double demand, double slip_angle)
                {
                    auto const f = self.force_for_demand(fz, demand, slip_angle);
                    return py::make_tuple(f.fx, f.fy, f.slip_ratio);
                },
                py::arg("fz"),
                py::arg("demand"),
                py::arg("slip_angle"),
                "(F_x, F_y, kappa) that deliver the longitudinal force demand; beyond the peak the "
                "wheel locks (braking) or spins (driving)")
            .def(
                "friction",
                [](wrapped_type const & self, double fz)
                { return py::make_tuple(self.friction(self.longitudinal(), fz), self.friction(self.lateral(), fz)); },
                py::arg("fz"),
                "(mu_x, mu_y): the peak force over the load")
            .def(
                "cornering_stiffness",
                [](wrapped_type const & self, double fz)
                { return self.slip_stiffness(self.lateral(), fz); },
                py::arg("fz"),
                "dF_y/dalpha at zero slip (N/rad)")
            .def(
                "longitudinal_stiffness",
                [](wrapped_type const & self, double fz)
                { return self.slip_stiffness(self.longitudinal(), fz); },
                py::arg("fz"),
                "dF_x/dkappa at zero slip (N)")
            .def("__copy__", [](wrapped_type const & self)
                 { return wrapped_type(self); })
            .def("__deepcopy__", [](wrapped_type const & self, py::dict const &)
                 { return wrapped_type(self); })
            //
            ;
    }

}; /* end class WrapMagicFormulaTire */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapDriveFunction
    : public WrapBase<WrapDriveFunction, mb::DriveFunction>
{

    using base_type = WrapBase<WrapDriveFunction, mb::DriveFunction>;
    using wrapped_type = typename base_type::wrapped_type;
    friend base_type;

    WrapDriveFunction(py::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        (*this)
            .def_static("polynomial", &wrapped_type::polynomial, py::arg("coeffs"), "C(t) = sum_k coeffs[k] t^k")
            .def_static("sine", &wrapped_type::sine, py::arg("amplitude"), py::arg("omega"), py::arg("phase") = 0.0, py::arg("offset") = 0.0, "C(t) = offset + amplitude sin(omega t + phase)")
            .def_static("table", &wrapped_type::table, py::arg("times"), py::arg("values"), "Piecewise-linear table; times must increase strictly")
            .def_static("spline", &wrapped_type::spline, py::arg("times"), py::arg("values"), "Natural cubic spline; times must increase strictly")
            .def_static("parameter", &wrapped_type::parameter, py::arg("value"), "Constant set by the caller; C' = C'' = 0")
            .def(
                "eval",
                [](wrapped_type const & self, double t)
                {
                    double c, cd, cdd;
                    self.eval(t, c, cd, cdd);
                    return py::make_tuple(c, cd, cdd);
                },
                py::arg("t"),
                "Return (C, C', C'') at time t")
            .def("set_value", &wrapped_type::set_value, py::arg("value"))
            //
            ;
    }

}; /* end class WrapDriveFunction */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapModalResult
    : public WrapBase<WrapModalResult, mb::ModalResult>
{

    using base_type = WrapBase<WrapModalResult, mb::ModalResult>;
    using wrapped_type = typename base_type::wrapped_type;
    friend base_type;

    WrapModalResult(py::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        (*this)
            .def_readonly("n_dof", &wrapped_type::n_dof)
            .def_property_readonly("frequencies", [](wrapped_type const & self)
                                   { return self.frequencies; })
            .def_property_readonly("damping_ratios", [](wrapped_type const & self)
                                   { return self.damping_ratios; })
            .def_property_readonly("mode_shapes", [](wrapped_type const & self)
                                   { return self.mode_shapes; },
                                   "(n_modes, n_dof): row m is the shape of mode m")
            //
            ;
    }

}; /* end class WrapModalResult */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapDynamicsSystem
    : public WrapBase<WrapDynamicsSystem, mb::DynamicsSystem>
{

    using base_type = WrapBase<WrapDynamicsSystem, mb::DynamicsSystem>;
    using wrapped_type = typename base_type::wrapped_type;
    friend base_type;

    WrapDynamicsSystem(py::module & mod, char const * pyname, char const * pydoc);

    static py::dict sweep(wrapped_type & self, py::object const & driver, py::object const & values, py::object const & rates);
    static py::list joint_reactions(wrapped_type const & self);

}; /* end class WrapDynamicsSystem */

WrapDynamicsSystem::WrapDynamicsSystem(py::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    (*this)
        .def(py::init<double>())
        //
        ;

    (*this)
        .def("get_nbody", &wrapped_type::nbody)
        .def("get_njoint", &wrapped_type::njoint)
        .def("Assembly", &wrapped_type::assemble)
        .def("init", &wrapped_type::init)
        .def("solve", &wrapped_type::solve, py::call_guard<py::gil_scoped_release>(), "Advance one output step. Other Python threads run during the call; two threads must not use one system at the same time.")
        .def("get_body_positions", &wrapped_type::body_positions)
        .def("get_body_angles", &wrapped_type::body_angles)
        .def("get_body_accelerations", &wrapped_type::body_accelerations)
        .def("get_body_angular_accelerations", &wrapped_type::body_angular_accelerations)
        .def("get_body_quaternions", &wrapped_type::body_quaternions)
        .def("get_body_velocities", &wrapped_type::body_velocities)
        .def("get_body_angular_velocities", &wrapped_type::body_angular_velocities)
        .def("get_sim_time", &wrapped_type::sim_time)
        .def("get_dof", &wrapped_type::dof, "Rank-based DOF from the last init()")
        .def("set_integrator", &wrapped_type::set_integrator, py::arg("name"), "rk4 (default), rk45, abm4, partition, hybrid, newmark, hht, houbolt")
        .def("get_integrator", &wrapped_type::integrator)
        .def("get_last_newton_iterations", &wrapped_type::last_newton_iterations)
        .def("get_last_substeps", &wrapped_type::last_substeps)
        .def("get_last_rejected", &wrapped_type::last_rejected)
        .def("set_tolerances", &wrapped_type::set_tolerances, py::arg("rtol"), py::arg("atol"), "Error tolerances of the rk45 integrator")
        .def("set_hht_alpha", &wrapped_type::set_hht_alpha, py::arg("alpha"), "HHT-alpha numerical damping, -1/3 <= alpha <= 0")
        .def("get_joint_reactions", &joint_reactions, "Per joint: global reaction force on each body and torque about the joint point; drivers also report 'effort' (actuator force or torque)")
        .def(
            "find_equilibrium",
            [](wrapped_type & self, double tol, int32_t max_iter)
            {
                mb::EquilibriumResult r;
                {
                    py::gil_scoped_release const release;
                    r = self.find_equilibrium(tol, max_iter);
                }
                py::dict d;
                d["converged"] = r.converged;
                d["iterations"] = r.iterations;
                d["residual"] = r.residual;
                d["message"] = r.message;
                return d;
            },
            py::arg("tol") = 1e-9,
            py::arg("max_iter") = 50,
            "Static equilibrium by Newton iteration; the state keeps the start pose if it does not converge")
        .def("get_joints", [](wrapped_type const & self)
             { return self.joints(); })
        .def(
            "get_diagnostics",
            [](wrapped_type const & self)
            {
                py::list out;
                for (auto const & d : self.diagnostics())
                {
                    py::dict e;
                    e["level"] = d.level;
                    e["code"] = d.code;
                    e["joint"] = d.joint;
                    e["row"] = d.row;
                    e["message"] = d.message;
                    out.append(e);
                }
                return out;
            },
            "Solver messages: list of {level, code, joint, row, message}")
        .def("sweep", &sweep, py::arg("driver"), py::arg("values"), py::arg("rates") = py::none(), "Quasi-static sweep over a parameter driver. Stops at the first value that does not assemble and reports it in 'message'.")
        .def(
            "evaluate_constraints",
            [](wrapped_type & self, double t)
            {
                mb::ConstraintEval ev = self.evaluate_constraints(t);
                py::dict d;
                d["C"] = py::cast(std::move(ev.c));
                d["Cq"] = py::cast(std::move(ev.cq));
                d["Cq_pos"] = py::cast(std::move(ev.cq_pos));
                d["Ct"] = py::cast(std::move(ev.ct));
                d["gamma"] = py::cast(std::move(ev.gamma));
                return d;
            },
            py::arg("t"),
            "Constraint values, Jacobians (velocity and position form), Phi_t, and gamma at the current body states and time t")
        .def(
            "set_body_state",
            [](wrapped_type & self, uint32_t i, std::vector<double> const & r, std::vector<double> const & v, std::vector<double> const & p, std::vector<double> const & pdot)
            {
                if (p.size() != 4 || pdot.size() != 4)
                {
                    throw std::invalid_argument("p and pdot must have 4 components");
                }
                self.set_body_state(i, to_vector3(r, "r"), to_vector3(v, "v"), to_quaternion(p, "p"), to_quaternion(pdot, "pdot"));
            },
            py::arg("i"),
            py::arg("r"),
            py::arg("v"),
            py::arg("p"),
            py::arg("pdot"))
        .def(
            "set_applied_load",
            [](wrapped_type & self, uint32_t i, std::vector<double> const & f, std::vector<double> const & t)
            { self.set_applied_load(i, to_vector3(f, "force"), to_vector3(t, "torque")); },
            py::arg("i"),
            py::arg("force"),
            py::arg("torque"),
            "Applied load of body i: global force at the COM and global torque")
        .def("AddBody", [](wrapped_type & self, mb::BodyPtr body)
             { self.add(std::move(body)); })
        .def("AddJoint", [](wrapped_type & self, mb::JointPtr joint)
             { self.add(std::move(joint)); })
        .def("AddForceElement", [](wrapped_type & self, mb::ForceElementPtr fe)
             { self.add(std::move(fe)); })
        .def("Add", [](wrapped_type & self, mb::BodyPtr body)
             { self.add(std::move(body)); })
        .def("Add", [](wrapped_type & self, mb::JointPtr joint)
             { self.add(std::move(joint)); })
        .def("Add", [](wrapped_type & self, mb::ForceElementPtr fe)
             { self.add(std::move(fe)); })
        .def("Add", [](wrapped_type & self, mb::GearCouplerPtr gc)
             { self.add(std::move(gc)); })
        //
        ;
}

py::dict WrapDynamicsSystem::sweep(wrapped_type & self, py::object const & driver, py::object const & values, py::object const & rates)
{
    mb::SweepResult r;
    if (py::isinstance<py::list>(driver) || py::isinstance<py::tuple>(driver))
    {
        auto drivers = driver.cast<std::vector<mb::JointPtr>>();
        auto rows = values.cast<std::vector<std::vector<double>>>();
        std::vector<double> rr = rates.is_none() ? std::vector<double>(drivers.size(), 1.0) : rates.cast<std::vector<double>>();
        r = self.sweep_group(drivers, rows, rr);
    }
    else
    {
        r = self.sweep(driver.cast<mb::JointPtr>(), values.cast<std::vector<double>>());
    }
    py::dict d;
    d["values"] = py::cast(std::move(r.values));
    d["positions"] = py::cast(std::move(r.positions));
    d["quaternions"] = py::cast(std::move(r.quaternions));
    d["dr_ds"] = py::cast(std::move(r.dr_ds));
    d["domega_ds"] = py::cast(std::move(r.domega_ds));
    d["iterations"] = py::cast(std::move(r.iterations));
    d["residual"] = py::cast(std::move(r.residual));
    d["rcond"] = py::cast(std::move(r.rcond));
    d["driver_joints"] = py::cast(std::move(r.driver_joints));
    d["efforts"] = py::cast(std::move(r.efforts));
    d["completed"] = r.completed;
    d["message"] = r.message;
    return d;
}

py::list WrapDynamicsSystem::joint_reactions(wrapped_type const & self)
{
    py::list out;
    for (auto & r : self.joint_reactions())
    {
        py::dict d;
        d["body_i"] = r.body_i;
        d["body_j"] = r.body_j;
        d["point_i"] = to_array(r.point_i);
        d["point_j"] = to_array(r.point_j);
        d["force_i"] = to_array(r.force_i);
        d["torque_i"] = to_array(r.torque_i);
        d["force_j"] = to_array(r.force_j);
        d["torque_j"] = to_array(r.torque_j);
        d["lambda"] = py::cast(std::move(r.lambda));
        if (r.is_driver)
        {
            d["effort"] = r.effort;
        }
        out.append(d);
    }
    return out;
}

void wrap_factories(py::module & mod)
{
    mod.def(
        "create_ground",
        [](uint32_t num) -> mb::BodyPtr
        { return std::make_shared<mb::Ground>(num); },
        "Create a Ground body");

    mod.def(
        "create_mobilized_body",
        [](uint32_t num, std::vector<double> const & pos, std::vector<double> const & vel, std::vector<double> const & acc, std::vector<double> const & ang, std::vector<double> const & ang_vel, std::vector<double> const & ang_acc, double mass, std::vector<double> const & inertia, std::vector<double> const & force, std::vector<double> const & torque) -> mb::BodyPtr
        {
            return std::make_shared<mb::MobilizedBody>(num, to_vector3(pos, "pos"), to_vector3(vel, "vel"), to_vector3(acc, "acc"), to_vector3(ang, "ang"), to_vector3(ang_vel, "ang_vel"), to_vector3(ang_acc, "ang_acc"), mass, mb::MobilizedBody::inertia_tensor(inertia), to_vector3(force, "force"), to_vector3(torque, "torque"));
        },
        "Create a Mobilized body");

    mod.def(
        "create_mobilized_body_quat",
        [](uint32_t num, std::vector<double> const & pos, std::vector<double> const & vel, std::vector<double> const & acc, std::vector<double> const & quat, std::vector<double> const & ang_vel, std::vector<double> const & ang_acc, double mass, std::vector<double> const & inertia, std::vector<double> const & force, std::vector<double> const & torque) -> mb::BodyPtr
        {
            mb::Vector3 const omega = to_vector3(ang_vel, "ang_vel");
            mb::Quaternion const q = to_quaternion(quat, "quat");
            auto body = std::make_shared<mb::MobilizedBody>(num, to_vector3(pos, "pos"), to_vector3(vel, "vel"), to_vector3(acc, "acc"), mb::Vector3(), omega, to_vector3(ang_acc, "ang_acc"), mass, mb::MobilizedBody::inertia_tensor(inertia), to_vector3(force, "force"), to_vector3(torque, "torque"));
            body->set_tbi(mb::tbi_from_quaternion(q));
            // p_dot from the global angular velocity at the given quaternion.
            body->set_p_dot(0.5 * mb::transpose_mul(mb::build_E(q), omega));
            return body;
        },
        "Create a Mobilized body with quaternion [e0,e1,e2,e3] orientation");

    auto joint = [](uint32_t type, std::vector<double> const & pi, std::vector<double> const & pj, std::vector<double> const & qi, std::vector<double> const & qj, std::vector<double> const & ai, std::vector<double> const & aj, mb::BodyPtr body_i, mb::BodyPtr body_j, double drive_omega, double distance)
    {
        return std::make_shared<mb::Joint>(type, to_vector3(pi, "pi"), to_vector3(pj, "pj"), to_vector3(qi, "qi"), to_vector3(qj, "qj"), to_vector3(ai, "ai"), to_vector3(aj, "aj"), std::move(body_i), std::move(body_j), drive_omega, distance);
    };
    mod.def(
        "create_joint",
        [joint](uint32_t type, std::vector<double> const & pi, std::vector<double> const & pj, std::vector<double> const & qi, std::vector<double> const & qj, std::vector<double> const & ai, std::vector<double> const & aj, mb::BodyPtr body_i, mb::BodyPtr body_j) -> mb::JointPtr
        { return joint(type, pi, pj, qi, qj, ai, aj, std::move(body_i), std::move(body_j), 0.0, 0.0); },
        "Create a Joint (type: 0=spherical, 1=revolute, 2=prismatic, 3=driving)");
    mod.def(
        "create_joint_driving",
        [joint](uint32_t type, std::vector<double> const & pi, std::vector<double> const & pj, std::vector<double> const & qi, std::vector<double> const & qj, std::vector<double> const & ai, std::vector<double> const & aj, mb::BodyPtr body_i, mb::BodyPtr body_j, double drive_omega) -> mb::JointPtr
        { return joint(type, pi, pj, qi, qj, ai, aj, std::move(body_i), std::move(body_j), drive_omega, 0.0); },
        "Create a driving Joint with angular velocity");
    mod.def(
        "create_joint_distance",
        [joint](uint32_t type, std::vector<double> const & pi, std::vector<double> const & pj, std::vector<double> const & qi, std::vector<double> const & qj, std::vector<double> const & ai, std::vector<double> const & aj, mb::BodyPtr body_i, mb::BodyPtr body_j, double distance) -> mb::JointPtr
        { return joint(type, pi, pj, qi, qj, ai, aj, std::move(body_i), std::move(body_j), 0.0, distance); },
        "Create a distance constraint Joint (type 6)");

    // Driving constraints (Haug Sec. 9.5).  Each returns a Joint with one row.
    mod.def(
        "create_absolute_driver",
        [](mb::BodyPtr body, mb::BodyPtr ground, std::vector<double> const & point, std::vector<double> const & direction, mb::DriveFunction const & func) -> mb::JointPtr
        {
            mb::Vector3 const z;
            mb::Vector3 const u = to_vector3(direction, "direction");
            return std::make_shared<mb::Joint>(8, to_vector3(point, "point"), z, z, z, u, u, std::move(body), std::move(ground), 0.0, 0.0, func);
        },
        py::arg("body"),
        py::arg("ground"),
        py::arg("point"),
        py::arg("direction"),
        py::arg("func"),
        "Absolute driver (Eq. 9.5.1): direction . r^P - C(t) = 0, direction is global");
    mod.def(
        "create_distance_driver",
        [](mb::BodyPtr body_i, mb::BodyPtr body_j, std::vector<double> const & point_i, std::vector<double> const & point_j, mb::DriveFunction const & func) -> mb::JointPtr
        {
            mb::Vector3 const z;
            mb::Vector3 const ax(0.0, 0.0, 1.0);
            return std::make_shared<mb::Joint>(9, to_vector3(point_i, "point_i"), to_vector3(point_j, "point_j"), z, z, ax, ax, std::move(body_i), std::move(body_j), 0.0, 0.0, func);
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("func"),
        "Distance driver (Eq. 9.5.2): |P_i P_j| = C(t), C != 0");
    mod.def(
        "create_translational_driver",
        [](mb::BodyPtr body_i, mb::BodyPtr body_j, std::vector<double> const & point_i, std::vector<double> const & point_j, std::vector<double> const & axis_i, mb::DriveFunction const & func) -> mb::JointPtr
        {
            mb::Vector3 const z;
            mb::Vector3 const h = to_vector3(axis_i, "axis_i");
            return std::make_shared<mb::Joint>(10, to_vector3(point_i, "point_i"), to_vector3(point_j, "point_j"), z, z, h, h, std::move(body_i), std::move(body_j), 0.0, 0.0, func);
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("axis_i"),
        py::arg("func"),
        "Relative translational driver (Eq. 9.5.3): h_i . (r_j^P - r_i^P) = C(t)");
    mod.def(
        "create_rotational_driver",
        [](mb::BodyPtr body_i, mb::BodyPtr body_j, std::vector<double> const & axis_i, std::vector<double> const & axis_j, mb::DriveFunction const & func) -> mb::JointPtr
        {
            mb::Vector3 const z;
            return std::make_shared<mb::Joint>(11, z, z, z, z, to_vector3(axis_i, "axis_i"), to_vector3(axis_j, "axis_j"), std::move(body_i), std::move(body_j), 0.0, 0.0, func);
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("axis_i"),
        py::arg("axis_j"),
        py::arg("func"),
        "Relative rotational driver (Eq. 9.5.4): angle from f_i to f_j about h_i = C(t)");

    // Composite joints (Haug Sec. 9.4.4-9.4.5).  distance <= 0: take it from the pose.
    auto composite = [](uint32_t type, mb::BodyPtr bi, mb::BodyPtr bj, std::vector<double> const & p_i, std::vector<double> const & p_j, std::vector<double> const & a_i, std::vector<double> const & a_j, std::vector<double> const & f_j, double param, double distance)
    {
        return std::make_shared<mb::Joint>(type, to_vector3(p_i, "point_i"), to_vector3(p_j, "point_j"), mb::Vector3(), to_vector3(f_j, "f_j"), to_vector3(a_i, "axis_i"), to_vector3(a_j, "axis_j"), std::move(bi), std::move(bj), param, distance);
    };
    std::vector<double> const ez = {0.0, 0.0, 1.0};
    std::vector<double> const zero3 = {0.0, 0.0, 0.0};
    mod.def(
        "create_screw_joint",
        [=](mb::BodyPtr bi, mb::BodyPtr bj, std::vector<double> const & p_i, std::vector<double> const & p_j, std::vector<double> const & a_i, std::vector<double> const & a_j, double pitch)
        { return composite(12, std::move(bi), std::move(bj), p_i, p_j, a_i, a_j, zero3, pitch, 0.0); },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("axis_i"),
        py::arg("axis_j"),
        py::arg("pitch"),
        "Screw joint (Eq. 9.4.26): translation along h_i = pitch * rotation angle [m/rad]");
    mod.def(
        "create_revolute_spherical_joint",
        [=](mb::BodyPtr bi, mb::BodyPtr bj, std::vector<double> const & p_i, std::vector<double> const & p_j, std::vector<double> const & a_i, double distance)
        { return composite(13, std::move(bi), std::move(bj), p_i, p_j, a_i, ez, zero3, 0.0, distance); },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("axis_i"),
        py::arg("distance") = 0.0,
        "Revolute-spherical composite joint (Eq. 9.4.29), 2 constraints");
    mod.def(
        "create_revolute_revolute_joint",
        [=](mb::BodyPtr bi, mb::BodyPtr bj, std::vector<double> const & p_i, std::vector<double> const & p_j, std::vector<double> const & a_i, std::vector<double> const & a_j, bool orthogonal, double distance)
        { return composite(orthogonal ? 15 : 14, std::move(bi), std::move(bj), p_i, p_j, a_i, a_j, zero3, 0.0, distance); },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("axis_i"),
        py::arg("axis_j"),
        py::arg("orthogonal") = false,
        py::arg("distance") = 0.0,
        "Revolute-revolute composite joint, parallel (Eq. 9.4.30) or orthogonal "
        "intersecting (Eq. 9.4.31) axes, 4 constraints");
    mod.def(
        "create_revolute_translational_joint",
        [=](mb::BodyPtr bi, mb::BodyPtr bj, std::vector<double> const & p_i, std::vector<double> const & p_j, std::vector<double> const & a_i, std::vector<double> const & a_j, std::vector<double> const & f_j)
        { return composite(16, std::move(bi), std::move(bj), p_i, p_j, a_i, a_j, f_j, 0.0, 0.0); },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("axis_i"),
        py::arg("axis_j"),
        py::arg("f_j"),
        "Revolute-translational composite joint (Eq. 9.4.33): h_i parallel to f_j, "
        "translation along h_j (f_j perpendicular to h_j), 4 constraints");
    mod.def(
        "create_strut_joint",
        [=](mb::BodyPtr bi, mb::BodyPtr bj, std::vector<double> const & p_i, std::vector<double> const & p_j, std::vector<double> const & a_i)
        { return composite(17, std::move(bi), std::move(bj), p_i, p_j, a_i, ez, zero3, 0.0, 0.0); },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("axis_i"),
        "Strut composite joint (Eq. 9.4.34): P_j on the line (P_i, h_i)");

    mod.def(
        "create_gear_coupler",
        [](mb::JointPtr joint_a, std::string const & kind_a, mb::JointPtr joint_b, std::string const & kind_b, double ratio)
        {
            auto kind = [](std::string const & k)
            {
                if (k == "rotation")
                {
                    return 0;
                }
                if (k == "translation")
                {
                    return 1;
                }
                throw std::invalid_argument("kind must be 'rotation' or 'translation', got '" + k + "'");
            };
            return std::make_shared<mb::GearCoupler>(std::move(joint_a), kind(kind_a), std::move(joint_b), kind(kind_b), ratio);
        },
        py::arg("joint_a"),
        py::arg("kind_a"),
        py::arg("joint_b"),
        py::arg("kind_b"),
        py::arg("ratio"),
        "Gear pair / rack and pinion (Haug 3.4.1): (g_A - g_A0) = ratio (g_B - g_B0). Both "
        "joints must also be added to the system.");
    mod.def(
        "create_point_follower",
        [](mb::BodyPtr cam, mb::BodyPtr follower, std::vector<double> const & center, std::vector<double> const & normal, std::vector<double> const & point, mb::CamProfile const & profile)
        {
            mb::Vector3 const z;
            mb::Vector3 const n = to_vector3(normal, "normal");
            return std::make_shared<mb::Joint>(18, to_vector3(center, "center"), to_vector3(point, "point"), z, z, n, n, std::move(cam), std::move(follower), 0.0, 0.0, mb::DriveFunction(), profile);
        },
        py::arg("cam"),
        py::arg("follower"),
        py::arg("center"),
        py::arg("normal"),
        py::arg("point"),
        py::arg("profile"),
        "Point (or roller, using the pitch curve) follower on a polar cam profile R(phi), "
        "phi measured from f to g of the cam plane (Haug 3.4.3)");
    mod.def(
        "create_flat_follower",
        [](mb::BodyPtr cam, mb::BodyPtr follower, std::vector<double> const & center, std::vector<double> const & normal, std::vector<double> const & face_point, std::vector<double> const & face_normal, mb::CamProfile const & support)
        {
            mb::Vector3 const z;
            return std::make_shared<mb::Joint>(19, to_vector3(center, "center"), to_vector3(face_point, "face_point"), z, z, to_vector3(normal, "normal"), to_vector3(face_normal, "face_normal"), std::move(cam), std::move(follower), 0.0, 0.0, mb::DriveFunction(), support);
        },
        py::arg("cam"),
        py::arg("follower"),
        py::arg("center"),
        py::arg("normal"),
        py::arg("face_point"),
        py::arg("face_normal"),
        py::arg("support"),
        "Flat-face follower on a convex cam given by its support function h(psi); "
        "face_normal points from the face toward the cam (Haug 3.4.2)");

    mod.def(
        "create_spring_damper",
        [](mb::BodyPtr body_i, mb::BodyPtr body_j, std::vector<double> const & pi, std::vector<double> const & pj, double k, double c, double free_length, double force) -> mb::ForceElementPtr
        {
            return std::make_shared<mb::ForceElement>(std::move(body_i), std::move(body_j), to_vector3(pi, "point_i"), to_vector3(pj, "point_j"), k, c, free_length, force);
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("k"),
        py::arg("c"),
        py::arg("free_length"),
        py::arg("force") = 0.0,
        "Translational spring-damper-actuator (Haug Sec. 11.4.1)");
    mod.def(
        "create_bushing",
        [](mb::BodyPtr bi, mb::BodyPtr bj, std::vector<double> const & p_i, std::vector<double> const & p_j, std::vector<std::vector<double>> const & stiffness, std::vector<std::vector<double>> const & damping, std::vector<double> const & frame_quat) -> mb::ForceElementPtr
        {
            if (frame_quat.size() != 4)
            {
                throw std::invalid_argument("frame_quat must have 4 components");
            }
            mb::Quaternion qf(frame_quat[0], frame_quat[1], frame_quat[2], frame_quat[3]);
            qf /= mb::norm(qf);
            mb::Matrix3 const tbi = mb::tbi_from_quaternion(qf);
            return std::make_shared<mb::Bushing>(std::move(bi), std::move(bj), to_vector3(p_i, "point_i"), to_vector3(p_j, "point_j"), to_matrix6(stiffness, "K"), to_matrix6(damping, "C"), tbi.transpose());
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("K"),
        py::arg("C"),
        py::arg("frame_quat") = std::vector<double>{1, 0, 0, 0},
        "Linear 6x6 bushing / massless beam (unloaded at the construction pose). "
        "K, C act on [dx, dy, dz, rx, ry, rz] in the bushing frame on body i.");
    mod.def(
        "create_length_stop",
        [](mb::BodyPtr body_i, mb::BodyPtr body_j, std::vector<double> const & pi, std::vector<double> const & pj, py::object const & min_length, py::object const & max_length, double k, double k2, double c) -> mb::ForceElementPtr
        {
            double const inf = std::numeric_limits<double>::infinity();
            double const lo = min_length.is_none() ? -inf : min_length.cast<double>();
            double const hi = max_length.is_none() ? inf : max_length.cast<double>();
            return std::make_shared<mb::LengthStop>(std::move(body_i), std::move(body_j), to_vector3(pi, "point_i"), to_vector3(pj, "point_j"), lo, hi, k, k2, c);
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("min_length") = py::none(),
        py::arg("max_length") = py::none(),
        py::arg("k"),
        py::arg("k2") = 0.0,
        py::arg("c") = 0.0,
        "One-sided travel stop on the line between two points: below min_length (bump stop) "
        "it pushes them apart, above max_length (droop stop) it pulls them together, with "
        "f = max(0, k d + k2 d^2 + c dd/dt) for the penetration d. None disables a limit.");
    mod.def(
        "create_curve_damper",
        [](mb::BodyPtr body_i, mb::BodyPtr body_j, std::vector<double> const & pi, std::vector<double> const & pj, double k, double free_length, std::vector<double> velocities, std::vector<double> forces) -> mb::ForceElementPtr
        {
            try
            {
                return std::make_shared<mb::CurveDamper>(std::move(body_i), std::move(body_j), to_vector3(pi, "point_i"), to_vector3(pj, "point_j"), k, free_length, std::move(velocities), std::move(forces));
            }
            catch (std::invalid_argument const & e)
            {
                throw py::value_error(e.what());
            }
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("point_i"),
        py::arg("point_j"),
        py::arg("k"),
        py::arg("free_length"),
        py::arg("velocities"),
        py::arg("forces"),
        "Spring-damper with a damper force-velocity curve: tension f = k (L - L0) + F(dL/dt), "
        "F piecewise linear through (velocities, forces), extended with the end slopes; "
        "dL/dt > 0 is rebound (extension), F > 0 tension.");
    mod.def(
        "get_curve_damper_state",
        [](mb::ForceElementPtr const & fe)
        {
            auto damper = std::dynamic_pointer_cast<mb::CurveDamper>(fe);
            if (!damper)
            {
                throw py::type_error("get_curve_damper_state: not a curve damper");
            }
            py::dict d;
            d["length"] = damper->length();
            d["velocity"] = damper->length_rate();
            d["damper_force"] = damper->last_damper_force();
            return d;
        },
        py::arg("damper"),
        "Length (m), rate dL/dt (m/s), and damper force (N, tension positive) of a curve "
        "damper at the last evaluation.");
    mod.def(
        "curve_damper_force",
        [](mb::ForceElementPtr const & fe, double velocity)
        {
            auto damper = std::dynamic_pointer_cast<mb::CurveDamper>(fe);
            if (!damper)
            {
                throw py::type_error("curve_damper_force: not a curve damper");
            }
            return damper->damper_force(velocity);
        },
        py::arg("damper"),
        py::arg("velocity"),
        "The damper force F(v) (N) of a curve damper at the rate v (m/s).");
    mod.def(
        "get_length_stop_state",
        [](mb::ForceElementPtr const & fe)
        {
            auto stop = std::dynamic_pointer_cast<mb::LengthStop>(fe);
            if (!stop)
            {
                throw py::type_error("get_length_stop_state: not a length stop");
            }
            py::dict d;
            d["length"] = stop->length();
            d["penetration"] = stop->penetration();
            d["force"] = stop->stop_force();
            d["engaged"] = stop->engaged();
            d["min_length"] = stop->min_length();
            d["max_length"] = stop->max_length();
            return d;
        },
        py::arg("stop"),
        "Length (m), penetration (m), force (N), and engaged limit (-1 bump, +1 droop, 0 "
        "none) of a length stop at the last evaluation.");
    mod.def(
        "get_force_element_loads",
        [](mb::ForceElementPtr const & fe)
        {
            py::dict d;
            d["force_i"] = to_array(fe->force_i());
            d["torque_i"] = to_array(fe->torque_i());
            d["force_j"] = to_array(fe->force_j());
            d["torque_j"] = to_array(fe->torque_j());
            d["point_i"] = to_array(fe->point_i());
            d["point_j"] = to_array(fe->point_j());
            return d;
        },
        py::arg("element"),
        "Last computed loads of a force element: global forces at the COM, torques about "
        "the COM, and the global attachment points (for UI force arrows)");
    mod.def(
        "create_python_force",
        [](mb::BodyPtr bi, mb::BodyPtr bj, py::function fn) -> mb::ForceElementPtr
        { return std::make_shared<PythonForce>(std::move(bi), std::move(bj), std::move(fn)); },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("fn"),
        "Force element from a Python function fn(t, state_i, state_j) -> (F_i, T_i, F_j, T_j), "
        "global forces at the COM and torques about the COM; state = {r, v, p, w}");
    mod.def(
        "create_tire",
        [](mb::BodyPtr road, mb::BodyPtr wheel, std::vector<double> const & spin_axis, double radius, double k, double c, double c_alpha, double mu, double road_z, py::object const & road_height, py::object const & road_slope, py::object const & magic_formula, double relaxation_length) -> mb::ForceElementPtr
        {
            auto tire = std::make_shared<mb::TireForce>(std::move(road), std::move(wheel), to_vector3(spin_axis, "spin_axis"), radius, k, c, c_alpha, mu, road_z);
            if (!magic_formula.is_none())
            {
                tire->set_magic_formula(magic_formula.cast<mb::MagicFormulaTire>());
            }
            tire->set_relaxation_length(relaxation_length);
            if (!road_height.is_none() || !road_slope.is_none())
            {
                mb::DriveFunction h = road_height.is_none() ? mb::DriveFunction::parameter(road_z) : road_height.cast<mb::DriveFunction>();
                mb::DriveFunction b = road_slope.is_none() ? mb::DriveFunction::parameter(0.0) : road_slope.cast<mb::DriveFunction>();
                tire->set_road(std::move(h), std::move(b));
            }
            return tire;
        },
        py::arg("road"),
        py::arg("wheel"),
        py::arg("spin_axis"),
        py::arg("radius"),
        py::arg("k"),
        py::arg("c") = 0.0,
        py::arg("c_alpha") = 0.0,
        py::arg("mu") = 1.0,
        py::arg("road_z") = 0.0,
        py::arg("road_height") = py::none(),
        py::arg("road_slope") = py::none(),
        py::arg("magic_formula") = py::none(),
        py::arg("relaxation_length") = 0.0,
        "Point-contact tire (Haug 12.5): spring-damper along the road normal without tension "
        "(the tire leaves the road), slip-angle lateral force -c_alpha*alpha capped by mu*Fz. "
        "Options: magic_formula (a MagicFormulaTire) replaces the linear-saturated side force "
        "and turns the longitudinal demand into a slip ratio; relaxation_length > 0 lags the slip angle over the rolled distance (call "
        "commit_tire after each step). "
        "The road is the plane z = road_z, or, with road_height h(t) and road_slope b(t) "
        "(DriveFunctions), the effective plane through (x_w, y_w, h) with the normal "
        "(-sin b, 0, cos b). radius is the free radius; the wheel body origin is the wheel "
        "center.");
    mod.def(
        "get_tire_state",
        [](mb::ForceElementPtr const & fe)
        {
            auto tire = std::dynamic_pointer_cast<mb::TireForce>(fe);
            if (!tire)
            {
                throw py::type_error("get_tire_state: not a tire");
            }
            py::dict d;
            d["normal_force"] = tire->vertical_force();
            d["deflection"] = tire->deflection();
            d["in_contact"] = tire->in_contact();
            d["slip_angle"] = tire->slip_angle();
            d["force_slip_angle"] = tire->force_slip_angle();
            d["lateral_force"] = tire->lateral_force();
            d["longitudinal_force"] = tire->longitudinal_force();
            mb::Vector3 const p = tire->point_j();
            d["contact"] = std::vector<double>{p[0], p[1], p[2]};
            return d;
        },
        py::arg("tire"),
        "Normal force (N), normal deflection (m), contact flag, slip angle (rad), and contact "
        "point of a tire at the last evaluation.");
    mod.def(
        "commit_tire",
        [](mb::ForceElementPtr const & fe)
        {
            auto tire = std::dynamic_pointer_cast<mb::TireForce>(fe);
            if (!tire)
            {
                throw py::type_error("commit_tire: not a tire");
            }
            tire->commit();
        },
        py::arg("tire"),
        "Store the lagged slip angle of a tire at the end of an accepted step (relaxation).");
    mod.def(
        "set_tire_longitudinal",
        [](mb::ForceElementPtr const & fe, double ratio)
        {
            auto tire = std::dynamic_pointer_cast<mb::TireForce>(fe);
            if (!tire)
            {
                throw py::type_error("set_tire_longitudinal: not a tire");
            }
            tire->set_longitudinal(ratio);
        },
        py::arg("tire"),
        py::arg("ratio"),
        "Longitudinal force demand F_x/F_z of a tire (+ drive, - brake). With a Magic Formula "
        "the demand sets the slip ratio (combined slip); otherwise it is capped at mu and the "
        "side force keeps the rest of the friction circle.");
    mod.def(
        "create_rsda",
        [](mb::BodyPtr body_i, mb::BodyPtr body_j, std::vector<double> const & axis_i, std::vector<double> const & axis_j, double k, double c, py::object const & torque) -> mb::ForceElementPtr
        {
            mb::DriveFunction t = torque.is_none() ? mb::DriveFunction::polynomial({0.0}) : torque.cast<mb::DriveFunction>();
            return std::make_shared<mb::RotationalSDA>(std::move(body_i), std::move(body_j), to_vector3(axis_i, "axis_i"), to_vector3(axis_j, "axis_j"), k, c, std::move(t));
        },
        py::arg("body_i"),
        py::arg("body_j"),
        py::arg("axis_i"),
        py::arg("axis_j"),
        py::arg("k"),
        py::arg("c") = 0.0,
        py::arg("torque") = py::none(),
        "Rotational spring-damper-actuator (Haug Sec. 11.4.2). The spring is free at the "
        "construction pose; torque(t) drives body j counterclockwise about axis_i.");

    mod.def(
        "modal_analysis",
        [](mb::DynamicsSystem & sys)
        { return mb::modal_analysis(sys); },
        py::call_guard<py::gil_scoped_release>(),
        "Run modal analysis on the current system state");
}

} /* end namespace */

void wrap_multibody(pybind11::module & mod)
{
    namespace py = pybind11;
    mod.attr("HAS_MULTIBODY") = true;
    mod.attr("FEATURE_ROAD_TIRE") = true;
    mod.attr("FEATURE_LENGTH_STOP") = true;
    mod.attr("FEATURE_CURVE_DAMPER") = true;

    // Identity round trips kept for scripts written for the standalone module.
    mod.def("vec_to_list", [](std::vector<double> const & v)
            { return v; },
            "Round-trip a list through the engine vector type");
    mod.def("list_to_vec", [](std::vector<double> const & v)
            { return v; },
            "Round-trip a list through the engine vector type");

    WrapHandle<mb::Body>::commit(mod, "Body", "Rigid body");
    WrapHandle<mb::Ground, mb::Body>::commit(mod, "Ground", "The fixed body");
    WrapHandle<mb::MobilizedBody, mb::Body>::commit(mod, "MobilizedBody", "Moving rigid body");
    WrapHandle<mb::Joint>::commit(mod, "Joint", "Joint, driver, or cam follower");
    WrapHandle<mb::ForceElement>::commit(mod, "ForceElement", "Force element between two bodies");
    WrapHandle<mb::RotationalSDA, mb::ForceElement>::commit(mod, "RotationalSDA", "Rotational spring-damper-actuator");
    WrapHandle<mb::GearCoupler>::commit(mod, "GearCoupler", "Gear pair or rack and pinion");
    WrapHandle<PythonForce, mb::ForceElement>::commit(mod, "PythonForce", "Force element from a Python function");
    WrapTireForce::commit(mod, "TireForce", "Point-contact tire");
    WrapBushing::commit(mod, "Bushing", "Linear 6x6 bushing");
    WrapCamProfile::commit(mod, "CamProfile", "Periodic cam profile");
    WrapMagicFormulaTire::commit(mod, "MagicFormulaTire", "Reduced Magic Formula tire with theoretical-slip combined slip");
    WrapDriveFunction::commit(mod, "DriveFunction", "Time function C(t) of a driver");
    WrapDynamicsSystem::commit(mod, "DynamicsSystem", "Constrained multibody system");
    WrapModalResult::commit(mod, "ModalResult", "Result of modal_analysis()");
    wrap_factories(mod);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
