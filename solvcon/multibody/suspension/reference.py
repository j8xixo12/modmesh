# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Independent geometric reference solver for one corner (roadmap P9.3).

This module shares no code with the multibody engine. It solves the pose of
the upright directly from the constraint equations (link lengths, the top
mount on the strut axis, the arm pivots on their chassis axis) and the
wheel-center height with Newton iteration in pure numpy, and computes
sensitivities by central differences. It exists only to cross-check the
engine results.
"""

import math

import numpy as np

from . import metrics as mt


def _rodrigues(phi):
    th = float(np.linalg.norm(phi))
    k = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]],
                  [-phi[1], phi[0], 0]])
    if th < 1e-12:
        return np.eye(3) + k
    k = k / th
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def _log(r):
    c = np.clip(0.5 * (np.trace(r) - 1), -1.0, 1.0)
    th = math.acos(c)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return 0.5 * w if th < 1e-9 else th / (2 * math.sin(th)) * w


def _perpendicular(axis):
    """Two unit vectors perpendicular to the unit ``axis`` and to each
    other."""
    other = np.array([0.0, 0.0, 1.0])
    if abs(axis @ other) > 0.9:
        other = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, other)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


class ReferenceCorner:
    """Pose solver of the upright from its 5 constraints and the wheel-center
    z.

    :param model: corner model (SI)
    :type model: suspension.model.CornerModel
    """

    def __init__(self, model):
        """Store the geometry of the model."""
        self.m = model
        up = model.upright_points()
        self.wc0 = up["wheel_center"]
        self.local = {name: up[outer] - self.wc0 for name, (_, _, outer)
                      in model.links.items()}
        self.inner = {name: model.chassis_point(inner)
                      for name, (_, inner, _) in model.links.items()}
        self.strut = None
        if model.has_strut:
            low = up["strut_outer"]
            top = model.chassis_point("strut_inner")
            axis = (top - low) / np.linalg.norm(top - low)
            # Upright frame = global frame at design.
            e1, e2 = _perpendicular(axis)
            self.strut = (low - self.wc0, e1, e2, top)
        self.arm = None
        if model.has_arm:
            p_in = model.chassis_point("arm_inner")
            p_out = model.chassis_point("arm_outer")
            axis = (p_out - p_in) / np.linalg.norm(p_out - p_in)
            e1, e2 = _perpendicular(axis)
            self.arm = (p_in - self.wc0, p_out - self.wc0, p_in, p_out, e1, e2)
        self.kind = {name: kind for name, (kind, _, _) in model.links.items()}
        self.length = {name: model.link_length(name) for name in model.links}
        self.x = np.zeros(6)  # [wheel center, rotation vector]
        self.x[:3] = self.wc0

    def residual(self, x, z_wc, rack):
        r, a = x[:3], _rodrigues(x[3:])
        out = []
        for name, loc in self.local.items():
            inner = self.inner[name] + (rack * self.m.rack_axis
                                        if self.kind[name] == "rack"
                                        else 0.0)
            d = r + a @ loc - inner
            out.append(d @ d - self.length[name] ** 2)
        if self.strut is not None:
            # The top mount is on the strut axis: no offset across it.
            low, e1, e2, top = self.strut
            d = top - (r + a @ low)
            out += [d @ (a @ e1), d @ (a @ e2)]
        if self.arm is not None:
            # The inner pivot stays put; the outer one stays on the axis.
            l_in, l_out, p_in, p_out, e1, e2 = self.arm
            d_in = r + a @ l_in - p_in
            d_out = r + a @ l_out - p_out
            out += [d_in[0], d_in[1], d_in[2], d_out @ e1, d_out @ e2]
        out.append(r[2] - z_wc)
        return np.array(out)

    def solve(self, z_wc, rack=0.0):
        """Solve the upright pose.

        :param z_wc: wheel-center height (m)
        :type z_wc: float
        :param rack: rack travel (m)
        :type rack: float
        :return: (wheel center, rotation matrix)
        :rtype: tuple
        """
        x = self.x.copy()
        for _ in range(60):
            f = self.residual(x, z_wc, rack)
            if np.max(np.abs(f)) < 1e-15:
                break
            jac = np.zeros((6, 6))
            for k in range(6):
                h = 1e-7
                xp, xm = x.copy(), x.copy()
                xp[k] += h
                xm[k] -= h
                jac[:, k] = (self.residual(xp, z_wc, rack)
                             - self.residual(xm, z_wc, rack)) / (2 * h)
            x = x - np.linalg.solve(jac, f)
        else:
            raise RuntimeError("reference solver did not converge at "
                               f"z={z_wc}, rack={rack}")
        self.x = x
        return x[:3].copy(), _rodrigues(x[3:])

    def metrics(self, z_wc, rack=0.0, h=1e-6):
        """Camber, toe, caster, KPI, scrub, trail, and roll center at one pose.

        :return: dict of floats (SI)
        :rtype: dict
        """
        m = self.m
        r, a = self.solve(z_wc, rack)
        wg = mt.wheel_geometry(r, a, m.spin_axis, m.wheel_radius, m.side)
        cp = wg["contact_patch"]
        cp_local = a.T @ (cp - r)

        def rates(dz, drack):
            r1, a1 = self.solve(z_wc + dz * h, rack + drack * h)
            r0, a0 = self.solve(z_wc - dz * h, rack - drack * h)
            self.solve(z_wc, rack)
            w = _log(a1 @ a0.T) / (2 * h)
            return ((r1 - r0) / (2 * h), w,
                    (r1 + a1 @ cp_local - r0 - a0 @ cp_local) / (2 * h))

        axis_points = m.steering_axis_points()
        pts = m.upright_points()
        n_ax = None
        if axis_points is not None:
            ends = [r + a @ (pts[n] - pts["wheel_center"]) if "upright" == f
                    else m.chassis_point(n) for f, n in axis_points]
            n_ax, p_ax = mt.ball_joint_axis(ends[0], ends[1])
        elif m.has_rack:
            dr_s, dw_s, _ = rates(0.0, 1.0)
            n_ax, p_ax = mt.screw_axis(r, dr_s, dw_s)
        dr_z, _, v_cp = rates(1.0, 0.0)
        out = {"camber": wg["camber"], "toe": wg["toe"], "contact_patch": cp,
               "roll_center": mt.roll_center_height(cp, v_cp)}
        if n_ax is None:
            for key in ("caster", "kpi", "scrub", "trail"):
                out[key] = float("nan")
        else:
            sa = mt.steering_axis_metrics(n_ax, p_ax, cp, m.side)
            for key in ("caster", "kpi", "scrub", "trail"):
                out[key] = sa[key]
        length = self.spring_length(z_wc, rack)
        rate = (self.spring_length(z_wc + h, rack)
                - self.spring_length(z_wc - h, rack)) / (2 * h)
        self.solve(z_wc, rack)
        out["spring_length"] = length
        out["motion_ratio"] = -rate
        return out

    def _moving_point(self, r, a, name):
        """Position of an upright point, or of a point on an arm: the arm
        turns about its inner pivot axis to its ball joint on the upright
        (the angle between the design and the current ball joint, both
        projected normal to the axis)."""
        m = self.m
        arm = m.arm_of(name)
        if arm is None:
            return r + a @ (m.upright_points()[name] - self.wc0)
        front, rear, ball0 = m.arm_points(arm)
        u = (rear - front) / np.linalg.norm(rear - front)
        outer = m.links[m.arms[arm]["links"][0]][2]
        ball = r + a @ (m.upright_points()[outer] - self.wc0)
        v0 = ball0 - front
        v1 = ball - front
        v0 = v0 - (v0 @ u) * u
        v1 = v1 - (v1 @ u) * u
        phi = math.atan2(u @ np.cross(v0, v1), v0 @ v1)
        return front + _rodrigues(phi * u) @ (m.hp[name] - front)

    def spring_length(self, z_wc, rack=0.0):
        """Length of the spring line at one pose (m).

        A spring on the upright moves with it.  A spring on a rocker: the
        rocker angle about its pivot axis keeps the pushrod length (scalar
        Newton from the last angle).
        """
        m = self.m
        r, a = self.solve(z_wc, rack)
        chassis = m.chassis_point("spring_chassis")
        if not m.has_rocker:
            return float(np.linalg.norm(
                self._moving_point(r, a, "spring_upright") - chassis))
        piv = m.hp["rocker_pivot"]
        axis = m.rocker_axis / np.linalg.norm(m.rocker_axis)
        rod_up = self._moving_point(r, a, "pushrod_outer")
        arm = m.hp["pushrod_inner"] - piv
        rod = float(np.linalg.norm(m.hp["pushrod_outer"]
                                   - m.hp["pushrod_inner"]))
        theta = getattr(self, "_theta", 0.0)
        for _ in range(60):
            f = np.linalg.norm(piv + _rodrigues(theta * axis) @ arm
                               - rod_up) - rod
            if abs(f) < 1e-15:
                break
            dt = 1e-7
            fp = np.linalg.norm(piv + _rodrigues((theta + dt) * axis) @ arm
                                - rod_up) - rod
            theta -= f * dt / (fp - f)
        else:
            raise RuntimeError("reference solver: rocker angle did not "
                               "converge")
        self._theta = theta
        spring = piv + _rodrigues(theta * axis) @ (m.hp["spring_rocker"]
                                                   - piv)
        return float(np.linalg.norm(spring - chassis))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
