# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Andrews' squeezer mechanism (Schiehlen 1990; IFToMM benchmark library).

Seven planar links, ten revolute joints (three of them redundant in 3D,
handled by the solver's rank-revealing constraint solve), a spring, and a
constant driving torque, without gravity.  The data follow the IFToMM
problem description (local link frames, Tables 1-3); the initial crank
angle is beta0 = -0.0620 rad with all velocities zero.

``simulate`` returns the time history of point F (the crank tip) and the
energy terms, the same columns as the IFToMM reference results file.
Run: python3 -m solvcon.multibody.models.validation.andrews_squeezer
"""

import math

import numpy as np

from .... import multibody as mbs

Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]

POINTS = {"O": (0.0, 0.0), "A": (-0.06934, -0.00227), "B": (-0.03635, 0.03273),
          "C": (0.014, 0.072)}
#: link: (local points, centre of mass, mass, Iz); the frame origin is the
#: first point, x along the rod (for E-B-D, y from E toward B).
LINKS = {
    "OF": ({"O": (0.0, 0.0), "F": (0.007, 0.0)}, (0.00092, 0.0), 0.04325,
           2.194e-6),
    "EF": ({"E": (0.0, 0.0), "F": (0.028, 0.0)}, (0.0165, 0.0), 0.00365,
           4.410e-7),
    "HE": ({"H": (0.0, 0.0), "E": (0.02, 0.0)}, (0.00579, 0.0), 0.00706,
           5.667e-7),
    "GE": ({"G": (0.0, 0.0), "E": (0.02, 0.0)}, (0.00579, 0.0), 0.00706,
           5.667e-7),
    "AG": ({"A": (0.0, 0.0), "G": (0.04, 0.0)}, (0.02308, 0.00916), 0.07050,
           1.169e-5),
    "AH": ({"A": (0.0, 0.0), "H": (0.04, 0.0)}, (0.01228, -0.00449), 0.05498,
           1.912e-5),
    "EBD": ({"E": (0.0, 0.0), "B": (0.0, 0.035), "D": (0.02, 0.017)},
            (0.01043, 0.01626), 0.02373, 5.255e-6),
}
#: (link i, point, link j or ground); joints at the same point name.
JOINTS = [("OF", "O", None), ("OF", "F", "EF"), ("EF", "E", "EBD"),
          ("HE", "E", "EBD"), ("GE", "E", "EBD"), ("AG", "G", "GE"),
          ("AH", "H", "HE"), ("AG", "A", None), ("AH", "A", None),
          ("EBD", "B", None)]
SPRING_K, SPRING_L0, TORQUE = 4530.0, 0.07785, 0.033
ORDER = ["OF", "EF", "HE", "GE", "AG", "AH", "EBD"]


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def assemble(beta0):
    """Solve the planar loop closure for the link angles at crank angle beta0.

    The start guess is the Hairer-Wanner initial state mapped to link
    angles (theta_EF = beta + Theta, theta_EBD = gamma, theta_AG = delta,
    theta_GE = Phi + delta - pi/2, theta_AH = epsilon - pi/2, theta_HE =
    Omega + epsilon).

    :return: {link: absolute angle}
    :rtype: dict
    """
    hw = dict(beta=-0.0617138900142764, Theta=0.0, gamma=0.455279819163070,
              Phi=0.222668390165886, delta=0.487364979543843,
              Omega=-0.222668390165886, epsilon=1.23054744454982)
    x = np.array([hw["beta"] + hw["Theta"], hw["gamma"], hw["delta"],
                  hw["Phi"] + hw["delta"] - math.pi / 2,
                  hw["epsilon"] - math.pi / 2, hw["Omega"] + hw["epsilon"]])
    a, b = np.array(POINTS["A"]), np.array(POINTS["B"])

    def residual(x):
        t_ef, t_ebd, t_ag, t_ge, t_ah, t_he = x
        f = _rot(beta0) @ LINKS["OF"][0]["F"]
        e = f - _rot(t_ef) @ LINKS["EF"][0]["F"]
        r1 = e + _rot(t_ebd) @ LINKS["EBD"][0]["B"] - b
        g = a + _rot(t_ag) @ LINKS["AG"][0]["G"]
        r2 = g + _rot(t_ge) @ LINKS["GE"][0]["E"] - e
        h = a + _rot(t_ah) @ LINKS["AH"][0]["H"]
        r3 = h + _rot(t_he) @ LINKS["HE"][0]["E"] - e
        return np.concatenate([r1, r2, r3])

    for _ in range(50):
        r = residual(x)
        if np.linalg.norm(r) < 1e-15:
            break
        jac = np.zeros((6, 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-8
            jac[:, k] = (residual(x + dx) - residual(x - dx)) / 2e-8
        x = x - np.linalg.solve(jac, r)
    else:
        raise RuntimeError("loop closure did not converge")
    names = ["EF", "EBD", "AG", "GE", "AH", "HE"]
    out = dict(zip(names, x))
    out["OF"] = beta0
    return out


def origin_of(name, angles):
    """Global position of the frame origin (first point) of a link."""
    a = np.array(POINTS["A"])
    f = _rot(angles["OF"]) @ LINKS["OF"][0]["F"]
    e = f - _rot(angles["EF"]) @ LINKS["EF"][0]["F"]
    return {"OF": np.zeros(2), "EF": e, "EBD": e, "AG": a, "AH": a,
            "GE": a + _rot(angles["AG"]) @ LINKS["AG"][0]["G"],
            "HE": a + _rot(angles["AH"]) @ LINKS["AH"][0]["H"]}[name]


def build(beta0=-0.062, dt=1e-4, rtol=1e-10, atol=1e-12):
    """Build the mechanism in the solver; return (system, handles)."""
    angles = assemble(beta0)
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    bodies, cm_local = {}, {}
    for num, name in enumerate(ORDER, start=1):
        pts, cm, mass, iz = LINKS[name]
        th = angles[name]
        cm_local[name] = np.array(cm)
        pos = origin_of(name, angles) + _rot(th) @ cm_local[name]
        quat = [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]
        body = mbs.create_mobilized_body_quat(num, [pos[0], pos[1], 0.0], Z, Z,
                                              quat, Z, Z, mass, [iz, iz, iz],
                                              Z, Z)
        s.Add(body)
        bodies[name] = body

    def local(name, point):
        p = np.array(LINKS[name][0][point]) - cm_local[name]
        return [p[0], p[1], 0.0]

    for li, point, lj in JOINTS:
        if lj is None:
            g = POINTS[point]
            s.Add(mbs.create_joint(1, [g[0], g[1], 0.0], local(li, point), Z,
                                   Z, EZ, EZ, ground, bodies[li]))
        else:
            s.Add(mbs.create_joint(1, local(li, point), local(lj, point), Z, Z,
                                   EZ, EZ, bodies[li], bodies[lj]))
    c = POINTS["C"]
    s.Add(mbs.create_spring_damper(ground, bodies["EBD"], [c[0], c[1], 0.0],
                                   local("EBD", "D"), SPRING_K, 0.0,
                                   SPRING_L0))
    s.init()
    s.set_applied_load(ORDER.index("OF") + 1, Z, [0.0, 0.0, TORQUE])
    s.set_integrator("rk45")
    s.set_tolerances(rtol, atol)
    return (s,
            {"bodies": bodies, "cm_local": cm_local, "angles": angles,
             "beta0": beta0})


def state(s, handles, beta_prev=None):
    """Point F, the energies, and the crank angle (unwrapped from
    beta_prev)."""
    pos = np.array(s.get_body_positions())
    quat = np.array(s.get_body_quaternions())
    vel = np.array(s.get_body_velocities())
    omg = np.array(s.get_body_angular_velocities())
    kinetic = 0.0
    for num, name in enumerate(ORDER, start=1):
        _, _, mass, iz = LINKS[name]
        kinetic += (0.5 * mass * float(vel[num] @ vel[num])
                    + 0.5 * iz * omg[num][2] ** 2)

    def angle(num):
        q = quat[num]
        return 2.0 * math.atan2(q[3], q[0])

    def glob(name, point):
        num = ORDER.index(name) + 1
        p = np.array(LINKS[name][0][point]) - handles["cm_local"][name]
        return pos[num][:2] + _rot(angle(num)) @ p

    beta = angle(1)
    if beta_prev is not None:
        beta += 2 * math.pi * round((beta_prev - beta) / (2 * math.pi))
    f = glob("OF", "F")
    d = glob("EBD", "D")
    length = float(np.linalg.norm(d - np.array(POINTS["C"])))
    potential = 0.5 * SPRING_K * (length - SPRING_L0) ** 2
    work = TORQUE * (beta - handles["beta0"])
    return {"t": s.get_sim_time(), "xF": f[0], "yF": f[1], "T": kinetic,
            "V": potential, "W": work, "beta": beta}


def simulate(t_end=0.05, dt=1e-4, **kw):
    s, h = build(dt=dt, **kw)
    rows = [state(s, h)]
    steps = int(round(t_end / dt))
    for _ in range(steps):
        s.solve()
        rows.append(state(s, h, rows[-1]["beta"]))
    return rows


def main():
    rows = simulate()
    e0 = rows[0]["T"] + rows[0]["V"]
    drift = max(abs(r["T"] + r["V"] - e0 - r["W"]) for r in rows)
    for r in rows[::100]:
        print(f"t {r['t']:.4f}  xF {r['xF']:+.9e}  yF {r['yF']:+.9e}  T "
              f"{r['T']:.9e}  "
              f"V {r['V']:.9e}  W {r['W']:.9e}")
    print(f"max energy balance drift {drift:.3e} J")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
