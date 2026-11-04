# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Example mechanisms of the animation gallery, as model builders.

Every ``build_*`` function returns a system that is ready for ``init()``
(the closed-loop ones also skip ``Assembly()``). Some return a tuple whose
first item is the system.

Reference: Haug, "Computer Aided Kinematics and Dynamics of Mechanical
Systems", Chapters 9-12.
"""

import math

import numpy as np

from ... import multibody as mbs

Z = [0, 0, 0]
AZ = [0, 0, 1]
AX = [1, 0, 0]
AY = [0, 1, 0]


def build_pendulum():
    s = mbs.DynamicsSystem(0.001)
    g = mbs.create_ground(0)
    s.Add(g)
    b = mbs.create_mobilized_body(1, Z, Z, Z, [1.5, 0, 0], Z, Z,
                                  1.0, [1, 1, 1], [0, -9.8, 0], Z)
    s.Add(b)
    s.Add(mbs.create_joint(0, Z, [0, 1, 0], Z, Z, AZ, AZ, g, b))
    return s


def build_double_pendulum():
    s = mbs.DynamicsSystem(0.0005)
    g = mbs.create_ground(0)
    s.Add(g)
    b1 = mbs.create_mobilized_body(1, Z, Z, Z, [2.5, 0, 0], Z, Z,
                                   2.0, [1, 1, 1], [0, -9.8 * 2, 0], Z)
    b2 = mbs.create_mobilized_body(2, Z, Z, Z, [2.0, 0, 0], Z, Z,
                                   1.0, [1, 1, 1], [0, -9.8, 0], Z)
    s.Add(b1)
    s.Add(b2)
    s.Add(mbs.create_joint(0, Z, [0, 1.5, 0], Z, Z, AZ, AZ, g, b1))
    s.Add(mbs.create_joint(0, [0, -1.5, 0], [0, 1.5, 0], Z, Z, AZ, AZ, b1, b2))
    return s


def build_triple_pendulum():
    s = mbs.DynamicsSystem(0.0005)
    g = mbs.create_ground(0)
    s.Add(g)
    b1 = mbs.create_mobilized_body(1, Z, Z, Z, [2.0, 0, 0], Z, Z,
                                   1.5, [0.5, 0.5, 0.5], [0, -9.8 * 1.5, 0],
                                   Z)
    b2 = mbs.create_mobilized_body(2, Z, Z, Z, [1.5, 0, 0], Z, Z,
                                   1.0, [0.3, 0.3, 0.3], [0, -9.8, 0], Z)
    b3 = mbs.create_mobilized_body(3, Z, Z, Z, [1.0, 0, 0], Z, Z,
                                   0.5, [0.1, 0.1, 0.1], [0, -9.8 * 0.5, 0],
                                   Z)
    s.Add(b1)
    s.Add(b2)
    s.Add(b3)
    s.Add(mbs.create_joint(0, Z, [0, 1.0, 0], Z, Z, AZ, AZ, g, b1))
    s.Add(mbs.create_joint(0, [0, -1.0, 0], [0, 1.0, 0], Z, Z, AZ, AZ, b1, b2))
    s.Add(mbs.create_joint(0, [0, -1.0, 0], [0, 0.8, 0], Z, Z, AZ, AZ, b2, b3))
    return s


def build_chain(n=4):
    s = mbs.DynamicsSystem(0.001)
    g = mbs.create_ground(0)
    s.Add(g)
    prev = g
    for i in range(n):
        ang = 1.0 if i == 0 else 0.4
        b = mbs.create_mobilized_body(i + 1, Z, Z, Z, [ang, 0, 0], Z, Z,
                                      1.0, [1, 1, 1], [0, -9.8, 0], Z)
        s.Add(b)
        s.Add(mbs.create_joint(0, Z, [0, 1, 0], Z, Z, AZ, AZ, prev, b))
        prev = b
    return s


def build_spherical_pendulum():
    s = mbs.DynamicsSystem(0.001)
    g = mbs.create_ground(0)
    s.Add(g)
    b = mbs.create_mobilized_body(1, Z, [0, 0, 2.0], Z, [0.3, 0, 0.3], Z, Z,
                                  1.0, [1, 1, 1], [0, -9.8, 0], Z)
    s.Add(b)
    s.Add(mbs.create_joint(0, Z, [0, 2.0, 0], Z, Z, AZ, AZ, g, b))
    return s


def build_driven_rotor():
    s = mbs.DynamicsSystem(0.001)
    g = mbs.create_ground(0)
    s.Add(g)
    # Rotor: revolute about Z at origin, driven
    rotor = mbs.create_mobilized_body(1, Z, Z, Z, Z, Z, Z,
                                      5.0, [0.5, 0.5, 0.5], Z, Z)
    s.Add(rotor)
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AZ, AZ, g, rotor))
    s.Add(mbs.create_joint_driving(3, Z, Z, Z, Z, AZ, AZ, g, rotor, 6.28))
    # Arm: revolute joint (axis Z) at rotor tip, so arm swings in XY plane
    arm = mbs.create_mobilized_body(2, Z, Z, Z, Z, Z, Z,
                                    2.0, [0.1, 0.1, 0.1], [0, -9.8 * 2, 0], Z)
    s.Add(arm)
    s.Add(mbs.create_joint(1, [1.0, 0, 0], [0, 0.5, 0], Z, Z, AZ, AZ, rotor,
                           arm))
    return s


def build_spring_pendulum():
    s = mbs.DynamicsSystem(0.001)
    g = mbs.create_ground(0)
    s.Add(g)
    b = mbs.create_mobilized_body(1, Z, Z, Z, [1.0, 0, 0], Z, Z,
                                  2.0, [0.5, 0.5, 0.5], [0, -9.8 * 2, 0], Z)
    s.Add(b)
    s.Add(mbs.create_joint(0, Z, [0, 1.5, 0], Z, Z, AZ, AZ, g, b))
    s.Add(mbs.create_spring_damper(g, b, Z, Z, 50.0, 3.0, 1.5, 0.0))
    return s


def build_slider_crank():
    dt = 0.0001
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    omega_0 = 2.0 * math.pi
    e0_c = math.sqrt(1.0 - 0.71**2)
    crank = mbs.create_mobilized_body_quat(
        1, [0.0, 0.1, 0.12], Z, Z,
        [e0_c, 0.71, 0.0, 0.0], [omega_0, 0, 0], Z,
        1.0, [0.01, 0.01, 0.01], Z, Z)
    s.Add(crank)
    e0_r = math.sqrt(1.0 - 0.21**2 - 0.40**2 - 0.1**2)
    rod = mbs.create_mobilized_body_quat(
        2, [0.1, 0.05, 0.1], Z, Z,
        [e0_r, -0.21, 0.40, -0.1], Z, Z,
        1.0, [0.01, 0.01, 0.01], Z, Z)
    s.Add(rod)
    slider = mbs.create_mobilized_body_quat(
        3, [0.2, 0.0, 0.0], Z, Z,
        [1.0, 0.0, 0.0, 0.0], Z, Z,
        1.0, [0.01, 0.01, 0.01], Z, Z)
    s.Add(slider)
    s.Add(mbs.create_joint(1, Z, [0, 0.1, 0.12], Z, Z, AX, AX, crank, g))
    s.Add(mbs.create_joint(0, [0, 0.08, 0], [-0.15, 0, 0], Z, Z, AX, AX, crank,
                           rod))
    s.Add(mbs.create_joint(7, [0.15, 0, 0], [0.2, 0, 0], Z, Z, AY, AX, rod,
                           slider))
    s.Add(mbs.create_joint(2, Z, [0.2, 0, 0], Z, Z, AX, AX, slider, g))
    s.Add(mbs.create_joint_distance(6, [0.15, 0, 0], [1.0, 0, 0], Z, Z,
                                    AX, AX, rod, slider, 1.0))
    s.Add(mbs.create_joint_driving(3, Z, Z, Z, Z, AX, AX, g, crank, omega_0))
    return s, dt


def build_spring_mass():
    s = mbs.DynamicsSystem(0.001)
    g = mbs.create_ground(0)
    s.Add(g)

    # Arm (body 1): pivots at origin, hangs down, displaced 30 deg initially
    arm = mbs.create_mobilized_body(
        1, Z, Z, Z, [0.5, 0, 0], Z, Z,
        3.0, [0.5, 0.5, 0.5], [0, -9.8 * 3, 0], Z)
    s.Add(arm)
    s.Add(mbs.create_joint(0, Z, [0, 0.8, 0], Z, Z, AZ, AZ, g, arm))

    # Spring-damper from a fixed point to the arm tip
    # k=100 N/m, c=5 Ns/m, L0=1.0m
    s.Add(mbs.create_spring_damper(
        g, arm, [0, 0, 1.0], [0, -0.8, 0], 100.0, 5.0, 1.0, 0.0))

    return s


def build_four_bar():
    s = mbs.DynamicsSystem(0.001)
    g = mbs.create_ground(0)
    s.Add(g)

    # Link 1 (crank): rotates about x at origin
    link1 = mbs.create_mobilized_body(
        1, Z, Z, Z, Z, Z, Z,
        1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(link1)

    # Link 2 (coupler)
    link2 = mbs.create_mobilized_body(
        2, [0, 0, 1.0], Z, Z, Z, Z, Z,
        2.0, [0.5, 0.5, 0.5], [0, -9.8 * 2, 0], Z)
    s.Add(link2)

    # Link 3 (output): rotates about x at (-4, -8.5, 0)
    link3 = mbs.create_mobilized_body(
        3, [0, -4.0, -8.5], Z, Z, Z, Z, Z,
        1.5, [0.3, 0.3, 0.3], [0, -9.8 * 1.5, 0], Z)
    s.Add(link3)

    # Joint A: Revolute (link1-ground) at origin, axis = x
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AX, AX, g, link1))

    # Joint B: Universal (link1-link2) at (0,0,2) on link1
    s.Add(mbs.create_joint(4, [0, 0, 2.0], [0, 6.1, 0], Z, Z,
                           AX, AX, link1, link2))

    # Joint C: Spherical (link2-link3)
    s.Add(mbs.create_joint(0, [0, -6.1, 0], [0, -3.7, 0], Z, Z,
                           AX, AX, link2, link3))

    # Joint D: Revolute (link3-ground) at (-4, -8.5, 0), axis = x
    s.Add(mbs.create_joint(1, [0, 3.7, 0], [-4.0, -8.5, 0], Z, Z,
                           AX, AX, link3, g))

    # Driver: link1 rotates at pi rad/s about x
    s.Add(mbs.create_joint_driving(3, Z, Z, Z, Z, AX, AX, g, link1, math.pi))

    return s


def build_crank_rocker():
    dt = 0.001
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    # Ground pivots: A=(0,0), D=(4,0)
    # Crank AB=1, Coupler BC=3, Rocker CD=3.5
    # Compute closed-loop initial positions analytically
    D = np.array([4.0, 0.0])
    B = np.array([1.0, 0.0])  # crank at angle 0
    # C: intersection of circle(B, 3) and circle(D, 3.5)
    d_BD = np.linalg.norm(D - B)
    a = (9.0 - 12.25 + d_BD**2) / (2 * d_BD)
    h = math.sqrt(max(0, 9.0 - a * a))
    ux = (D - B) / d_BD
    C = B + a * ux + h * np.array([-ux[1], ux[0]])

    # Body rotations (Z-axis, planar)
    def z_quat(theta):
        return [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]
    theta_coupler = math.atan2(C[1] - B[1], C[0] - B[0])
    theta_rocker = math.atan2(D[1] - C[1], D[0] - C[0])
    mid_BC = ((B + C) / 2).tolist() + [0]
    mid_CD = ((C + D) / 2).tolist() + [0]
    # Crank (body 1): angle 0
    crank = mbs.create_mobilized_body_quat(
        1, [0.5, 0, 0], Z, Z, [1, 0, 0, 0], Z, Z,
        1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(crank)
    # Coupler (body 2): rotated
    coupler = mbs.create_mobilized_body_quat(
        2, mid_BC, Z, Z, z_quat(theta_coupler), Z, Z,
        1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(coupler)
    # Rocker (body 3): rotated
    rocker = mbs.create_mobilized_body_quat(
        3, mid_CD, Z, Z, z_quat(theta_rocker), Z, Z,
        1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(rocker)
    # Joint A: revolute crank-ground at (0,0), axis Z
    s.Add(mbs.create_joint(1, [-0.5, 0, 0], Z, Z, Z, AZ, AZ, crank, g))
    # Joint B: revolute crank-coupler at crank tip (1,0)
    s.Add(mbs.create_joint(1, [0.5, 0, 0], [-1.5, 0, 0], Z, Z, AZ, AZ, crank,
                           coupler))
    # Joint C: revolute coupler-rocker at coupler end
    s.Add(mbs.create_joint(1, [1.5, 0, 0], [-1.75, 0, 0], Z, Z, AZ, AZ,
                           coupler, rocker))
    # Joint D: revolute rocker-ground at (4,0)
    s.Add(mbs.create_joint(1, [1.75, 0, 0], [4, 0, 0], Z, Z, AZ, AZ, rocker,
                           g))
    # Drive crank at 2*pi rad/s
    s.Add(mbs.create_joint_driving(3, Z, Z, Z, Z, AZ, AZ, g, crank,
                                   2 * math.pi))
    return s, dt


def build_scotch_yoke():
    dt = 0.001
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    R = 1.5  # crank radius
    # Crank (body 1): rotates about Z at origin
    crank = mbs.create_mobilized_body(
        1, Z, Z, Z, Z, Z, Z,
        1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(crank)
    # Yoke (body 2): slides along Y axis, initially at Y=0
    yoke = mbs.create_mobilized_body(
        2, Z, Z, Z, Z, Z, Z,
        2.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(yoke)
    # Crank revolute at origin
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AZ, AZ, g, crank))
    # Yoke prismatic along Y (only Y translation allowed)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, AY, AY, g, yoke))
    # Pin-in-slot: the crank pin at (R,0) stays on the yoke slot line (yoke
    # x axis through the yoke origin) and may rotate in the slot. This is the
    # revolute-cylindrical joint (Haug Eq. 9.4.32): pin axis z on the crank,
    # slot axis x on the yoke. A prismatic joint here would lock the relative
    # rotation of crank and yoke, which the mechanism needs.
    s.Add(mbs.create_joint(7, [R, 0, 0], Z, Z, Z, AZ, AX, crank, yoke))
    # Drive crank
    s.Add(mbs.create_joint_driving(3, Z, Z, Z, Z, AZ, AZ, g, crank,
                                   2 * math.pi))
    return s, dt


# Reachable arm range from a parameter sweep: 36.7 to 139.25 degrees.
PEAUCELLIER_MID = math.radians(88.0)
PEAUCELLIER_AMP = math.radians(43.0)  # 45 to 131 degrees, starts at 45


def build_peaucellier(drive_func=None):
    dt = 0.001
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    # Parameters: L=5, l=2.5, d=4, driving arm=d
    L = 5.0   # long links OB, OC
    l_rh = 2.5  # rhombus side l
    d = 4.0   # distance O to A
    # Fixed pivots: O=(0,0), A=(-d,0)=(-4,0)
    # Driving arm: A to M, length=d, M starts at (0,0)
    # At initial config: M=(0,0), angle=0
    # B is above M: B=(0, sqrt(L^2 - 0^2))... need to think about initial
    # positions. When driving arm is horizontal: M = A + d*[cos(0), sin(0)] =
    # (-4+4, 0) = (0,0) B and C are on circle of radius L from O, and on circle
    # of radius l from M |OB|=L=5, |MB|=l=2.5, O=(0,0), M=(0,0) initially
    # Wait, if M=O=(0,0), then B is at distance L from O AND l from M=O
    # That means L=l=5=2.5, contradiction. Let me reconsider.
    # Better: start with driving arm at angle theta=pi/4
    theta0 = math.pi / 4
    M = [-d + d * math.cos(theta0), d * math.sin(theta0)]
    # B and C symmetric about line OM
    # |OB|=L, |MB|=l
    # B lies at intersection of circle(O,L) and circle(M,l)
    Mx, My = M
    OM = math.sqrt(Mx**2 + My**2)
    # Using intersection of two circles:
    # x_B along OM direction: cos_alpha = (OM^2 + L^2 - l^2) / (2*OM*L)
    cos_alpha = (OM**2 + L**2 - l_rh**2) / (2 * OM * L)
    sin_alpha = math.sqrt(max(0, 1 - cos_alpha**2))
    # Direction from O to M
    ux, uy = Mx / OM, My / OM
    # Perpendicular
    px, py = -uy, ux
    Bx = L * (cos_alpha * ux + sin_alpha * px)
    By = L * (cos_alpha * uy + sin_alpha * py)
    Cx = L * (cos_alpha * ux - sin_alpha * px)
    Cy = L * (cos_alpha * uy - sin_alpha * py)
    # P is the opposite vertex of rhombus from M: P = B + C - M
    Px = Bx + Cx - Mx
    Py = By + Cy - My

    # Helper: Z-axis quaternion from angle
    def z_quat(theta):
        return [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]

    def link_quat(x1, y1, x2, y2):
        return z_quat(math.atan2(y2 - y1, x2 - x1))

    # Bodies (all in XY plane, use Z-axis revolute joints)
    # Provide correct rotation quaternions so NR converges for closed loops
    Ax, Ay = -d, 0
    # Body 1: driving arm (A to M), center at midpoint
    arm_cx, arm_cy = (Ax + Mx) / 2, (Ay + My) / 2
    arm = mbs.create_mobilized_body_quat(
        1, [arm_cx, arm_cy, 0], Z, Z, link_quat(Ax, Ay, Mx, My), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(arm)
    # Body 2: link OB (O to B), center at midpoint
    link_OB = mbs.create_mobilized_body_quat(
        2, [Bx / 2, By / 2, 0], Z, Z, link_quat(0, 0, Bx, By), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(link_OB)
    # Body 3: link OC (O to C), center at midpoint
    link_OC = mbs.create_mobilized_body_quat(
        3, [Cx / 2, Cy / 2, 0], Z, Z, link_quat(0, 0, Cx, Cy), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(link_OC)
    # Body 4: link BP (B to P), center at midpoint
    link_BP = mbs.create_mobilized_body_quat(
        4, [(Bx + Px) / 2, (By + Py) / 2, 0], Z, Z,
        link_quat(Bx, By, Px, Py), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(link_BP)
    # Body 5: link CP (C to P), center at midpoint
    link_CP = mbs.create_mobilized_body_quat(
        5, [(Cx + Px) / 2, (Cy + Py) / 2, 0], Z, Z,
        link_quat(Cx, Cy, Px, Py), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(link_CP)
    # Body 6: link BM (B to M), center at midpoint
    link_BM = mbs.create_mobilized_body_quat(
        6, [(Bx + Mx) / 2, (By + My) / 2, 0], Z, Z,
        link_quat(Mx, My, Bx, By), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(link_BM)
    # Body 7: link CM (C to M), center at midpoint
    link_CM = mbs.create_mobilized_body_quat(
        7, [(Cx + Mx) / 2, (Cy + My) / 2, 0], Z, Z,
        link_quat(Mx, My, Cx, Cy), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(link_CM)

    # Half-lengths for local joint positions
    arm_half = d / 2
    OB_half = L / 2
    OC_half = L / 2
    BP_half = l_rh / 2
    CP_half = l_rh / 2
    BM_half = l_rh / 2
    CM_half = l_rh / 2

    # Joints (all revolute about Z, planar mechanism)
    # Joint at A=(-d,0): arm to ground
    s.Add(mbs.create_joint(1, [-arm_half, 0, 0], [-d, 0, 0], Z, Z, AZ, AZ, arm,
                           g))
    # Joint at M: arm tip to link_BM, link_CM
    s.Add(mbs.create_joint(1, [arm_half, 0, 0], [-BM_half, 0, 0], Z, Z, AZ, AZ,
                           arm, link_BM))
    s.Add(mbs.create_joint(1, [arm_half, 0, 0], [-CM_half, 0, 0], Z, Z, AZ, AZ,
                           arm, link_CM))
    # Joint at O=(0,0): link_OB and link_OC to ground
    s.Add(mbs.create_joint(1, [-OB_half, 0, 0], Z, Z, Z, AZ, AZ, link_OB, g))
    s.Add(mbs.create_joint(1, [-OC_half, 0, 0], Z, Z, Z, AZ, AZ, link_OC, g))
    # Joint at B: link_OB, link_BP, link_BM meet
    s.Add(mbs.create_joint(1, [OB_half, 0, 0], [-BP_half, 0, 0], Z, Z, AZ, AZ,
                           link_OB, link_BP))
    s.Add(mbs.create_joint(1, [OB_half, 0, 0], [BM_half, 0, 0], Z, Z, AZ, AZ,
                           link_OB, link_BM))
    # Joint at C: link_OC, link_CP, link_CM meet
    s.Add(mbs.create_joint(1, [OC_half, 0, 0], [-CP_half, 0, 0], Z, Z, AZ, AZ,
                           link_OC, link_CP))
    s.Add(mbs.create_joint(1, [OC_half, 0, 0], [CM_half, 0, 0], Z, Z, AZ, AZ,
                           link_OC, link_CM))
    # Joint at P: link_BP and link_CP meet
    s.Add(mbs.create_joint(1, [BP_half, 0, 0], [CP_half, 0, 0], Z, Z, AZ, AZ,
                           link_BP, link_CP))
    # Drive: arm rotates about A
    # The inversor locks up when the driving arm passes about 139 degrees
    # (the rhombus folds flat). Drive the arm back and forth inside its
    # reachable range instead of spinning it (range from a parameter sweep).
    s.Add(mbs.create_rotational_driver(
        body_i=g, body_j=arm, axis_i=AZ, axis_j=AZ,
        func=drive_func or mbs.DriveFunction.sine(amplitude=PEAUCELLIER_AMP,
                                                  omega=math.pi * 0.5,
                                                  phase=-math.pi / 2,
                                                  offset=PEAUCELLIER_MID)))
    return s, dt


def build_klann_leg():
    dt = 0.001
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    # Klann linkage geometry (normalized, scaled by factor 2)
    # Ground pivots: O=(0,0) and E=(2,0)
    # Crank OA: length 1
    # Link AB: length 2.5
    # Triangle BCD (ternary link): B, C, D are vertices
    # Link EC: length 2.0
    # Foot at D traces the walking path
    # Approximate dimensions from Klann patent:
    OA = 1.0  # crank
    # Upper connecting rod AB=2.5, rocker EC=2.0
    # Triangle BCD: BC=1.5, BD=2.0, CD=1.8 (ternary link)
    Ex, Ey = 2.0, 0.0

    # Initial positions (crank at 0 degrees, A at (1,0))
    Ax, Ay = OA, 0.0
    # B: intersection of circle(A, AB) and circle(E, EC+some offset)
    # For simplicity, place B above the line: B ~ (1.5, 2.0)
    Bx, By = 1.5, 2.0
    # C: on link from E, C ~ (2.5, 1.8)
    Cx, Cy = 2.5, 1.8
    # D: foot point, below B: D ~ (1.0, -0.5)
    Dx, Dy = 1.0, -0.5

    # Helper: Z-axis quaternion
    def z_quat(theta):
        return [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]

    def link_quat(x1, y1, x2, y2):
        return z_quat(math.atan2(y2 - y1, x2 - x1))

    # Bodies (with correct rotation quaternions for closed-loop convergence)
    # Body 1: Crank (O to A)
    crank = mbs.create_mobilized_body_quat(
        1, [Ax / 2, Ay / 2, 0], Z, Z,
        link_quat(0, 0, Ax, Ay), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(crank)
    # Body 2: Upper link (A to B)
    link_AB = mbs.create_mobilized_body_quat(
        2, [(Ax + Bx) / 2, (Ay + By) / 2, 0], Z, Z,
        link_quat(Ax, Ay, Bx, By), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(link_AB)
    # Body 3: Ternary link (triangle BCD) - center approx at centroid
    tcx = (Bx + Cx + Dx) / 3
    tcy = (By + Cy + Dy) / 3
    ternary = mbs.create_mobilized_body_quat(
        3, [tcx, tcy, 0], Z, Z, [1, 0, 0, 0], Z, Z,
        1.0, [0.05, 0.05, 0.05], Z, Z)
    s.Add(ternary)
    # Body 4: Rocker (E to C)
    rocker = mbs.create_mobilized_body_quat(
        4, [(Ex + Cx) / 2, (Ey + Cy) / 2, 0], Z, Z,
        link_quat(Ex, Ey, Cx, Cy), Z, Z,
        0.5, [0.01, 0.01, 0.01], Z, Z)
    s.Add(rocker)

    # Local coordinates: each body's center is at the midpoint of its link
    OA_h = OA / 2
    AB_h = math.sqrt((Bx - Ax)**2 + (By - Ay)**2) / 2
    EC_h = math.sqrt((Cx - Ex)**2 + (Cy - Ey)**2) / 2
    # Ternary link local coords (relative to centroid)
    tB = [Bx - tcx, By - tcy, 0]
    tC = [Cx - tcx, Cy - tcy, 0]
    tD = [Dx - tcx, Dy - tcy, 0]

    # Joints
    # O: crank to ground
    s.Add(mbs.create_joint(1, [-OA_h, 0, 0], Z, Z, Z, AZ, AZ, crank, g))
    # A: crank to link_AB
    s.Add(mbs.create_joint(1, [OA_h, 0, 0], [-AB_h, 0, 0], Z, Z, AZ, AZ, crank,
                           link_AB))
    # B: link_AB to ternary
    s.Add(mbs.create_joint(1, [AB_h, 0, 0], tB, Z, Z, AZ, AZ, link_AB,
                           ternary))
    # E: rocker to ground
    s.Add(mbs.create_joint(1, [-EC_h, 0, 0], [Ex, Ey, 0], Z, Z, AZ, AZ, rocker,
                           g))
    # C: rocker to ternary
    s.Add(mbs.create_joint(1, [EC_h, 0, 0], tC, Z, Z, AZ, AZ, rocker, ternary))
    # Drive crank
    s.Add(mbs.create_joint_driving(3, Z, Z, Z, Z, AZ, AZ, g, crank,
                                   2 * math.pi))
    return s, dt, tD  # return foot point local coords for animation

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
