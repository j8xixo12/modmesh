# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Transfer Matrix Method for multibody systems (Rui, Wang, Zhang, Wiley 2018).

Independent second engine for cross-checks with the Haug engine
(``solvcon.multibody``). Pure Python + numpy, SI units.

Module ownership (one module per book area):

* ``linear``    - linear MSTMM: state vectors, element transfer matrices,
  overall transfer equation, eigenvalues (undamped, damped), forced and
  steady-state response, augmented eigenvectors, transient and static
  response (book Ch. 2-3, library Ch. 15.2-15.10).
* ``riccati``   - Riccati transfer and numerical stability (Ch. 10.9-10.10),
  transfer matrix derivation utilities (Ch. 10.2-10.8).
* ``rigid``     - multi-rigid-body TMM (Ch. 5).
* ``dttmm``     - discrete time TMM for rigid bodies (Ch. 7).
* ``topology``  - automatic deduction of the overall transfer equation for
  chain, tree, closed-loop, and general systems (Ch. 11).
* ``flexible``  - multi-flexible-body TMM and large-motion beams (Ch. 6, 8).
* ``continua``  - incremental TMM, 2D continua, plates, disks, cylinders
  (Ch. 4, 15.11-15.15).
* ``control``   - mixed / FE / finite segment TMM, controlled systems (Ch. 9).

Conventions (book Sec. 1.7 and Ch. 2):

* A state vector is a 1-D numpy array. Its layout is stated in the docstring
  of every element. Spatial rigid body state (Eq. 7.20):
  [x, y, z, theta_x, theta_y, theta_z, m_x, m_y, m_z, q_x, q_y, q_z, 1].
* An element transfer matrix maps the input state to the output state:
  z_out = U z_in.
* Every function that reproduces a book result cites the equation or example
  number in its docstring.
"""

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
