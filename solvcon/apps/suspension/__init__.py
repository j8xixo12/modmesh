# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Suspension kinematics and compliance of a race car front corner.

The corner is solved by an engine outside solvcon: the pybind11 module
``multibody_solver`` and the Python package ``suspension`` of the
Multibody-Dynamics-Solver project, which :mod:`._engine` finds and imports.
The session holds the analyses without Qt, so a script or a test runs the
same sweeps the GUI behind the pilot toggle shows.
"""

from ... import pilot

__all__, __getattr__, __dir__ = pilot.lazy_exports(__name__, {
    '_app': ['SuspensionApp'],
    '_engine': ['Engine', 'EngineUnavailable', 'load'],
    '_plots': ['KCPlot'],
    '_scene': ['CornerGeometry', 'vehicle_geometries', 'vehicle_view_pose',
               'vehicle_body_segments'],
    '_view': ['CornerView'],
    '_session': ['SuspensionSession', 'SweepResult', 'QuarterCarRun'],
}, ungated=('_engine', '_scene', '_session'))

if pilot.enable:
    from . import _app
    pilot.register_app("suspension", _app.build_app)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
