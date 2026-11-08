# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""A static GLB model viewer, with a headless reader and scene adapter."""

from ... import pilot

__all__, __getattr__, __dir__ = pilot.lazy_exports(__name__, {
    '_app': ['ModelViewerApp'],
    '_gltf': ['Model', 'Node', 'Primitive', 'load_glb'],
    '_scene': ['Surface', 'surfaces', 'bounds', 'world_matrices'],
}, ungated=('_gltf', '_scene'))

if pilot.enable:
    from . import _app
    pilot.register_app('modelview', _app.build_app)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
