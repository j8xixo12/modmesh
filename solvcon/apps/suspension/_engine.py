# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Load the multibody engine and the suspension package the app drives.

Both ship with solvcon: the engine is :mod:`solvcon.multibody` (the C++
solver) and the corner models and K&C analyses are
:mod:`solvcon.multibody.suspension`.  :func:`load` imports them and reports
in words a user can act on when solvcon was built without the engine.
"""

import importlib
import os

__all__ = [
    'Engine',
    'EngineUnavailable',
    'load',
]


class EngineUnavailable(ImportError):
    """The engine or the suspension package cannot be imported."""


class Engine(object):
    """The imported engine and the suspension modules, by name.

    :ivar mbs: The :mod:`solvcon.multibody` module.
    """

    MODULES = ('alignment', 'axle', 'car', 'compliance', 'corner', 'kc',
               'longitudinal', 'metrics', 'model', 'reference', 'ride',
               'road', 'sensitivity', 'tuning', 'vehicle')

    def __init__(self, mbs, modules):
        self.mbs = mbs
        for name, module in modules.items():
            setattr(self, name, module)

    @property
    def data_dir(self):
        """The directory of the bundled corner models."""
        return os.path.join(os.path.dirname(self.model.__file__), 'data')


def load():
    """Import the engine and the suspension package.

    :return: The imported modules.
    :rtype: Engine
    :raises EngineUnavailable: With the reason, when either is missing.
    """
    try:
        mbs = importlib.import_module('solvcon.multibody')
    except ImportError as exc:
        raise EngineUnavailable(
            f"cannot import the multibody engine solvcon.multibody: "
            f"{exc}") from exc
    if not getattr(mbs, 'HAS_MULTIBODY', False):
        raise EngineUnavailable(
            "this solvcon build has no multibody engine "
            "(solvcon.multibody.HAS_MULTIBODY is false)")
    modules = {}
    for name in Engine.MODULES:
        try:
            modules[name] = importlib.import_module(
                f'solvcon.multibody.suspension.{name}')
        except ImportError as exc:
            raise EngineUnavailable(
                f"cannot import solvcon.multibody.suspension.{name}: "
                f"{exc}") from exc
    return Engine(mbs, modules)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
