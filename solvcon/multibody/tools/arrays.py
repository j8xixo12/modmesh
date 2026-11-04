# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Numpy views of the numeric results of the multibody engine.

The engine returns numeric results as lists, numpy arrays, or solvcon
``SimpleArray`` objects. A ``SimpleArray`` supports the buffer protocol, so
``numpy.asarray`` gives a view without a copy, but it does not support
partial indexing (``a[k]`` of a 2D array) or iteration. Callers convert a
result with these helpers before they index it.
"""

import numpy as np

__all__ = [
    'as_list',
    'result_arrays',
    'reaction_arrays',
]

_SCALAR = (str, bytes, bool, int, float)


def as_list(value):
    """Return an engine array result as nested Python lists.

    :param value: list, numpy array, or SimpleArray
    :return: nested lists of floats, the same shape as the result
    :rtype: list
    """
    return np.asarray(value).tolist()


def result_arrays(result):
    """Return a result dict with every array value as a numpy array.

    Scalar values (str, bool, int, float, None) are kept as they are.

    :param result: result dict of ``sweep``, ``evaluate_constraints``, or
        one entry of ``get_joint_reactions``
    :type result: dict
    :return: a new dict with numpy arrays in place of the array values
    :rtype: dict
    """
    return {key: (value if value is None or isinstance(value, _SCALAR)
                  else np.asarray(value))
            for key, value in result.items()}


def reaction_arrays(reactions):
    """Return the joint reactions with numpy arrays in every entry.

    :param reactions: result of ``get_joint_reactions``
    :type reactions: list
    :return: list of dicts converted by :func:`result_arrays`
    :rtype: list
    """
    return [result_arrays(reaction) for reaction in reactions]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
