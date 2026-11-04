# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Target-driven alignment: solve the adjusters for static targets (roadmap
P11.6).

Each target maps to one adjuster: camber -> camber_shim, toe -> toe_link,
caster -> caster_shim. All targeted adjusters are solved together by Newton
iteration with a central-difference Jacobian of the static alignment at
design travel 0 on the K&C rig. The adjusters that have no target keep the
values of the input model.

An adjuster has a physical range (default +/- 25 mm). The kinematics can
reach extreme alignments with non-physical adjuster values (for example
camber -30 deg with a 0.2 m shim on the shipped model), so the range is what
makes a target unreachable.
"""

import copy

import numpy as np

from . import kc

ADJUSTER_OF = {"camber": "camber_shim", "toe": "toe_link",
               "caster": "caster_shim"}
ORDER = ("camber", "toe", "caster")


def static_alignment(mbs, model):
    """Static camber, toe, and caster at design travel 0.

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model
    :type model: suspension.model.CornerModel
    :return: (dict camber, toe, caster in rad, or None; message)
    :rtype: tuple
    """
    try:
        r = kc.bump_sweep(mbs, model, [0.0])
    except RuntimeError as e:  # the rig does not assemble
        return None, str(e)
    if not r["completed"]:
        return None, r["message"]
    return {k: float(r[k][0]) for k in ORDER}, ""


def _fmt(adjust):
    return ", ".join(f"{k} = {v * 1e3:.4f} mm" for k, v in adjust.items())


def solve_alignment(mbs, model, targets, tol=1e-10, max_iter=20,
                    max_adjust=0.025, h=1e-5):
    """Solve the adjusters so that the static alignment meets the targets.

    The input model is not modified (the solve works on a copy).

    :param mbs: compiled solver module
    :type mbs: module
    :param model: corner model; its current adjusters are the start point
    :type model: suspension.model.CornerModel
    :param targets: any subset of "camber", "toe", "caster" (rad)
    :type targets: dict
    :param tol: convergence bound on max |achieved - target| (rad)
    :type tol: float
    :param max_iter: maximum Newton iterations
    :type max_iter: int
    :param max_adjust: physical range of each adjuster, +/- (m)
    :type max_adjust: float
    :param h: adjuster step of the finite-difference Jacobian (m)
    :type h: float
    :return: dict with "converged" (bool), "adjusters" ({name: m}; the
        start values if not converged), "achieved" ({camber, toe, caster}
        rad at the returned adjusters, or at the last iterate if not
        converged; None if the start setup does not assemble),
        "iterations", "message", and "last_adjusters" (the last iterate)
    :rtype: dict
    :raises ValueError: for an unknown target name
    """
    unknown = sorted(set(targets) - set(ADJUSTER_OF))
    if unknown:
        raise ValueError(f"unknown alignment target(s): {', '.join(unknown)}; "
                         f"known: {', '.join(ORDER)}")
    keys = [k for k in ORDER if k in targets]
    names = [ADJUSTER_OF[k] for k in keys]
    missing = [n for n in names if n not in model.adjusters]
    if missing:
        raise ValueError(f"model '{model.name}' has no adjuster for the "
                         f"target(s): {', '.join(missing)}")
    goal = np.array([float(targets[k]) for k in keys])
    start = dict(model.adjust)
    work = copy.deepcopy(model)

    def evaluate(x):
        adj = dict(start)
        adj.update({n: float(v) for n, v in zip(names, x)})
        work.set_adjusters(**adj)
        al, msg = static_alignment(mbs, work)
        if al is None:
            return None, None, msg
        return np.array([al[k] for k in keys]), al, ""

    def result(converged, x, achieved, it, message):
        last = dict(start)
        last.update({n: float(v) for n, v in zip(names, x)})
        return {"converged": converged,
                "adjusters": last if converged else dict(start),
                "achieved": achieved, "iterations": it, "message": message,
                "last_adjusters": last}

    x = np.array([start[n] for n in names], float)
    f, achieved, msg = evaluate(x)
    if f is None:
        return result(False, x, None, 0,
                      f"the start setup does not assemble: {msg}")
    if not keys:
        return result(True, x, achieved, 0, "no targets")
    r = f - goal
    for it in range(max_iter + 1):
        if np.max(np.abs(r)) <= tol:
            return result(True, x, achieved, it,
                          f"converged in {it} iterations")
        if it == max_iter:
            break
        jac = np.zeros((len(keys), len(keys)))
        for j in range(len(keys)):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            fp, _, mp = evaluate(xp)
            fm, _, mm = evaluate(xm)
            if fp is None or fm is None:
                return result(False, x, achieved, it,
                              f"Jacobian step of {names[j]} does not "
                              f"assemble: {mp or mm}")
            jac[:, j] = (fp - fm) / (2 * h)
        try:
            dx = -np.linalg.solve(jac, r)
            if not np.all(np.isfinite(dx)):
                raise np.linalg.LinAlgError("non-finite step")
        except np.linalg.LinAlgError:
            return result(False, x, achieved, it,
                          "singular Jacobian: the adjusters "
                          f"({', '.join(names)}) cannot "
                          f"set the targets ({', '.join(keys)}) independently")
        # A Newton step out of the physical range is cut at the range. If an
        # adjuster already sits at its range limit and Newton pushes it
        # further, the target is not reachable.
        out_of_range = np.abs(x + dx) > max_adjust
        at_limit = np.abs(np.abs(x) - max_adjust) <= 1e-12 * max_adjust
        pushing = out_of_range & at_limit & (np.sign(dx) == np.sign(x))
        if np.any(pushing):
            j = int(np.flatnonzero(pushing)[0])
            return result(False, x, achieved, it,
                          f"target {keys[j]} not reachable: {names[j]} needs "
                          "more than its "
                          f"range +/-{max_adjust * 1e3:.1f} mm "
                          f"({_fmt(dict(zip(names, x)))})")
        if np.any(out_of_range):
            room = np.where(dx > 0, max_adjust - x, -max_adjust - x)
            scale = np.min(np.where(out_of_range, room / dx, 1.0))
            dx = dx * max(scale, 0.0)
            x_try = np.clip(x + dx, -max_adjust, max_adjust)
            # Snap the limiting adjusters exactly onto the range limit.
            x_try = np.where(np.abs(np.abs(x_try) - max_adjust)
                             <= 1e-12 * max_adjust,
                             np.sign(x_try) * max_adjust, x_try)
            dx = x_try - x
        # Backtracking: halve the step while the rig does not assemble or the
        # residual does not decrease.
        for _ in range(10):
            f_new, al_new, msg = evaluate(x + dx)
            if (f_new is not None
                    and np.max(np.abs(f_new - goal)) < np.max(np.abs(r))):
                break
            dx = 0.5 * dx
        else:
            cause = f"the rig does not assemble: {msg}" if f_new is None else (
                "the residual does not decrease")
            return result(False, x, achieved, it + 1,
                          "line search failed at "
                          f"{_fmt(dict(zip(names, x)))}: {cause}")
        x = x + dx
        r = f_new - goal
        achieved = al_new
    worst = keys[int(np.argmax(np.abs(r)))]
    return result(False, x, achieved, max_iter,
                  f"not converged after {max_iter} iterations: {worst} misses "
                  "its target "
                  f"by {np.max(np.abs(r)):.3g} rad "
                  f"({_fmt(dict(zip(names, x)))})")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
