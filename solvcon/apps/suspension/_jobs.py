# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Run a long session analysis in a child process, so the window stays live.

The engine holds the Python thread while it solves, so a thread does not
help.  A :class:`Job` starts a fresh process (the ``spawn`` start method:
no Qt state is copied), rebuilds the session from the state the window
has (the model file, the unsaved model edits, the setup, the adjusters,
the vehicle edits and the added vehicle files),
runs the session calls, and sends back their results and warnings.
:meth:`Job.cancel` ends the process.
"""

import multiprocessing
import traceback

__all__ = [
    'Job',
    'session_state',
]


def session_state(session):
    """The state a child process needs to rebuild ``session``.

    :rtype: dict
    """
    return {
        'model_path': session.model_path,
        'model_raw': session.base_model.raw,
        'model_dirty': session.model_dirty,
        'setup': dict(session.setup),
        'adjusters': dict(session.adjusters),
        'vehicle_raw': session.vehicle_raw,
        'vehicle_dirty': set(session.vehicle_dirty),
        'extra_vehicles': list(session.extra_vehicles),
        'vehicle_stem': session.vehicle_stem,
    }


def _rebuild(state):
    from . import _engine
    from . import _session

    engine = _engine.load()
    sess = _session.SuspensionSession(engine, state['model_path'])
    sess.base_model = engine.model.CornerModel(state['model_raw'])
    sess.model_dirty = state['model_dirty']
    sess.setup = dict(state['setup'])
    sess.adjusters = dict(state['adjusters'])
    # The vehicle edits kept in the session and the added vehicle files.
    sess.vehicle_raw = state['vehicle_raw']
    sess.vehicle_dirty = set(state['vehicle_dirty'])
    sess.extra_vehicles = list(state['extra_vehicles'])
    sess.vehicle_stem = state['vehicle_stem']
    return sess


def _worker(conn, state, calls):
    try:
        sess = _rebuild(state)
        outs = []
        warnings = []
        for name, args, kwargs in calls:
            outs.append(getattr(sess, name)(*args, **kwargs))
            warnings.extend(sess.warnings)
        conn.send(('done', outs, warnings))
    except Exception:
        conn.send(('error', traceback.format_exc(), []))
    finally:
        conn.close()


class Job(object):
    """One background run of session calls.

    :ivar calls: ``[(method name, args, kwargs)]`` of the session
    """

    def __init__(self, state, calls):
        self.calls = list(calls)
        ctx = multiprocessing.get_context('spawn')
        self._conn, child = ctx.Pipe(duplex=False)
        self._process = ctx.Process(target=_worker,
                                    args=(child, state, self.calls),
                                    daemon=True)
        self._process.start()
        child.close()
        self.result = None

    def poll(self):
        """Take the result when it is there.

        :return: None while running; else ``(status, outs, warnings)``
            with ``status`` ``'done'`` or ``'error'`` (``outs`` is then the
            traceback text)
        """
        if self.result is not None:
            return self.result
        if self._conn.poll():
            try:
                self.result = self._conn.recv()
            except EOFError:
                self.result = ('error', "the analysis process ended "
                               "without a result", [])
            self._process.join(timeout=5.0)
        elif not self._process.is_alive():
            self.result = ('error', "the analysis process ended without a "
                           f"result (exit code {self._process.exitcode})",
                           [])
        return self.result

    def cancel(self):
        """End the process; the job has no result."""
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self.result = ('cancelled', None, [])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
