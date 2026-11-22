# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Run a long session analysis in a child process, so the window stays live.

A :class:`Job` starts a fresh interpreter (no Qt state is copied), rebuilds
the session from the state the window has (the model file, the unsaved
model edits, the setup, the adjusters, the vehicle edits and the added
vehicle files), runs the session calls, and sends back their results and
warnings.  :meth:`Job.cancel` ends the process.

The child is started with :func:`solvcon.system.python_command`:
:mod:`multiprocessing` starts ``sys.executable -c``, which the pilot binary
with its embedded interpreter does not take.
"""

import os
import pickle
import subprocess
import sys
import threading
import traceback

from ... import system

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


#: The child takes the module search path of the parent from the request,
#: before it imports solvcon.
_BOOTSTRAP = (
    "import pickle, sys; "
    "request = pickle.load(sys.stdin.buffer); "
    "sys.path[:] = request['path']; "
    "from solvcon.apps.suspension import _jobs; "
    "_jobs.work(request['state'], request['calls'])")


def work(state, calls):
    """Run ``calls`` in the child and write the pickled result to stdout."""
    # The engine prints to stdout: keep the result channel apart.
    sys.stdout.flush()
    channel = os.dup(1)
    os.dup2(2, 1)
    try:
        sess = _rebuild(state)
        outs = []
        warnings = []
        for name, args, kwargs in calls:
            outs.append(getattr(sess, name)(*args, **kwargs))
            warnings.extend(sess.warnings)
        result = pickle.dumps(('done', outs, warnings))
    except Exception:
        result = pickle.dumps(('error', traceback.format_exc(), []))
    with os.fdopen(channel, 'wb') as fobj:
        fobj.write(result)


class Job(object):
    """One background run of session calls.

    :ivar calls: ``[(method name, args, kwargs)]`` of the session
    """

    def __init__(self, state, calls):
        self.calls = list(calls)
        request = pickle.dumps({'path': list(sys.path), 'state': state,
                                'calls': self.calls})
        self._process = subprocess.Popen(
            system.python_command('-c', _BOOTSTRAP),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._output = b''
        # A thread, so that a result larger than the pipe buffer does not
        # block the child and poll() does not block the window.
        self._reader = threading.Thread(target=self._exchange,
                                        args=(request,), daemon=True)
        self._reader.start()
        self.result = None

    def _exchange(self, request):
        self._output = self._process.communicate(request)[0]

    def poll(self):
        """Take the result when it is there.

        :return: None while running; else ``(status, outs, warnings)``
            with ``status`` ``'done'`` or ``'error'`` (``outs`` is then the
            traceback text)
        """
        if self.result is None and not self._reader.is_alive():
            try:
                self.result = pickle.loads(self._output)
            except (EOFError, pickle.UnpicklingError):
                self.result = (
                    'error', "the analysis process ended without a result "
                    f"(exit code {self._process.returncode})", [])
        return self.result

    def cancel(self):
        """End the process; the job has no result."""
        self._process.terminate()
        self._reader.join(timeout=5.0)
        self.result = ('cancelled', None, [])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
