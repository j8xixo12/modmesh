# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Result handling of the suspension app, free of Qt: the run list, the
measured curves read from a CSV file, the summary of a curve, the geometry
overlays of the 3D view, and the PDF report.

The suspension package computes in SI units; everything here is in the
display units of :mod:`._session` (mm, deg, N).
"""

import csv
import math
import re

import numpy as np

__all__ = [  # noqa: F822
    'MeasuredSet',
    'RUN_COLORS',
    'Run',
    'overlay_segments',
    'read_measured_csv',
    'summary',
    'write_report',
]

MM = 1.e3
DEG = 180.0 / math.pi

#: The colors of the runs, in the order runs are made.  The blue of the
#: current curve is left out, so a run never looks like the current one.
RUN_COLORS = ((194, 87, 26), (46, 139, 87), (148, 103, 189), (214, 39, 40),
              (23, 160, 180), (140, 86, 75), (120, 120, 30), (227, 119, 194))


class Run(object):
    """One named sweep result of the run list.

    :ivar name: The name shown in the run list and the plot legend.
    :ivar result: The :class:`~._session.SweepResult`.
    :ivar color: ``(r, g, b)`` of its curves.
    :ivar visible: Whether the plots draw it.
    """

    def __init__(self, name, result, color, ident=''):
        self.name = name
        self.result = result
        self.color = color
        self.visible = True
        self.ident = ident

    @property
    def label(self):
        """The short plot label: the run number, and a renamed run's name
        cut to a few characters (the run list shows the full name)."""
        if not self.ident or self.name.startswith(self.ident):
            return self.ident or self.name
        short = self.name if len(self.name) <= 14 else self.name[:13] + "~"
        return f"{self.ident} {short}"


def _column_key(text):
    """The column name in a canonical form: lower case, units in brackets
    or after an underscore, spaces and punctuation dropped."""
    text = text.strip().lower()
    text = text.replace('/', '_per_').replace('+', '')
    return re.sub(r'[^a-z0-9_]+', '', text.replace(' ', '_'))


def _column_names(curve):
    """The names a CSV column of ``curve`` may have: the export label, the
    key, and the key with the unit."""
    unit = curve.unit if '-' != curve.unit else ''
    names = {_column_key(curve.label), _column_key(curve.key)}
    if unit:
        names.add(_column_key(f"{curve.key}_{unit}"))
    return names


class MeasuredSet(object):
    """Measured (rig or test) curves of one sweep kind.

    :ivar name: The file name, the legend label of the curves.
    :ivar kind: The name of the sweep kind the abscissa matches.
    :ivar x: The abscissa, in mm.
    :ivar curves: ``{curve key: ordinates in display units}``.
    """

    def __init__(self, name, kind, x, curves):
        self.name = name
        self.kind = kind
        self.x = np.asarray(x, dtype='float64')
        self.curves = {key: np.asarray(values, dtype='float64')
                       for key, values in curves.items()}


def read_measured_csv(path, kinds, name=None):
    """Read measured curves from a CSV file with a header row.

    Lines starting with ``#`` are comments.  One column is the abscissa of
    a sweep kind (its export label, like ``Wheel travel (+ bump) [mm]``, or
    ``travel_mm``); the other known columns are curves of that kind (export
    labels, keys, or ``key_unit`` like ``camber_deg``).  Unknown columns are
    reported, not read.

    :param kinds: ``{name: SweepKind}``.
    :return: ``(MeasuredSet, unknown column names)``.
    :raises ValueError: Without a known abscissa, without a known curve, or
        for a cell that is not a number.
    """
    with open(path, newline='', encoding='utf-8', errors='replace') as fobj:
        rows = [row for row in csv.reader(
            line for line in fobj if line.strip()
            and not line.lstrip().startswith('#'))]
    if not rows:
        raise ValueError(f"{path}: no header row")
    header = [_column_key(h) for h in rows[0]]
    found = None
    for kind in kinds.values():
        names = _column_names(kind.x)
        for index, column in enumerate(header):
            if column in names:
                found = (kind, index)
                break
        if found:
            break
    if found is None:
        raise ValueError(
            f"{path}: no sweep column; the header needs one of "
            + ", ".join(sorted(f"'{k.x.label}'" for k in kinds.values())))
    kind, xcol = found
    columns = {}
    unknown = []
    for index, column in enumerate(header):
        if index == xcol:
            continue
        for curve in kind.curves:
            if column in _column_names(curve):
                columns[curve.key] = index
                break
        else:
            unknown.append(rows[0][index].strip())
    if not columns:
        raise ValueError(f"{path}: no known {kind.name} curve column")
    data = []
    for number, row in enumerate(rows[1:], start=2):
        try:
            data.append([float(cell) for cell in row])
        except ValueError:
            raise ValueError(f"{path}: data row {number} has a cell that is "
                             f"not a number: {row}") from None
    data = np.array(data, dtype='float64').reshape(-1, len(header))
    curves = {key: data[:, index] for key, index in columns.items()}
    label = name or str(path).replace('\\', '/').rsplit('/', 1)[-1]
    return MeasuredSet(label, kind.name, data[:, xcol], curves), unknown


def summary(xs, ys):
    """The summary of a curve at design (x = 0): the value (linear
    interpolation), the gradient (central difference of the two points
    around x = 0, per unit of x), the minimum, and the maximum.

    :return: dict with ``value``, ``gradient`` (None when x = 0 is not
        inside the curve), ``min``, ``max``; all None for an all-NaN
        curve.
    """
    xs = np.asarray(xs, dtype='float64')
    ys = np.asarray(ys, dtype='float64')
    defined = ys.size and not np.isnan(ys).all()
    out = {'value': None, 'gradient': None,
           'min': float(np.nanmin(ys)) if defined else None,
           'max': float(np.nanmax(ys)) if defined else None}
    if not defined:
        # The quantity does not exist for the model (a steering axis
        # without a rack or ball joints): nothing to summarize.
        return out
    if xs.size < 2:
        return out
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if not xs[0] <= 0.0 <= xs[-1]:
        return out
    out['value'] = float(np.interp(0.0, xs, ys))
    hit = np.flatnonzero(xs == 0.0)
    if hit.size and 0 < hit[0] < xs.size - 1:
        lo, hi = hit[0] - 1, hit[0] + 1
    else:
        hi = int(np.searchsorted(xs, 0.0))
        hi = min(max(hi, 1), xs.size - 1)
        lo = hi - 1
    out['gradient'] = float((ys[hi] - ys[lo]) / (xs[hi] - xs[lo]))
    return out


#: Lines toward an instant center farther than this (mm) are drawn this
#: long, and the center itself is left out.
IC_DRAW_LIMIT = 1500.0
CROSS_HALF = 20.0


def _cross(point, half=CROSS_HALF):
    ends = []
    for axis in np.eye(3):
        ends += [point - half * axis, point + half * axis]
    return ends


def _toward(start, end, limit=IC_DRAW_LIMIT):
    """Segment ends from ``start`` toward ``end``, cut at ``limit``, with a
    cross at ``end`` when it is within the limit."""
    vector = end - start
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 0.0:
        return []
    if length > limit:
        return [start, start + vector * (limit / length)]
    return [start, end] + _cross(end)


def overlay_segments(data, index):
    """The geometry overlays of a bump sweep at point ``index``, in mm:
    ``{name: segment ends}`` for the kingpin axis, the lines from the
    contact patch toward the front- and side-view instant centers, the roll
    center, and the contact patch path over the sweep.  A sweep without
    the geometry keys gives no overlay."""
    needed = ('contact_patch', 'fvic', 'svic', 'roll_center_point',
              'kingpin_point', 'kingpin_dir')
    if any(key not in data for key in needed):
        return {}
    cp = np.asarray(data['contact_patch'][index], dtype='float64') * MM
    out = {}
    point = np.asarray(data['kingpin_point'][index], dtype='float64') * MM
    axis = np.asarray(data['kingpin_dir'][index], dtype='float64')
    if np.all(np.isfinite(point)) and np.all(np.isfinite(axis)):
        out['kingpin'] = [point - 50.0 * axis, point + 650.0 * axis]
    ends = []
    for key in ('fvic', 'svic'):
        center = np.asarray(data[key][index], dtype='float64') * MM
        if np.all(np.isfinite(center)):
            ends += _toward(cp, center)
    out['instant_centers'] = ends
    rc = np.asarray(data['roll_center_point'][index], dtype='float64') * MM
    out['roll_center'] = (_cross(rc, 30.0) if np.all(np.isfinite(rc))
                          else [])
    path = np.asarray(data['contact_patch'], dtype='float64') * MM
    ends = []
    for it in range(len(path) - 1):
        ends += [path[it], path[it + 1]]
    out['patch_path'] = ends
    return out


def write_report(path, sheet, pages, table):
    """Write the PDF report: the setup sheet, the plot pages, and the
    summary table.

    :param sheet: ``[(title, [(field, text)])]`` sections of the setup
        sheet.
    :param pages: ``[(page title, [plot])]`` where a plot is ``(title,
        xlabel, ylabel, [(label, xs, ys, color, dashed, markers)])``.
    :param table: ``(columns, rows)`` of the summary table.
    :return: The number of pages written.
    """
    import matplotlib
    matplotlib.use('Agg', force=False)
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    count = 0
    with PdfPages(path) as pdf:
        fig = Figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.96, "Setup sheet", fontsize=16, weight='bold')
        y = 0.93
        for title, fields in sheet:
            fig.text(0.07, y, title, fontsize=11, weight='bold')
            y -= 0.022
            for field, text in fields:
                fig.text(0.09, y, field, fontsize=8, family='monospace')
                fig.text(0.42, y, text, fontsize=8, family='monospace')
                y -= 0.016
                if y < 0.04:
                    pdf.savefig(fig)
                    count += 1
                    fig = Figure(figsize=(8.27, 11.69))
                    y = 0.95
            y -= 0.01
        pdf.savefig(fig)
        count += 1
        for title, plots in pages:
            fig = Figure(figsize=(11.69, 8.27))
            fig.suptitle(title, fontsize=13)
            ncol = 3
            nrow = max(1, int(math.ceil(len(plots) / float(ncol))))
            for it, (ptitle, xlabel, ylabel, curves) in enumerate(plots):
                ax = fig.add_subplot(nrow, ncol, it + 1)
                for label, xs, ys, color, dashed, markers in curves:
                    rgb = tuple(c / 255.0 for c in color)
                    if markers:
                        ax.plot(xs, ys, 'o', ms=3, color=rgb, label=label)
                    else:
                        ax.plot(xs, ys, '--' if dashed else '-', color=rgb,
                                lw=1.2, label=label)
                ax.set_title(ptitle, fontsize=9)
                ax.set_xlabel(xlabel, fontsize=7)
                ax.set_ylabel(ylabel, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(alpha=0.3)
                if it == 0:
                    ax.legend(fontsize=5)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            count += 1
        columns, rows = table
        fig = Figure(figsize=(11.69, 8.27))
        fig.suptitle("Summary at design (x = 0)", fontsize=13)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        if rows:
            cell = ax.table(cellText=rows, colLabels=columns, loc='upper '
                            'center', cellLoc='left')
            cell.auto_set_font_size(False)
            cell.set_fontsize(7)
            cell.scale(1.0, 1.3)
        pdf.savefig(fig)
        count += 1
    return count

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
