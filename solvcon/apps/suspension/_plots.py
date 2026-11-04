# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The K&C curve plot: the pilot line plot with a dashed previous curve and the
shared cursor.

:class:`~solvcon.pilot.visual.LinePlotWidget` draws every series solid and
has no marker.  An adjuster change has to show the curve it replaced, and
the sweep slider has to show where the 3D pose sits on each curve, so the
plot here draws the series it is told are dashed with a dashed pen, and
paints the cursor (a vertical line and a dot on each curve) over the curves.
"""

import math

import numpy as np

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

from ... import pilot
from ...pilot import visual

__all__ = [
    'CURRENT_COLOR',
    'PREVIOUS_COLOR',
    'KCPlot',
]

CURRENT_COLOR = (31, 95, 168)
PREVIOUS_COLOR = (194, 87, 26)


class KCPlot(visual.LinePlotWidget):
    """One K&C quantity over the sweep, with units on both axes.

    :ivar cursor: The cursor abscissa, or None for no cursor.
    """

    def __init__(self, title, xlabel, ylabel, parent=None):
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel,
                         parent=parent)
        self.setMinimumSize(220, 180)
        self.cursor = None
        self._dashed = set()
        self._markers = set()

    def set_curves(self, curves):
        """Replace the curves with ``[(label, x, y, color, dashed)]``.

        The plot model cannot drop a series, so a fresh one takes the new
        curves; the view follows the data again.
        """
        self.model = pilot.RPlotModel()
        self._dashed = set()
        for label, xs, ys, color, dashed in curves:
            series = self.add_series(label, color=pilot.PlotColor(*color))
            # A column of a table is a strided view; the plot model takes
            # contiguous data only.
            series.set_data(visual.plot_array(np.ascontiguousarray(xs)),
                            visual.plot_array(np.ascontiguousarray(ys)))
            series.line_width = 1.5
            if dashed:
                self._dashed.add(label)
        self.refresh()

    def set_markers(self, labels):
        """Draw the series named in ``labels`` as dots without a line
        (measured points)."""
        self._markers = set(labels)
        self.update()

    def markers(self, label):
        """Whether the series named ``label`` is drawn as dots."""
        return label in self._markers

    def set_xlabel(self, text):
        self._xlabel = text
        self.update()

    def set_cursor(self, x):
        self.cursor = x
        self.update()

    def labels(self):
        return [self.model.series(it).label for it in range(self.model.size)]

    def dashed(self, label):
        """Whether the series named ``label`` is drawn dashed."""
        return label in self._dashed

    def data(self, label):
        """Return ``(x, y)`` lists of the series named ``label``."""
        for it in range(self.model.size):
            series = self.model.series(it)
            if series.label == label:
                return ([series.x(jt) for jt in range(series.size)],
                        [series.y(jt) for jt in range(series.size)])
        raise KeyError(label)

    def marker(self, label):
        """The ordinate of the series ``label`` at the cursor, linearly
        interpolated, or None outside it."""
        if self.cursor is None:
            return None
        xs, ys = self.data(label)
        for it in range(len(xs) - 1):
            x0, x1 = xs[it], xs[it + 1]
            if min(x0, x1) <= self.cursor <= max(x0, x1):
                if x1 == x0:
                    return ys[it]
                frac = (self.cursor - x0) / (x1 - x0)
                return ys[it] + frac * (ys[it + 1] - ys[it])
        return None

    def _draw_series(self, painter, rect, to_screen):
        painter.save()
        painter.setClipRect(rect)
        for it in range(self.model.size):
            series = self.model.series(it)
            points = self._drawn_points(series, rect)
            if len(points) < 2:
                continue
            color = series.color
            pen = QPen(QColor(color.r, color.g, color.b, color.a))
            pen.setWidthF(series.line_width)
            if series.label in self._markers:
                painter.setPen(pen)
                painter.setBrush(QColor(color.r, color.g, color.b))
                for x, y in points:
                    painter.drawEllipse(QPointF(to_screen(x, y)), 2.5, 2.5)
                continue
            if series.label in self._dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPolyline(
                QPolygonF([to_screen(x, y) for x, y in points]))
        painter.restore()

    def paintEvent(self, event):
        super().paintEvent(event)
        limits = self.limits()
        if self.cursor is None or limits is None:
            return
        if not limits.xmin <= self.cursor <= limits.xmax:
            return
        rect = self._lineplot_rect()
        to_screen = self._mapper(rect)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        ink = QColor(self.palette().windowText().color())
        ink.setAlpha(140)
        top = to_screen(self.cursor, limits.ymax)
        bottom = to_screen(self.cursor, limits.ymin)
        painter.setPen(QPen(ink, 1.0, Qt.DotLine))
        painter.drawLine(top, bottom)
        for it in range(self.model.size):
            series = self.model.series(it)
            value = self.marker(series.label)
            if value is None or not math.isfinite(value):
                continue
            color = series.color
            painter.setPen(QPen(QColor(color.r, color.g, color.b), 1.5))
            painter.setBrush(QColor(color.r, color.g, color.b))
            painter.drawEllipse(QPointF(to_screen(self.cursor, value)),
                                3.5, 3.5)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
