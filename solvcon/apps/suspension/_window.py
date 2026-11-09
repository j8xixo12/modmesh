# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The analysis panel of the suspension app: the Sweep, Statics, Dynamics, and
Modal tabs beside the warnings.

Every control reports through a callback the app sets, and every readout is
filled by the app from the session, so the panel holds no result.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDoubleSpinBox, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QMenu,
                               QPushButton, QScrollArea, QSlider,
                               QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from . import _panel
from . import _plots
from . import _session

__all__ = [
    'AnalysisPanel',
    'scrollable',
]


def _spin(lo, hi, value, suffix, decimals=1, step=1.0):
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    return spin


def _clear_grid(grid):
    """Remove and delete every widget of a grid layout; a removed widget is
    hidden at once (the deletion waits for the event loop)."""
    while grid.count():
        item = grid.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()


def _button(text, func):
    button = QPushButton(text)
    button.clicked.connect(lambda: func())
    return button


def _swatch(color):
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor(*color))
    return QIcon(pixmap)


def _summary_text(stats, unit, xunit):
    """One line of the summary of a curve at design."""
    def num(value):
        return "-" if value is None else f"{value:+.4g}"
    return (f"at 0: {num(stats['value'])} {unit} | gradient "
            f"{num(stats['gradient'])} {unit}/{xunit} | range "
            f"[{num(stats['min'])}, {num(stats['max'])}]")


#: The height (px) below which a page scrolls instead of shrinking its
#: plots and tables further: short enough for a laptop dock, tall enough
#: that a plot stays readable.
PAGE_MIN_HEIGHT = 420


def scrollable(widget, min_height=PAGE_MIN_HEIGHT):
    """Put ``widget`` in a scroll area: a short dock scrolls the content
    instead of squeezing it below a readable size."""
    widget.setMinimumHeight(max(min_height,
                                widget.minimumSizeHint().height()))
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setWidget(widget)
    return scroll


class PlotCell(QWidget):
    """A K&C plot with its summary lines under it."""

    def __init__(self, plot, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plot = plot
        self.summary = QLabel("")
        self.summary.setStyleSheet("QLabel { font-size: 8pt; color: #303040;"
                                   " }")
        plot.setMinimumSize(240, 170)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(plot, 1)
        layout.addWidget(self.summary)


class RunList(QWidget):
    """The run list: a check box shows a run in the plots; the buttons
    rename, delete, or mark the baseline.

    :ivar changed: Called with ``(action, index, value)`` when set:
        ``visible`` (bool), ``rename`` (str), ``delete``, ``baseline``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.changed = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Runs"))
        self.tree = QTreeWidget()
        self.tree.setRootIsDecorated(False)
        self.tree.setHeaderLabels(["run"])
        self.tree.itemChanged.connect(self._on_item)
        layout.addWidget(self.tree, 1)
        row = QHBoxLayout()
        for text, action in (("Baseline", 'baseline'),
                             ("Delete", 'delete')):
            row.addWidget(_button(text, lambda a=action: self._emit(a)))
        layout.addLayout(row)
        self._filling = False

    def show_runs(self, runs, baseline):
        self._filling = True
        self.tree.clear()
        for run in runs:
            text = run.name + ("  [baseline]" if run is baseline else "")
            item = QTreeWidgetItem([text])
            item.setIcon(0, _swatch(run.color))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable
                          | Qt.ItemIsEditable)
            item.setCheckState(0, Qt.Checked if run.visible
                               else Qt.Unchecked)
            item.setData(0, Qt.UserRole, run.name)
            self.tree.addTopLevelItem(item)
        self._filling = False

    def _emit(self, action, index=None, value=None):
        if index is None:
            item = self.tree.currentItem()
            if item is None:
                return
            index = self.tree.indexOfTopLevelItem(item)
        if self.changed is not None:
            self.changed(action, index, value)

    def _on_item(self, item, column):
        if self._filling:
            return
        index = self.tree.indexOfTopLevelItem(item)
        name = item.data(0, Qt.UserRole)
        text = item.text(0).replace("  [baseline]", "")
        if text != name:
            self._emit('rename', index, text)
        else:
            self._emit('visible', index, Qt.Checked == item.checkState(0))


class SweepTab(QWidget):
    """The K&C sweep: its kind and range, the slider, and the curves."""

    COLUMNS = 3

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        self._handlers = handlers
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.kind = QComboBox()
        for name in _session.SWEEPS:
            self.kind.addItem(name, name)
        self.kind.currentIndexChanged.connect(
            lambda _: handlers['kind'](self.kind.currentData()))
        self.lo = _spin(-500.0, 500.0, -40.0, " mm")
        self.hi = _spin(-500.0, 500.0, 40.0, " mm")
        self.count = QSpinBox()
        self.count.setRange(2, _session.SuspensionSession.MAX_POINTS)
        self.count.setValue(33)
        self.count.setSuffix(" points")
        self.run = _button("Run sweep", self._on_run)
        self.export = _button("Export CSV...", handlers['export'])
        self.forces = QCheckBox("Force arrows")
        self.forces.toggled.connect(lambda on: handlers['forces'](on))
        self.measured = _button("Import measured...", handlers['measured'])
        self.report = _button("Report PDF...", handlers['report'])
        self.layout_button = QToolButton()
        self.layout_button.setText("Plots")
        self.layout_button.setPopupMode(QToolButton.InstantPopup)
        self.layout_menu = QMenu(self)
        self.layout_button.setMenu(self.layout_menu)
        self.overlay_button = QToolButton()
        self.overlay_button.setText("3D overlays")
        self.overlay_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        self.overlay_actions = {}
        for key, text in (('kingpin', "Kingpin axis"),
                          ('instant_centers', "Instant centers (front, "
                                              "side view)"),
                          ('roll_center', "Roll center"),
                          ('patch_path', "Contact patch path")):
            action = menu.addAction(text)
            action.setCheckable(True)
            action.toggled.connect(
                lambda on, k=key: handlers['overlay'](k, on))
            self.overlay_actions[key] = action
        self.overlay_button.setMenu(menu)
        for label, widget in (("Sweep", self.kind), ("from", self.lo),
                              ("to", self.hi), (None, self.count),
                              (None, self.run), (None, self.export),
                              (None, self.measured), (None, self.report),
                              (None, self.layout_button),
                              (None, self.overlay_button),
                              (None, self.forces)):
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row)

        cursor = QHBoxLayout()
        self.step_back = QToolButton()
        self.step_back.setText("|<")
        self.step_back.setToolTip("Previous point")
        self.step_back.clicked.connect(lambda: handlers['step'](-1))
        self.play = QToolButton()
        self.play.setText("Play")
        self.play.setCheckable(True)
        self.play.setToolTip("Play the sweep through the 3D view and the "
                             "plots")
        self.play.toggled.connect(lambda on: handlers['play'](on))
        self.step_forward = QToolButton()
        self.step_forward.setText(">|")
        self.step_forward.setToolTip("Next point")
        self.step_forward.clicked.connect(lambda: handlers['step'](1))
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(True)
        self.speed = QComboBox()
        for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed.addItem(f"{factor:g}x", factor)
        self.speed.setCurrentIndex(2)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(lambda v: handlers['cursor'](v))
        self.cursor_label = QLabel("")
        self.cursor_label.setMinimumWidth(260)
        cursor.addWidget(QLabel("Cursor"))
        for widget in (self.step_back, self.play, self.step_forward,
                       self.loop, self.speed):
            cursor.addWidget(widget)
        cursor.addWidget(self.slider, 1)
        cursor.addWidget(self.cursor_label)
        layout.addLayout(cursor)

        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        split = QSplitter(Qt.Horizontal)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        # The plots keep their minimum size and scroll when the dock is
        # short, instead of overlapping.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)
        self.runs = RunList()
        self.runs.setMinimumWidth(180)
        self.runs.setMaximumWidth(340)
        scroll.setMinimumWidth(500)
        split.addWidget(scroll)
        split.addWidget(self.runs)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.plots = {}
        self.cells = {}
        self._kind = None
        self._keys = None

    def _on_run(self):
        self._handlers['range'](self.lo.value(), self.hi.value(),
                                self.count.value())

    def show_kinds(self, kinds):
        """Enable only the sweep kinds the model can run."""
        model = self.kind.model()
        for it in range(self.kind.count()):
            item = model.item(it)
            item.setEnabled(self.kind.itemData(it) in kinds)

    def show_range(self, kind, lo, hi, count):
        for widget, value in ((self.lo, lo), (self.hi, hi),
                              (self.count, count)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        index = self.kind.findData(kind)
        self.kind.blockSignals(True)
        self.kind.setCurrentIndex(index)
        self.kind.blockSignals(False)

    def set_kind(self, kind, keys=None):
        """Lay out one plot per curve of the sweep ``kind`` that the
        layout ``keys`` (all by default) holds."""
        keys = [c.key for c in kind.curves] if keys is None else list(keys)
        if self._kind is kind and self._keys == keys:
            return
        for cell in self.cells.values():
            self.grid.removeWidget(cell)
            cell.deleteLater()
        self.plots = {}
        self.cells = {}
        shown = [c for c in kind.curves if c.key in keys]
        # The plot titles name the quantity; the axis labels carry the unit.
        for it, curve in enumerate(shown):
            plot = _plots.KCPlot(curve.title, kind.x.label, curve.unit)
            cell = PlotCell(plot)
            self.grid.addWidget(cell, it // self.COLUMNS, it % self.COLUMNS)
            self.plots[curve.key] = plot
            self.cells[curve.key] = cell
        self._kind = kind
        self._keys = keys
        self.layout_menu.clear()
        for curve in kind.curves:
            action = self.layout_menu.addAction(curve.title)
            action.setCheckable(True)
            action.setChecked(curve.key in keys)
            action.toggled.connect(
                lambda on, k=curve.key: self._handlers['layout'](
                    'toggle', k, on))
        self.layout_menu.addSeparator()
        self.layout_menu.addAction("Save layout...").triggered.connect(
            lambda: self._handlers['layout']('save', None, None))
        self.layout_menu.addAction("Load layout...").triggered.connect(
            lambda: self._handlers['layout']('load', None, None))

    def show_summaries(self, rows):
        """Write the summary lines under each plot: ``{key: [text]}``."""
        for key, cell in self.cells.items():
            cell.summary.setText("\n".join(rows.get(key, [])))

    def show_cursor(self, index, count, text):
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, count - 1))
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.cursor_label.setText(text)


class StaticsTab(QWidget):
    """A load at the contact patch of the compliant rig."""

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.force = [_spin(-20000.0, 20000.0, value, " N", decimals=0,
                            step=100.0)
                      for value in (0.0, 1000.0, 0.0)]
        for label, spin in zip(("Fx", "Fy", "Fz"), self.force):
            row.addWidget(QLabel(label))
            row.addWidget(spin)
        self.run = _button("Apply load", lambda: handlers['statics'](
            [spin.value() for spin in self.force]))
        row.addWidget(self.run)
        row.addStretch(1)
        layout.addWidget(QLabel(
            "Force at the contact patch (global, N); links on inboard "
            "bushings; wheel center height and rack held."))
        layout.addLayout(row)
        self.status = QLabel("")
        layout.addWidget(self.status)
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(
            ["load [N]", "toe change [deg]", "camber change [deg]",
             "patch dx [mm]", "patch dy [mm]", "patch dz [mm]"])
        layout.addWidget(self.table, 1)

        ramp = QGroupBox("Compliance sweep: ramp one load from -max to +max")
        ramp_layout = QVBoxLayout(ramp)
        row = QHBoxLayout()
        self.case = QComboBox()
        for key, (title, _) in _session.SuspensionSession.COMPLIANCE_CASES\
                .items():
            self.case.addItem(title, key)
        self.fmax = _spin(1.0, 50000.0, 2000.0, " N / N m", decimals=0,
                          step=100.0)
        self.points = QSpinBox()
        self.points.setRange(3, 41)
        self.points.setSingleStep(2)
        self.points.setValue(11)
        self.points.setSuffix(" points (odd)")
        self.ramp_run = _button("Run sweep", lambda: handlers['compliance'](
            self.case.currentData(), self.fmax.value(), self.points.value()))
        for label, widget in (("Load", self.case), ("max", self.fmax),
                              (None, self.points), (None, self.ramp_run)):
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addStretch(1)
        ramp_layout.addLayout(row)
        self.ramp_status = QLabel("")
        self.replay = ReplayBar('statics', handlers)
        self.ramp_status.setWordWrap(True)
        self.ramp_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ramp_layout.addWidget(self.ramp_status)
        ramp_layout.addWidget(self.replay)
        plots = QHBoxLayout()
        self.ramp_plots = {}
        for key, title, unit in (('toe', "Toe change", 'deg'),
                                 ('camber', "Camber change", 'deg'),
                                 ('patch', "Contact patch displacement",
                                  'mm')):
            plot = _plots.KCPlot(title, "Load", unit)
            plots.addWidget(plot)
            self.ramp_plots[key] = plot
        ramp_layout.addLayout(plots, 1)
        layout.addWidget(ramp, 2)

    def show_ramp(self, out, unit):
        """Draw a compliance sweep result."""
        load = list(out['load'])
        for key in ('toe', 'camber'):
            self.ramp_plots[key].set_curves([
                (key, load, list(out[key]), _plots.CURRENT_COLOR, False)])
        patch = out['patch']
        colors = ((200, 40, 40), (40, 150, 40), (40, 60, 200))
        self.ramp_plots['patch'].set_curves([
            (f"d{axis}", load, list(patch[:, it]), colors[it], False)
            for it, axis in enumerate("xyz")])
        grad = out['gradient']
        per = 'kN m' if 'N m' == unit else 'kN'
        self.ramp_status.setText(
            f"gradient at zero load: toe {grad['toe']:+.5f} deg/{per}, "
            f"camber {grad['camber']:+.5f} deg/{per}, patch "
            f"({grad['patch_x']:+.4f}, {grad['patch_y']:+.4f}, "
            f"{grad['patch_z']:+.4f}) mm/{per}"
            + ("" if out['completed'] else f"; stopped: {out['message']}")
            + ".  Parallel and opposed axle loads give the same corner "
              "response here: no compliance couples the corners.")

    def clear(self):
        """Drop the results of another model."""
        self.table.clear()
        self.status.setText("")
        self.ramp_status.setText("")
        for plot in self.ramp_plots.values():
            plot.set_curves([])

    def add_row(self, result):
        fx, fy, fz = result['force']
        patch = result['patch']
        self.table.addTopLevelItem(QTreeWidgetItem([
            f"({fx:.0f}, {fy:.0f}, {fz:.0f})", f"{result['toe']:+.4f}",
            f"{result['camber']:+.4f}", f"{patch[0]:+.4f}",
            f"{patch[1]:+.4f}", f"{patch[2]:+.4f}"]))


class DynamicsTab(QWidget):
    """The quarter car driving over a road (or on a shaker rig), marched by
    the app's timer."""

    #: The parameter fields: key, label, unit, range, step, decimals.
    FIELDS = (
        ('speed', "speed", " km/h", (1.0, 300.0), 5.0, 1),
        ('height', "height", " mm", (-200.0, 200.0), 5.0, 1),
        ('depth', "depth", " mm", (1.0, 300.0), 5.0, 1),
        ('length', "length", " mm", (5.0, 5000.0), 10.0, 0),
        ('amplitude', "amplitude", " mm", (0.1, 100.0), 1.0, 1),
        ('wavelength', "wavelength", " mm", (100.0, 100000.0), 500.0, 0),
        ('frequency', "frequency", " Hz", (0.05, 50.0), 0.5, 2),
        ('duration', "duration", " s", (0.05, 30.0), 0.25, 2),
    )

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.mass = _spin(10.0, 2000.0, 200.0, " kg", decimals=0, step=10.0)
        row.addWidget(QLabel("Sprung mass"))
        row.addWidget(self.mass)
        self.kind = QComboBox()
        for key, (title, _) in _session.ROAD_INPUTS.items():
            self.kind.addItem(title, key)
        self.kind.currentIndexChanged.connect(lambda _: self._show_fields())
        row.addWidget(QLabel("Road"))
        row.addWidget(self.kind)
        self.fields = {}
        self.labels = {}
        for key, label, unit, (lo, hi), step, decimals in self.FIELDS:
            spin = _spin(lo, hi, lo, unit, decimals=decimals, step=step)
            text = QLabel(label)
            row.addWidget(text)
            row.addWidget(spin)
            self.fields[key] = spin
            self.labels[key] = text
        self.csv_button = _button("Profile CSV...", handlers['road_csv'])
        self.csv_path = ''
        row.addWidget(self.csv_button)
        self.run = _button("Run", handlers['dynamics'])
        self.cancel = _button("Cancel", handlers['cancel'])
        self.cancel.setEnabled(False)
        row.addWidget(self.run)
        row.addWidget(self.cancel)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        replay = QHBoxLayout()
        self.replay = QSlider(Qt.Horizontal)
        self.replay.setEnabled(False)
        self.replay.setToolTip("Replay the run in the 3D view")
        self.replay.valueChanged.connect(lambda v: handlers['replay'](v))
        self.replay_label = QLabel("")
        self.replay_label.setMinimumWidth(160)
        replay.addWidget(QLabel("Replay"))
        replay.addWidget(self.replay, 1)
        replay.addWidget(self.replay_label)
        layout.addLayout(replay)
        plots = QHBoxLayout()
        self.plot = _plots.KCPlot("Heights from the static equilibrium",
                                  "Time [s]", "mm")
        self.force_plot = _plots.KCPlot("Tire normal force (0: off the "
                                        "road)", "Time [s]", "N")
        plots.addWidget(self.plot, 2)
        plots.addWidget(self.force_plot, 1)
        layout.addLayout(plots, 1)
        self.kind.setCurrentIndex(self.kind.findData('bump'))
        self._show_fields()

    def _show_fields(self):
        key = self.kind.currentData()
        defaults = _session.ROAD_INPUTS[key][1]
        for name, spin in self.fields.items():
            shown = name in defaults
            spin.setVisible(shown)
            self.labels[name].setVisible(shown)
            if shown:
                spin.blockSignals(True)
                spin.setValue(float(defaults[name]))
                spin.blockSignals(False)
        self.csv_button.setVisible('csv' == key)

    def road(self):
        """``(kind, parameters)`` of the road input the controls show."""
        key = self.kind.currentData()
        params = {name: self.fields[name].value()
                  for name in _session.ROAD_INPUTS[key][1]
                  if name in self.fields}
        if 'csv' == key:
            params['path'] = self.csv_path
        return key, params

    def clear(self):
        """Drop the run of another model: the plots, the status, and the
        replay."""
        self.plot.set_curves([])
        self.force_plot.set_curves([])
        self.status.setText("")
        self.replay.blockSignals(True)
        self.replay.setRange(0, 0)
        self.replay.setEnabled(False)
        self.replay.blockSignals(False)
        self.replay_label.setText("")
        self.run.setEnabled(True)
        self.cancel.setEnabled(False)

    def set_road(self, key, **params):
        """Select a road input and set its parameters (for scripts)."""
        self.kind.setCurrentIndex(self.kind.findData(key))
        for name, value in params.items():
            if 'path' == name:
                self.csv_path = value
            else:
                self.fields[name].setValue(float(value))


class ModalTab(QWidget):
    """The ride and wheel hop modes of the quarter car."""

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.mass = _spin(10.0, 2000.0, 200.0, " kg", decimals=0, step=10.0)
        row.addWidget(QLabel("Sprung mass"))
        row.addWidget(self.mass)
        self.run = _button("Run modal analysis", lambda: handlers['modal'](
            self.mass.value()))
        row.addWidget(self.run)
        self.animate = _button("Animate the selected mode",
                               lambda: handlers['mode_shape'](
                                   self.mass.value(), self.selected_mode()))
        row.addWidget(self.animate)
        row.addStretch(1)
        layout.addLayout(row)
        self.replay = ReplayBar('modal', handlers)
        layout.addWidget(self.replay)
        self.status = QLabel("")
        layout.addWidget(self.status)
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(["mode", "frequency [Hz]",
                                    "damping ratio [-]"])
        layout.addWidget(self.table, 1)

    def clear(self):
        """Drop the modes of another model."""
        self.table.clear()
        self.status.setText("")

    def selected_mode(self):
        """The index of the selected mode row (the first one by default)."""
        item = self.table.currentItem()
        return self.table.indexOfTopLevelItem(item) if item else 0

    def show_modes(self, modes):
        self.table.clear()
        for name, freq, zeta in modes:
            self.table.addTopLevelItem(QTreeWidgetItem(
                [name, f"{freq:.3f}", f"{zeta:.4f}"]))

    def rows(self):
        return [[self.table.topLevelItem(it).text(col) for col in range(3)]
                for it in range(self.table.topLevelItemCount())]


class ReplayBar(QWidget):
    """A slider over the frames of an analysis and a Play button: the 3D
    view shows the frame.

    :param key: the name the handlers get for this bar
    """

    def __init__(self, key, handlers, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.play = QPushButton("Play 3D")
        self.play.setCheckable(True)
        self.play.setEnabled(False)
        self.play.toggled.connect(lambda on: handlers['animate'](key, on))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(
            lambda v: handlers['frame'](key, v))
        self.label = QLabel("")
        self.label.setMinimumWidth(180)
        row.addWidget(QLabel("3D"))
        row.addWidget(self.play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.label)

    def set_count(self, count, index=0):
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, count - 1))
        self.slider.setValue(index)
        self.slider.setEnabled(count > 1)
        self.slider.blockSignals(False)
        self.play.setEnabled(count > 1)
        if count <= 1:
            self.play.setChecked(False)


class SetupTab(QWidget):
    """The setup sheet of the corner: the springs, the ride height, the
    damper, the bar, and the tires on the left, and the engineering
    numbers of the reference and the current setup on the right."""

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        self._handlers = handlers
        # The setup fields above the numbers: the dock is often narrow.
        layout = QVBoxLayout(self)
        left = QVBoxLayout()
        box = QGroupBox("Setup (the model file is the first reference)")
        self.grid = QGridLayout(box)
        left.addWidget(box)
        row = QHBoxLayout()
        self.mass = _spin(10.0, 2000.0, 200.0, " kg", decimals=0, step=10.0)
        self.mass.setToolTip("The sprung mass on this corner")
        row.addWidget(QLabel("Corner sprung mass"))
        row.addWidget(self.mass)
        row.addStretch(1)
        left.addLayout(row)
        row = QHBoxLayout()
        for key, text in (('readouts', "Update numbers"),
                          ('reference', "Use as reference"),
                          ('reset', "Back to the model file"),
                          ('save', "Save setup..."),
                          ('load', "Load setup...")):
            button = _button(text, lambda k=key: handlers['setup_action'](
                k, self.mass.value()))
            row.addWidget(button)
            setattr(self, f"{key}_button", button)
        left.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        layout.addLayout(left)
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(["quantity", "reference", "current",
                                    "change"])
        self.table.header().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(260)
        layout.addWidget(self.table, 1)
        self.spins = {}

    def show_fields(self, rows):
        """Lay out the setup fields: ``rows`` of
        :meth:`SuspensionSession.setup_fields`."""
        _clear_grid(self.grid)
        self.spins = {}
        for col, text in enumerate(("", "current", "reference")):
            self.grid.addWidget(QLabel(text), 0, col)
        for it, (key, label, unit, value, ref, (lo, hi), step,
                 decimals) in enumerate(rows, start=1):
            if 'clicks' == unit:
                spin = QSpinBox()
                spin.setRange(int(lo), int(hi))
                spin.setValue(int(round(value)))
                spin.setSuffix(" clicks")
                spin.setKeyboardTracking(False)
                ref_text = f"{int(round(ref))} clicks"
            else:
                spin = _spin(lo, hi, value, f" {unit}", decimals=decimals,
                             step=step)
                ref_text = f"{ref:.{decimals}f} {unit}"
            spin.valueChanged.connect(
                lambda v, k=key: self._handlers['setup'](k, v,
                                                         self.mass.value()))
            self.grid.addWidget(QLabel(label), it, 0)
            self.grid.addWidget(spin, it, 1)
            self.grid.addWidget(QLabel(ref_text), it, 2)
            self.spins[key] = spin

    def set_value(self, key, value):
        """Set a field without running the handler (for scripts)."""
        spin = self.spins[key]
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def clear(self):
        self.table.clear()
        self.status.setText("")

    def show_readouts(self, out):
        self.table.clear()
        for label, unit, fmt, ref, cur in out['rows']:
            def text(v):
                return "" if v is None else f"{v:{fmt}} {unit}"
            change = ""
            if ref is not None and cur is not None:
                delta = cur - ref
                change = (f"{delta:+{fmt.lstrip('+')}}"
                          if abs(delta) > 1e-9 else "")
            self.table.addTopLevelItem(QTreeWidgetItem(
                [label, text(ref), text(cur), change]))

    def rows(self):
        return [[self.table.topLevelItem(it).text(col) for col in range(4)]
                for it in range(self.table.topLevelItemCount())]


class RideTab(QWidget):
    """Ride comfort and road holding of the quarter car on an ISO 8608
    random road: the linear random response with the ISO 2631-1 comfort
    reaction, the frequency response, and a time run of the engine on a
    road sample of the class."""

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.road_class = QComboBox()
        for name in _session.ROAD_CLASSES:
            self.road_class.addItem(f"ISO 8608 class {name}", name)
        self.road_class.setCurrentIndex(2)
        self.speed = _spin(5.0, 300.0, 60.0, " km/h", decimals=0, step=10.0)
        self.mass = _spin(10.0, 2000.0, 200.0, " kg", decimals=0, step=10.0)
        for label, widget in (("Road", self.road_class),
                              ("Speed", self.speed),
                              ("Sprung mass", self.mass)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.run = _button("Ride response", lambda: handlers['ride'](
            self.road_class.currentData(), self.speed.value(),
            self.mass.value()))
        self.run_time = _button("Time run (12 s)",
                                lambda: handlers['ride_time'](
                                    self.road_class.currentData(),
                                    self.speed.value(), self.mass.value()))
        row.addWidget(self.run)
        row.addWidget(self.run_time)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.replay = ReplayBar('ride', handlers)
        layout.addWidget(self.replay)
        plots = QHBoxLayout()
        self.plot = _plots.KCPlot("Body acceleration per road height",
                                  "Frequency [Hz]", "m/s^2 per mm")
        self.time_plot = _plots.KCPlot("Body acceleration (time run)",
                                       "Time [s]", "m/s^2")
        plots.addWidget(self.plot, 1)
        plots.addWidget(self.time_plot, 1)
        layout.addLayout(plots, 1)

    def clear(self):
        """Drop the results of another model."""
        self.plot.set_curves([])
        self.time_plot.set_curves([])
        self.status.setText("")

    def show_response(self, out):
        st = out['stats']
        self.plot.set_curves([
            ("body acceleration", list(out['freq']), list(out['body_acc']),
             _plots.PREVIOUS_COLOR, False),
            ("Wk weighted", list(out['freq']), list(out['body_acc_w']),
             _plots.CURRENT_COLOR, False)])
        holding = ("the tire stays on the road"
                   if st['tire_load_ratio'] < 0.3 else
                   "the tire load reaches zero: loss of road holding")
        self.status.setText(
            f"ISO 8608 class {st['road_class']} at {st['speed'] * 3.6:g} "
            f"km/h (linear random response): body acceleration "
            f"{st['body_acc']:.3f} m/s^2 rms; ISO 2631-1 Wk weighted a_w "
            f"{st['body_acc_w']:.3f} m/s^2: {st['comfort']}.  Tire load "
            f"ratio sigma_F / F_static {st['tire_load_ratio']:.3f} "
            f"({holding}).  Suspension travel {st['travel'] * 1.e3:.1f} mm "
            f"rms.")

    def show_time(self, out):
        t = list(out['time'])
        self.time_plot.set_curves([
            ("body acceleration", t, list(out['body_acc']),
             _plots.CURRENT_COLOR, False)])
        self.status.setText(
            self.status.text() + f"  Time run (engine, one road sample, "
            f"after 2 s): a_w {out['body_acc_w']:.3f} m/s^2 "
            f"({out['comfort']}), tire load ratio "
            f"{out['tire_load_ratio']:.3f}, travel "
            f"{out['travel_std'] * 1.e3:.1f} mm rms, off the road "
            f"{out['airborne'] * 1.e3:.0f} ms.")


class VehicleTab(QWidget):
    """The full vehicle of a bundled vehicle file: steady-state cornering
    (roll gradient, lateral load transfer, LLTD, understeer gradient)
    against the roll center model, and a step steer at speed against the
    linear bicycle model."""

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        self._handlers = handlers
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.vehicle = QComboBox()
        row.addWidget(QLabel("Vehicle"))
        row.addWidget(self.vehicle, 1)
        self.vehicle.currentIndexChanged.connect(
            lambda _: handlers['vehicle_picked'](self.vehicle.currentData()))
        self.run = _button("Steady-state cornering",
                           lambda: handlers['vehicle'](
                               self.vehicle.currentData(), self.setups()))
        self.handling = _button("Handling diagram",
                                lambda: handlers['handling'](
                                    self.vehicle.currentData(),
                                    self.setups()))
        self.reference = _button("Use as reference",
                                 lambda: handlers['vehicle_reference']())
        for button in (self.run, self.handling, self.reference):
            row.addWidget(button)
        layout.addLayout(row)
        self.replay = ReplayBar('vehicle', handlers)
        layout.addWidget(self.replay)
        box = QGroupBox("Vehicle setup (front / rear axle)")
        self.setup_grid = QGridLayout(box)
        layout.addWidget(box)
        self.setup_spins = {}
        row = QHBoxLayout()
        self.speed = _spin(10.0, 250.0, 72.0, " km/h", decimals=0, step=5.0)
        self.rack = _spin(-50.0, 50.0, 2.0, " mm", decimals=2, step=0.5)
        for label, widget in (("Speed", self.speed), ("Rack step",
                                                      self.rack)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.step = _button("Step steer", lambda: handlers['step_steer'](
            self.vehicle.currentData(), self.speed.value(),
            self.rack.value(), self.setups()))
        row.addWidget(self.step)
        row.addStretch(1)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.bias = _spin(20.0, 95.0, 65.0, " % front", decimals=1,
                          step=1.0)
        self.drive = QComboBox()
        for label, split in _session.SuspensionSession.DRIVES:
            self.drive.addItem(label, split)
        for label, widget in (("Brake bias", self.bias),
                              ("Drive", self.drive)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.longitudinal = _button(
            "Braking / traction", lambda: handlers['longitudinal'](
                self.vehicle.currentData(), self.bias.value() / 100.0,
                self.drive.currentData(), self.setups()))
        row.addWidget(self.longitudinal)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        plots = QGridLayout()
        self.roll_plot = _plots.KCPlot("Roll and load transfer",
                                       "Lateral acceleration [g]",
                                       "deg, kN")
        self.yaw_plot = _plots.KCPlot("Step steer yaw rate", "Time [s]",
                                      "deg/s")
        self.handling_plot = _plots.KCPlot(
            "Handling diagram: understeer angle delta - L/R",
            "Lateral acceleration [g]", "deg")
        self.brake_plot = _plots.KCPlot(
            "Braking: grip used by each axle", "Deceleration [g]",
            "share of grip")
        # Two by two: the handling and the braking above, the roll and the
        # step steer below.
        plots.addWidget(self.handling_plot, 0, 0)
        plots.addWidget(self.brake_plot, 0, 1)
        plots.addWidget(self.roll_plot, 1, 0)
        plots.addWidget(self.yaw_plot, 1, 1)
        #: The handling result the plot compares with (dashed).
        self.reference_handling = None
        self.last_handling = None
        layout.addLayout(plots, 1)

    #: Labels and spin settings of the vehicle setup fields.
    SETUP_FIELDS = {
        'spring_rate': ("spring rate", " N/mm", (1.0, 500.0), 5.0, 1),
        'arb_rate': ("anti-roll bar", " N m/deg", (0.0, 500.0), 2.0, 1),
        'ride_height': ("ride height", " mm", (-80.0, 60.0), 5.0, 1),
        'damper_bump': ("damper bump, low speed", " N.s/mm", (0.0, 50.0),
                        0.1, 2),
        'damper_rebound': ("damper rebound, low speed", " N.s/mm",
                           (0.0, 50.0), 0.1, 2),
        'bump_clicks': ("damper bump (measured map)", " clicks", (0, 64),
                        1, 0),
        'rebound_clicks': ("damper rebound (measured map)", " clicks",
                           (0, 64), 1, 0),
    }

    def show_vehicle_setup(self, setups):
        """Lay out the setup fields of each axle (``{"front": {...},
        "rear": {...}}``)."""
        _clear_grid(self.setup_grid)
        self.setup_spins = {}
        keys = [k for k in self.SETUP_FIELDS
                if k in setups.get('front', {}) or k in setups.get('rear',
                                                                   {})]
        for col, text in enumerate(("", "front", "rear"), start=0):
            self.setup_grid.addWidget(QLabel(text), 0, col)
        for it, key in enumerate(keys, start=1):
            label, unit, (lo, hi), step, decimals = self.SETUP_FIELDS[key]
            self.setup_grid.addWidget(QLabel(label), it, 0)
            for col, axle_name in enumerate(("front", "rear"), start=1):
                if key not in setups.get(axle_name, {}):
                    continue
                value = setups[axle_name][key]
                if 'clicks' in unit:
                    spin = QSpinBox()
                    spin.setRange(int(lo), int(hi))
                    spin.setValue(int(round(value)))
                    spin.setSuffix(unit)
                else:
                    spin = _spin(lo, hi, value, unit, decimals=decimals,
                                 step=step)
                # A changed setup makes the shown results stale.
                spin.valueChanged.connect(
                    lambda _: self._handlers['vehicle_setup_changed']())
                self.setup_grid.addWidget(spin, it, col)
                self.setup_spins[(axle_name, key)] = spin

    def clear_results(self, note="", keep_reference=True):
        """Drop every result shown: the status, the plots, the handling
        curve (and the reference unless ``keep_reference``)."""
        if not keep_reference:
            self.reference_handling = None
        self.last_handling = None
        ref = self.reference_handling
        if ref is not None:
            deg = 180.0 / 3.141592653589793
            self.handling_plot.set_curves([
                ("reference", list(ref['a_y']),
                 list(ref['understeer_angle'] * deg),
                 _plots.PREVIOUS_COLOR, True)])
        else:
            self.handling_plot.set_curves([])
        for plot in (self.brake_plot, self.roll_plot, self.yaw_plot):
            plot.set_curves([])
        self.status.setText(note)

    def setups(self):
        """The vehicle setup the controls show."""
        out = {'front': {}, 'rear': {}}
        for (axle_name, key), spin in self.setup_spins.items():
            out[axle_name][key] = float(spin.value())
        return out

    def show_handling(self, out, readouts=None):
        deg = 180.0 / 3.141592653589793
        curves = []
        ref = self.reference_handling
        if ref is not None:
            curves.append(("reference", list(ref['a_y']),
                           list(ref['understeer_angle'] * deg),
                           _plots.PREVIOUS_COLOR, True))
        curves.append(("current", list(out['a_y']),
                       list(out['understeer_angle'] * deg),
                       _plots.CURRENT_COLOR, False))
        self.handling_plot.set_curves(curves)
        self.last_handling = out
        use = out['grip_use']
        at = int(min(range(len(out['a_y'])),
                     key=lambda k: abs(out['a_y'][k] - 0.8))) \
            if len(out['a_y']) else None
        balance = {"front": "the front axle saturates first: the car "
                            "pushes (limit understeer)",
                   "rear": "the rear axle saturates first: the car spins "
                           "(limit oversteer)",
                   "lift-off": "an inner wheel leaves the road first",
                   None: "no limit in the range"}[out['limited_by']]
        text = (f"Limit lateral acceleration {out['limit']:.3f} g: "
                f"{balance}.")
        if at is not None:
            text += (f"  At {out['a_y'][at]:.2f} g: understeer angle "
                     f"{out['understeer_angle'][at] * deg:+.3f} deg, grip "
                     f"used front {use[at][0] * 100:.0f} % / rear "
                     f"{use[at][1] * 100:.0f} %.")
        if ref is not None:
            text += f"  Reference limit {ref['limit']:.3f} g."
        if readouts is not None:
            f, r = readouts['front'], readouts['rear']
            share = f['roll_stiffness'] / (f['roll_stiffness']
                                           + r['roll_stiffness'])
            text += (f"  Ride frequency front {f['ride_frequency']:.2f} Hz,"
                     f" rear {r['ride_frequency']:.2f} Hz (rear / front "
                     f"{r['ride_frequency'] / f['ride_frequency']:.2f}); "
                     f"roll stiffness front {f['roll_stiffness']:.0f}, rear "
                     f"{r['roll_stiffness']:.0f} N m/deg (front share "
                     f"{share * 100:.0f} %);"
                     f" damping ratio bump / rebound front "
                     f"{f['damping_ratio_bump']:.2f} / "
                     f"{f['damping_ratio_rebound']:.2f}, rear "
                     f"{r['damping_ratio_bump']:.2f} / "
                     f"{r['damping_ratio_rebound']:.2f}.")
        self.status.setText(text)

    def show_longitudinal(self, out):
        brake, traction = out['brake'], out['traction']
        decel = list(brake['decel'])
        self.brake_plot.set_curves([
            ("front", decel, list(brake['use'][:, 0]),
             _plots.CURRENT_COLOR, False),
            ("rear", decel, list(brake['use'][:, 1]),
             _plots.PREVIOUS_COLOR, False),
            ("ideal front bias", decel, list(brake['ideal_bias']),
             (40, 150, 70), True)])
        k = min(range(len(decel)), key=lambda i: abs(decel[i]
                                                     - brake['limit']))
        b, d, f = out['braking'], out['driving'], out['formula']
        spin = {"front": "the front wheels spin first",
                "rear": "the rear wheels spin first",
                None: "no limit in the range"}[traction['spins']]
        self.status.setText(
            f"Braking with {out['brake_bias'] * 100:.0f} % front: limit "
            f"{brake['limit']:.3f} g, the {brake['locks'] or 'no'} axle "
            f"locks first; the ideal bias there is "
            f"{brake['ideal_bias'][k] * 100:.0f} % front.  Pitch "
            f"{b['pitch']:+.2f} deg/g (nose down +); dive "
            f"{b['travel'][0]:+.1f} mm/g front, {b['travel'][1]:+.1f} mm/g "
            f"rear (+ bump); anti-dive {b['anti'][0] * 100:.0f} % (RCVD "
            f"{f['anti_dive'] * 100:.0f} %), anti-lift "
            f"{b['anti'][1] * 100:.0f} % (RCVD {f['anti_lift'] * 100:.0f} "
            f"%).  Traction ({out['drive_split'] * 100:.0f} % front): "
            f"limit {traction['limit']:.3f} g, {spin}; pitch "
            f"{d['pitch']:+.2f} deg/g; squat {d['travel'][1]:+.1f} mm/g "
            f"rear, front {d['travel'][0]:+.1f} mm/g; anti-squat "
            f"{d['anti'][1] * 100:.0f} % (RCVD "
            f"{f['anti_squat'] * 100:.0f} %), front anti-lift "
            f"{d['anti'][0] * 100:.0f} % (RCVD "
            f"{f['anti_lift_drive'] * 100:.0f} %).")

    def show_vehicles(self, rows):
        self.vehicle.blockSignals(True)
        self.vehicle.clear()
        for stem, name in rows:
            self.vehicle.addItem(name, stem)
        self.vehicle.blockSignals(False)

    def show_steady(self, out):
        ss, cf = out['steady'], out['closed']
        ay = list(ss['a_y'])
        deg = 180.0 / 3.141592653589793
        self.roll_plot.set_curves([
            ("roll [deg]", ay, list(ss['roll'] * deg),
             _plots.CURRENT_COLOR, False),
            ("front load transfer [kN]", ay,
             list(ss['load_transfer'][:, 0] * 1.e-3),
             _plots.PREVIOUS_COLOR, False),
            ("rear load transfer [kN]", ay,
             list(ss['load_transfer'][:, 1] * 1.e-3), (40, 150, 70),
             False)])
        kus = out['understeer'] * deg
        balance = ("understeer" if kus > 0.0 else "oversteer")
        self.status.setText(
            f"{out['name']}: roll gradient {ss['roll_gradient']:.2f} deg/g "
            f"(roll center model {cf['roll_gradient']:.2f}); LLTD "
            f"{ss['lltd'] * 100:.1f} % front (roll center model "
            f"{cf['lltd'] * 100:.1f} %); roll centers "
            f"{cf['roll_center'][0] * 1.e3:.0f} / "
            f"{cf['roll_center'][1] * 1.e3:.0f} mm; roll steer "
            f"{ss['roll_steer'][0] * deg:+.3f} / "
            f"{ss['roll_steer'][1] * deg:+.3f} deg/g (front / rear, + into "
            f"the turn); understeer gradient {kus:+.2f} deg/g "
            f"({balance}).")

    def show_step(self, out):
        deg = 180.0 / 3.141592653589793
        t = list(out['time'])
        self.yaw_plot.set_curves([
            ("yaw rate", t, list(out['yaw_rate'] * deg),
             _plots.CURRENT_COLOR, False),
            ("bicycle model, steady", [t[0], t[-1]],
             [out['bicycle_yaw_rate'] * deg] * 2, _plots.PREVIOUS_COLOR,
             True)])
        st = out['steady']
        self.status.setText(
            f"Step steer at {st['speed'] * 3.6:.0f} km/h, front wheels "
            f"{out['delta'] * deg:+.3f} deg: steady yaw rate "
            f"{st['yaw_rate'] * deg:+.3f} deg/s (bicycle model "
            f"{out['bicycle_yaw_rate'] * deg:+.3f}), lateral acceleration "
            f"{st['lateral_acceleration'] / 9.81:+.3f} g.")


class HardpointTab(QWidget):
    """The hardpoint table: edit a coordinate, jog the selected point, and
    set its limit box.  A row selection selects the point in 3D."""

    COLUMNS = ("hardpoint", "x [mm]", "y [mm]", "z [mm]", "limit box [mm]")

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        self._handlers = handlers
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.linked = QCheckBox("Move coincident points together")
        self.linked.setChecked(True)
        self.linked.setToolTip("The two outer points of a wishbone share "
                               "its ball joint")
        self.linked.toggled.connect(lambda on: handlers['hardpoint'](
            'linked', None, on))
        self.axis = QComboBox()
        for it, axis in enumerate("xyz"):
            self.axis.addItem(axis, it)
        self.step = _spin(0.01, 50.0, 1.0, " mm", decimals=2, step=0.5)
        minus = _button("-", lambda: self._jog(-1.0))
        plus = _button("+", lambda: self._jog(1.0))
        minus.setMaximumWidth(30)
        plus.setMaximumWidth(30)
        self.box = _spin(0.1, 200.0, 10.0, " mm", decimals=1)
        set_box = _button("Set box", lambda: handlers['hardpoint'](
            'limit', self.current_name(), self.box.value()))
        clear_box = _button("Clear box", lambda: handlers['hardpoint'](
            'limit', self.current_name(), None))
        for label, widget in ((None, self.linked), ("Jog", self.axis),
                              (None, self.step), (None, minus),
                              (None, plus), ("Box +/-", self.box),
                              (None, set_box), (None, clear_box)):
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item)
        self.table.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.table, 1)
        self._filling = False

    def show_rows(self, rows):
        picked = self.current_name()
        self._filling = True
        self.table.setRowCount(len(rows))
        for it, (name, xyz, limits) in enumerate(rows):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(it, 0, item)
            for col, value in enumerate(xyz):
                self.table.setItem(it, col + 1,
                                   QTableWidgetItem(f"{value:.2f}"))
            text = "-"
            if limits is not None:
                lo, hi = limits
                text = " ".join(f"[{a:.1f}, {b:.1f}]" for a, b in zip(lo, hi))
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(it, 4, item)
        self._filling = False
        if picked is not None:
            self.select(picked)

    def names(self):
        return [self.table.item(it, 0).text()
                for it in range(self.table.rowCount())]

    def current_name(self):
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 0) is None:
            return None
        return self.table.item(row, 0).text()

    def select(self, name):
        """Select the row of ``name`` without telling the app."""
        self._filling = True
        names = self.names()
        if name in names:
            self.table.selectRow(names.index(name))
        else:
            self.table.clearSelection()
        self._filling = False

    def coordinates(self, row):
        return [float(self.table.item(row, col).text()) for col in (1, 2, 3)]

    def _jog(self, sign):
        name = self.current_name()
        if name is None:
            return
        row = self.names().index(name)
        xyz = self.coordinates(row)
        xyz[self.axis.currentData()] += sign * self.step.value()
        self._handlers['hardpoint']('set', name, xyz)

    def _on_item(self, item):
        if self._filling or item.column() not in (1, 2, 3):
            return
        row = item.row()
        name = self.table.item(row, 0).text()
        try:
            xyz = self.coordinates(row)
        except ValueError:
            self._handlers['hardpoint']('invalid', name, item.text())
            return
        self._handlers['hardpoint']('set', name, xyz)

    def _on_select(self):
        if not self._filling:
            self._handlers['hardpoint']('select', self.current_name(), None)


class BatchTab(QWidget):
    """Vary one hardpoint coordinate and plot the K&C summaries at design
    against it."""

    COLUMNS = 4

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.point = QComboBox()
        self.axis = QComboBox()
        for it, axis in enumerate("xyz"):
            self.axis.addItem(axis, it)
        self.axis.setCurrentIndex(2)
        self.lo = _spin(-100.0, 100.0, -5.0, " mm")
        self.hi = _spin(-100.0, 100.0, 5.0, " mm")
        self.count = QSpinBox()
        self.count.setRange(2, 51)
        self.count.setValue(11)
        self.count.setSuffix(" offsets")
        self.run = _button("Run batch", lambda: handlers['batch'](
            self.point.currentText(), self.axis.currentData(),
            self.lo.value(), self.hi.value(), self.count.value()))
        for label, widget in (("Hardpoint", self.point), ("axis", self.axis),
                              ("offset from", self.lo), ("to", self.hi),
                              (None, self.count), (None, self.run)):
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel("")
        layout.addWidget(self.status)
        grid = QGridLayout()
        self.plots = {}
        for it, (key, title, unit, _) in enumerate(
                _session.SuspensionSession.BATCH_SUMMARIES):
            plot = _plots.KCPlot(title, "Offset [mm]", unit)
            grid.addWidget(plot, it // self.COLUMNS, it % self.COLUMNS)
            self.plots[key] = plot
        layout.addLayout(grid, 1)

    def clear(self, note=""):
        """Drop the results of another model."""
        for plot in self.plots.values():
            plot.set_curves([])
        self.status.setText(note)

    def show_points(self, names):
        current = self.point.currentText()
        self.point.blockSignals(True)
        self.point.clear()
        self.point.addItems(names)
        if current in names:
            self.point.setCurrentText(current)
        elif 'tie_rod_inner' in names:
            self.point.setCurrentText('tie_rod_inner')
        self.point.blockSignals(False)

    def show_result(self, out, label):
        for key, plot in self.plots.items():
            plot.set_curves([(label, list(out['offset']), list(out[key]),
                              _plots.CURRENT_COLOR, False)])


class ProfileTab(QWidget):
    """A motion profile of the axle: rows of heave, roll, and rack travel,
    typed or read from a CSV file; the slider poses a row in 3D."""

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["heave [mm]", "roll [deg]",
                                              "rack [mm]"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        left.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(_button("Add row", self.add_row))
        row.addWidget(_button("Delete row", self.delete_row))
        row.addWidget(_button("Import CSV...", handlers['profile_import']))
        self.run = _button("Run profile", lambda: handlers['profile'](
            self.rows()))
        row.addWidget(self.run)
        left.addLayout(row)
        right = QVBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        cursor = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(
            lambda v: handlers['profile_cursor'](v))
        self.cursor_label = QLabel("")
        cursor.addWidget(QLabel("Row"))
        cursor.addWidget(self.slider, 1)
        cursor.addWidget(self.cursor_label)
        right.addLayout(cursor)
        plots = QHBoxLayout()
        self.plots = {}
        for key, title in (('camber', "Camber"), ('toe', "Toe-in")):
            plot = _plots.KCPlot(title, "Profile row", "deg")
            plots.addWidget(plot)
            self.plots[key] = plot
        right.addLayout(plots, 1)
        layout.addLayout(left, 2)
        layout.addLayout(right, 3)
        for heave in (-30.0, -15.0, 0.0, 15.0, 30.0):
            self.add_row((heave, 0.0, 0.0))

    def clear(self, note=""):
        """Drop the results of another model (the rows stay)."""
        for plot in self.plots.values():
            plot.set_curves([])
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.cursor_label.setText("")
        self.status.setText(note)

    def add_row(self, values=(0.0, 0.0, 0.0)):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, value in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(f"{value:g}"))

    def delete_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def set_rows(self, rows):
        self.table.setRowCount(0)
        for values in rows:
            self.add_row(values)

    def rows(self):
        """The rows as floats; a cell that is not a number raises."""
        out = []
        for row in range(self.table.rowCount()):
            values = []
            for col in range(3):
                item = self.table.item(row, col)
                text = item.text() if item is not None else "0"
                try:
                    values.append(float(text))
                except ValueError:
                    raise ValueError(f"row {row + 1}, column {col + 1}: "
                                     f"{text!r} is not a number") from None
            out.append(tuple(values))
        return out

    def show_result(self, out):
        xs = list(range(1, len(out['rows']) + 1))
        for key, plot in self.plots.items():
            plot.set_curves([
                ("left", xs, list(out[key + '_left']), _plots.CURRENT_COLOR,
                 False),
                ("right", xs, list(out[key + '_right']),
                 _plots.PREVIOUS_COLOR, True)])
        count = len(out['poses'])
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, count - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(count > 0)
        self.slider.blockSignals(False)


class AnalysisPanel(QWidget):
    """The analysis tabs and the warning list side by side.

    :param handlers: The app's callbacks by name: ``kind``, ``range``,
        ``cursor``, ``export``, ``forces``, ``statics``, ``dynamics``,
        ``cancel``, ``modal``, ``ride``, ``ride_time``, ``vehicle``, and
        ``step_steer``.
    """

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        splitter = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget()
        self.sweep = SweepTab(handlers)
        self.statics = StaticsTab(handlers)
        self.dynamics = DynamicsTab(handlers)
        self.modal = ModalTab(handlers)
        self.hardpoints = HardpointTab(handlers)
        self.batch = BatchTab(handlers)
        self.profile = ProfileTab(handlers)
        self.ride = RideTab(handlers)
        self.vehicle = VehicleTab(handlers)
        self.setup = SetupTab(handlers)
        for title, tab in (("Setup", self.setup),
                           ("Sweep", self.sweep), ("Statics", self.statics),
                           ("Dynamics", self.dynamics),
                           ("Modal", self.modal), ("Ride", self.ride),
                           ("Vehicle", self.vehicle),
                           ("Hardpoints", self.hardpoints),
                           ("Batch", self.batch),
                           ("Profile", self.profile)):
            self.tabs.addTab(scrollable(tab), title)
        # The sweep plots first; the setup sheet is one tab to the left.
        self.tabs.setCurrentIndex(1)
        box = QGroupBox("Warnings")
        self.warnings = _panel.WarningList()
        QVBoxLayout(box).addWidget(self.warnings)
        splitter.addWidget(self.tabs)
        splitter.addWidget(box)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
