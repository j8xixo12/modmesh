# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Overview tab: the workflow in steps and the KPI cards of the corner and
the vehicle, grouped by the engineering question each answers.

A card shows the value with its unit, the target band of the chosen
profile (green inside, orange outside, grey without a band), and the
change from the baseline the user set.  Its tooltip gives the definition
and the source of the band (see :mod:`._glossary`).  The tab holds no
analysis: the app fills it from the session, and clears the cards of an
analysis whenever an input of that analysis changes.
"""

from PySide6.QtCore import QEvent, Signal

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from . import _glossary

__all__ = [
    'CORNER_GROUPS',
    'OverviewTab',
    'VEHICLE_GROUPS',
]

#: The groups the corner KPIs fill and the ones the vehicle runs fill.
CORNER_GROUPS = ('ride', 'kc')
VEHICLE_GROUPS = ('handling', 'braking')

#: What to run for the cards of a group.
HOW = {
    'ride': "press Update corner KPIs",
    'kc': "press Update corner KPIs",
    'handling': "run Steady-state cornering, Handling diagram, or Step "
                "steer in the Vehicle tab",
    'braking': "run Braking / traction in the Vehicle tab",
}

COLORS = {
    False: {'in': ('#edf7f1', '#26734d'), 'out': ('#fff3df', '#955b16'),
            'none': ('#f2f4f7', '#526174')},
    True: {'in': ('#1c352e', '#77d6ac'), 'out': ('#3b3020', '#f0bd70'),
           'none': ('#252d39', '#a8b9cf')},
}


class OverviewTab(QWidget):
    """The KPI cards.

    :ivar values: ``{kpi: value}`` shown now
    :ivar baseline: ``{kpi: value}`` the changes are from
    :ivar custom_targets: ``{kpi: (lo, hi)}`` the user loaded
    :ivar cards: ``{kpi: QLabel}``
    """

    navigate = Signal(str, int)

    def __init__(self, handlers, parent=None):
        super().__init__(parent)
        self._handlers = handlers
        self.values = {}
        self.baseline = {}
        self.custom_targets = {}
        self.notes = {}
        layout = QVBoxLayout(self)
        heading = QLabel("Suspension workspace")
        heading.setStyleSheet("font-size: 18pt; font-weight: 600;")
        layout.addWidget(heading)
        guide = QLabel("Set up your vehicle, explore suspension travel, "
                       "then compare performance against your targets.")
        guide.setWordWrap(True)
        layout.addWidget(guide)
        steps = QHBoxLayout()
        self.shortcuts = {}
        for label, page, task in (("1  Vehicle setup", 'vehicle', 0),
                                  ("2  Kinematics", 'sweep', 0),
                                  ("3  Handling", 'vehicle', 1)):
            button = QPushButton(label)
            button.setMinimumHeight(36)
            button.clicked.connect(
                lambda checked=False, p=page, t=task: self.navigate.emit(p, t))
            steps.addWidget(button)
            self.shortcuts[page, task] = button
        layout.addLayout(steps)
        row = QHBoxLayout()
        self.profile = QComboBox()
        for key, label in _glossary.PROFILES:
            self.profile.addItem(label, key)
        self.profile.currentIndexChanged.connect(lambda _: self.refresh())
        self.mass = QDoubleSpinBox()
        self.mass.setRange(20.0, 2000.0)
        self.mass.setDecimals(0)
        self.mass.setSingleStep(10.0)
        self.mass.setValue(400.0)
        self.mass.setSuffix(" kg")
        self.mass.setKeyboardTracking(False)
        self.update_button = QPushButton("Update corner KPIs")
        self.update_button.clicked.connect(
            lambda: handlers['overview_update'](self.mass.value()))
        self.baseline_button = QPushButton("Set as baseline")
        self.baseline_button.clicked.connect(self.set_baseline)
        self.targets_button = QPushButton("Load targets...")
        self.targets_button.clicked.connect(
            lambda: handlers['overview_targets']())
        for widget in (QLabel("Targets"), self.profile,
                       QLabel("Corner sprung mass"), self.mass,
                       self.update_button, self.baseline_button,
                       self.targets_button):
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row)
        self.cards = {}
        self.group_notes = {}
        self._grids = {}
        for group, title, question in _glossary.GROUPS:
            box = QGroupBox(title)
            inner = QVBoxLayout(box)
            ask = QLabel(question)
            ask.setWordWrap(True)
            inner.addWidget(ask)
            note = QLabel("")
            note.setWordWrap(True)
            inner.addWidget(note)
            self.group_notes[group] = note
            grid = QGridLayout()
            grid.setSpacing(10)
            keys = [k for k, v in _glossary.KPIS.items()
                    if v['group'] == group]
            for it, key in enumerate(keys):
                card = QLabel()
                card.setWordWrap(True)
                card.setMinimumWidth(150)
                card.setMargin(6)
                grid.addWidget(card, it // 4, it % 4)
                self.cards[key] = card
            self._grids[group] = (grid, keys)
            inner.addLayout(grid)
            layout.addWidget(box)
        layout.addStretch(1)
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = max(1, min(4, self.width() // 260))
        for grid, keys in self._grids.values():
            for index, key in enumerate(keys):
                grid.removeWidget(self.cards[key])
                grid.addWidget(self.cards[key], index // columns,
                               index % columns)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange and hasattr(self, 'cards'):
            self.refresh()

    def profile_key(self):
        return self.profile.currentData()

    def set_values(self, values):
        """Show ``values`` (``{kpi: value}``); other cards stay."""
        self.values.update({k: float(v) for k, v in values.items()
                            if k in _glossary.KPIS})
        self.refresh()

    def set_note(self, group, text):
        self.notes[group] = text
        self.refresh()

    def clear(self, groups, note=""):
        """Drop the values of ``groups`` (their inputs changed); the
        baseline stays."""
        for key, kpi in _glossary.KPIS.items():
            if kpi['group'] in groups:
                self.values.pop(key, None)
        for group in groups:
            self.notes[group] = note
        self.refresh()

    def set_baseline(self):
        """The values shown now become the baseline of the changes."""
        self.baseline = dict(self.values)
        self.refresh()

    def set_custom_targets(self, targets):
        self.custom_targets = dict(targets)
        self.refresh()

    def target(self, key):
        return _glossary.band(key, self.profile_key(), self.custom_targets)

    def state(self, key):
        """``'in'``, ``'out'``, ``'none'`` (no band), or ``'missing'`` (no
        value)."""
        if key not in self.values:
            return 'missing'
        return _glossary.status(self.values[key], self.target(key))

    def refresh(self):
        for group, _, _ in _glossary.GROUPS:
            text = self.notes.get(group, "")
            self.group_notes[group].setText(text)
            self.group_notes[group].setVisible(bool(text))
        for key, card in self.cards.items():
            kpi = _glossary.KPIS[key]
            unit = f" {kpi['unit']}" if kpi['unit'] else ""
            target = self.target(key)
            state = self.state(key)
            if 'missing' == state:
                value = "&mdash;"
                extra = "Not calculated"
            else:
                value = format(self.values[key], kpi['fmt']) + unit
                extra = ""
                if key in self.baseline:
                    delta = self.values[key] - self.baseline[key]
                    extra = (f"change from baseline "
                             f"{format(delta, '+' + kpi['fmt'].lstrip('+'))}"
                             f"{unit}")
            band = ("no target band" if target is None else
                    f"target {format(target[0], kpi['fmt'].lstrip('+'))} "
                    f"to {format(target[1], kpi['fmt'].lstrip('+'))}{unit}")
            state_label = {'in': 'Within target', 'out': 'Outside target'}
            extra = ' | '.join(filter(None, (state_label.get(state), extra)))
            card.setText(f"<b>{kpi['label']}</b><br>"
                         f"<span style='font-size:13pt'>{value}</span><br>"
                         f"{band}<br>"
                         f"{extra}")
            dark = self.palette().window().color().lightness() < 128
            colors = COLORS[dark]
            background, accent = colors.get(state, colors['none'])
            ink = '#e5edf7' if dark else '#243247'
            card.setStyleSheet(
                f"background: {background}; color: {ink}; "
                f"border-left: 3px solid {accent}; border-radius: 6px; "
                "padding: 8px;")
            tip = kpi['definition'] + '\nTo calculate: ' + HOW[kpi['group']]
            if target is not None:
                source = ("your targets file" if key in self.custom_targets
                          else kpi['source'])
                tip += f"\nTarget band: {source}."
            card.setToolTip(tip)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
