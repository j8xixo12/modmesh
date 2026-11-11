# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The docked panels of the suspension app: the model tree, the properties
with the adjusters, and the warnings.

The panels hold no state of the analysis.  They show what the session holds
and report what the user asks for through callbacks the app sets.
"""

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLayout,
                               QListWidget,
                               QListWidgetItem, QLineEdit, QPushButton,
                               QTabWidget,
                               QScrollArea, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import _session

__all__ = [
    'AdjusterPanel',
    'ModelTree',
    'WarningList',
]

#: The tree item role that holds the item name.
NAME_ROLE = Qt.UserRole


class ModelTree(QWidget):
    """The bodies, links, drivers, forces, and adjusters of the model.

    :ivar selected: Called with ``(group, name, properties)`` when an item
        is picked, when set.
    :ivar open_clicked: Called when "Open model" is clicked, when set.
    :ivar library_picked: Called with the file stem of a library model
        picked in the combo box, when set.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected = None
        self.open_clicked = None
        self.library_picked = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.file_label = QLabel("")
        self.file_label.setWordWrap(True)
        self.open_button = QPushButton("Open model...")
        self.open_button.clicked.connect(self._on_open)
        self.save_clicked = None
        self.save_button = QPushButton("Save model...")
        self.save_button.setToolTip("Write the model with its changes: "
                                    "over its file or to a new one")
        self.save_button.clicked.connect(
            lambda: self.save_clicked() if self.save_clicked else None)
        self.dirty_label = QLabel("")
        self.library = QComboBox()
        self.library.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.library.setMinimumContentsLength(18)
        self.library.setToolTip("Suspension types in the model library")
        self.library.activated.connect(self._on_library)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a component...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        #: Called with ``(group, name)`` on a double-click, when set.
        self.activated = None
        self.tree.itemDoubleClicked.connect(self._on_activated)
        layout.addWidget(self.file_label)
        layout.addWidget(QLabel("Library:"))
        layout.addWidget(self.library)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        layout.addWidget(self.dirty_label)
        layout.addWidget(self.search)
        self.search_empty = QLabel("No matching components")
        self.search_empty.setWordWrap(True)
        self.search_empty.hide()
        layout.addWidget(self.search_empty)
        layout.addWidget(self.tree, 1)
        self._items = {}
        self._properties = {}
        self._expanded_before_search = None

    def _filter_tree(self, text):
        query = text.strip().casefold()
        if query and self._expanded_before_search is None:
            self._expanded_before_search = {
                self.tree.topLevelItem(i).text(0):
                self.tree.topLevelItem(i).isExpanded()
                for i in range(self.tree.topLevelItemCount())}
        matches = False
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            group_match = query in group.text(0).casefold()
            visible = group_match
            for child_index in range(group.childCount()):
                child = group.child(child_index)
                match = group_match or query in child.text(0).casefold()
                child.setHidden(not match)
                visible = visible or match
            group.setHidden(not visible)
            matches = matches or visible
            if query and visible:
                group.setExpanded(True)
            elif not query and self._expanded_before_search is not None:
                group.setExpanded(self._expanded_before_search.get(
                    group.text(0), True))
        if not query:
            self._expanded_before_search = None
        self.search_empty.setVisible(bool(query) and not matches)

    def show_outline(self, outline, model_path):
        """Fill the tree from :meth:`SuspensionSession.outline`."""
        self.file_label.setText(f"Model: {os.path.basename(model_path)}")
        self.file_label.setToolTip(str(model_path))
        expanded = {self.tree.topLevelItem(i).text(0):
                    self.tree.topLevelItem(i).isExpanded()
                    for i in range(self.tree.topLevelItemCount())}
        scroll = self.tree.verticalScrollBar().value()
        self.tree.clear()
        self._items = {}
        self._properties = {}
        for group, entries in outline:
            top = QTreeWidgetItem([group])
            self.tree.addTopLevelItem(top)
            for name, props in entries:
                item = QTreeWidgetItem([name])
                item.setData(0, NAME_ROLE, name)
                top.addChild(item)
                self._items[name] = item
                self._properties[name] = (group, props)
            top.setExpanded(expanded.get(group, True))
        self._filter_tree(self.search.text())
        self.tree.verticalScrollBar().setValue(scroll)

    def names(self):
        return list(self._items)

    def select(self, name):
        """Select the item ``name``; return whether it exists."""
        item = self._items.get(name)
        if item is None:
            return False
        if item.isHidden() or item.parent().isHidden():
            self.search.clear()
        self.tree.setCurrentItem(item)
        return True

    def current_name(self):
        item = self.tree.currentItem()
        return None if item is None else item.data(0, NAME_ROLE)

    def _on_selection(self):
        name = self.current_name()
        if name is None or self.selected is None:
            return
        group, props = self._properties[name]
        self.selected(group, name, props)

    def _on_activated(self, item, column):
        name = item.data(0, NAME_ROLE)
        if name is None or self.activated is None:
            return
        group, _ = self._properties[name]
        self.activated(group, name)

    def activate(self, name):
        """Double-click the item ``name`` (for scripts and tests)."""
        item = self._items.get(name)
        if item is not None:
            self._on_activated(item, 0)

    def _on_open(self):
        if self.open_clicked is not None:
            self.open_clicked()

    def show_library(self, rows, current=None):
        """Fill the library combo with ``(stem, name, type)`` rows and pick
        ``current`` (a stem), without reporting it."""
        self.library.blockSignals(True)
        self.library.clear()
        for stem, name, _ in rows:
            self.library.addItem(name, stem)
        index = self.library.findData(current)
        self.library.setCurrentIndex(index)
        self.library.blockSignals(False)

    def _on_library(self, index):
        stem = self.library.itemData(index)
        if stem is not None and self.library_picked is not None:
            self.library_picked(stem)


class AdjusterPanel(QWidget):
    """The properties of the picked item, the adjusters, and the alignment
    at the cursor.

    :ivar changed: Called with ``(name, value_mm)`` when an adjuster is
        committed, when set.
    """

    #: Range and step of the adjusters, in mm (the range the target
    #: alignment solve allows).
    LIMIT = 25.0
    STEP = 0.1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.changed = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        props = QGroupBox("Properties")
        props_layout = QVBoxLayout(props)
        self.title = QLabel("(pick an item in the model tree)")
        self.title.setWordWrap(True)
        self.edited = None
        self.table = QTreeWidget()
        self.table.setColumnCount(3)
        self.table.setHeaderLabels(["property", "value", "unit"])
        self.table.setToolTip("Double-click a value of the model file to "
                              "change it")
        self.table.itemChanged.connect(self._on_item_changed)
        self.edit_status = QLabel("")
        self.edit_status.setWordWrap(True)
        self.table.setRootIsDecorated(False)
        self.table.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        props_layout.addWidget(self.title)
        props_layout.addWidget(self.table)
        props_layout.addWidget(self.edit_status)

        adjust = QGroupBox("Adjusters")
        form = QFormLayout(adjust)
        self.spins = {}
        for name, title in _session.ADJUSTERS:
            spin = QDoubleSpinBox()
            spin.setRange(-self.LIMIT, self.LIMIT)
            spin.setSingleStep(self.STEP)
            spin.setDecimals(2)
            spin.setSuffix(" mm")
            # Commit on Enter, an arrow step, or leaving the field, not on
            # every key, since each commit re-runs the sweep.
            spin.setKeyboardTracking(False)
            spin.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
            spin.valueChanged.connect(
                lambda value, key=name: self._on_changed(key, value))
            form.addRow(f"{title}:", spin)
            self.spins[name] = spin

        target = QGroupBox("Target alignment (static, at design)")
        grid = QGridLayout(target)
        self.target_on = {}
        self.target_value = {}
        for it, (key, title, value) in enumerate(
                (('camber', "Camber", -1.5), ('toe', "Toe-in", 0.0),
                 ('caster', "Caster", 1.7))):
            check = QCheckBox(title)
            check.setChecked(key != 'caster')
            spin = QDoubleSpinBox()
            spin.setRange(-30.0, 30.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(value)
            spin.setSuffix(" deg")
            grid.addWidget(check, it, 0)
            grid.addWidget(spin, it, 1)
            self.target_on[key] = check
            self.target_value[key] = spin
        self.solve = QPushButton("Solve adjusters")
        self.solve.clicked.connect(self._on_solve)
        grid.addWidget(self.solve, 3, 0, 1, 2)
        self.target_status = QLabel("")
        self.target_status.setWordWrap(True)
        grid.addWidget(self.target_status, 4, 0, 1, 2)
        self.solved = None

        align = QGroupBox("Alignment at the cursor")
        self.alignment = QLabel("")
        self.alignment.setTextInteractionFlags(Qt.TextSelectableByMouse)
        QVBoxLayout(align).addWidget(self.alignment)

        tabs = QTabWidget()
        tuning = QWidget()
        tuning_layout = QVBoxLayout(tuning)
        tuning_layout.setSizeConstraint(QLayout.SetMinimumSize)
        tuning_layout.setContentsMargins(0, 6, 0, 0)
        tuning_layout.addWidget(adjust)
        tuning_layout.addWidget(target)
        tuning_layout.addWidget(align)
        tuning_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(tuning)
        tabs.addTab(scroll, "Adjust")
        tabs.addTab(props, "Inspect")
        self.tabs = tabs
        layout.addWidget(tabs)

    def targets(self):
        """The checked targets, ``{key: deg}``."""
        return {key: spin.value() for key, spin in self.target_value.items()
                if self.target_on[key].isChecked()}

    def _on_solve(self):
        if self.solved is not None:
            self.solved(self.targets())

    def show_properties(self, title, props, editable=()):
        """Show the model file values of an item (``editable``: ``(label,
        path, unit, text)``, the value cell editable) and then its derived
        properties (``props``: ``(label, text)``, read only)."""
        self.title.setText(title)
        self.table.blockSignals(True)
        self.table.clear()
        for label, path, unit, text in editable:
            item = QTreeWidgetItem([label, text, unit])
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(1, Qt.UserRole, path)
            item.setToolTip(1, "model file value: double-click to change")
            self.table.addTopLevelItem(item)
        for key, text in props:
            item = QTreeWidgetItem([key, text, ""])
            item.setForeground(0, Qt.gray)
            item.setForeground(1, Qt.gray)
            self.table.addTopLevelItem(item)
        self.table.blockSignals(False)

    def property_rows(self):
        return [(self.table.topLevelItem(it).text(0),
                 self.table.topLevelItem(it).text(1))
                for it in range(self.table.topLevelItemCount())]

    def edit_property(self, label, text):
        """Change the value of an editable row as a user does (for
        scripts)."""
        for it in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(it)
            if item.text(0) == label and item.data(1, Qt.UserRole):
                item.setText(1, text)
                return
        raise KeyError(f"no editable property {label!r}")

    def _on_item_changed(self, item, column):
        path = item.data(1, Qt.UserRole)
        if 1 == column and path and self.edited is not None:
            self.edited(tuple(path), item.text(1))

    def show_adjusters(self, values, present=None):
        """Show ``values`` (mm) without reporting them as changes; the
        adjusters not in ``present`` (all by default) are disabled."""
        for name, spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(values[name])
            spin.setEnabled(present is None or name in present)
            spin.blockSignals(False)

    def show_alignment(self, text):
        self.alignment.setText(text)

    def _on_changed(self, name, value):
        if self.changed is not None:
            self.changed(name, value)


class WarningList(QListWidget):
    """The warnings of the last analysis, the newest run only.

    :ivar picked: Called with the joint name of a warning that names one,
        when set.
    """

    entries_changed = Signal(int, int)

    MARKS = {'error': "ERROR", 'warning': "WARNING", 'info': "info"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.picked = None
        self.setWordWrap(True)
        self.itemClicked.connect(self._on_clicked)

    def show_entries(self, entries):
        self.clear()
        for entry in entries:
            mark = self.MARKS.get(entry['level'], entry['level'])
            joint = entry.get('joint')
            where = f" [{joint}]" if joint else ""
            item = QListWidgetItem(f"{mark}{where}: {entry['message']}")
            item.setData(Qt.UserRole, joint)
            if 'error' == entry['level']:
                item.setForeground(Qt.red)
            elif 'warning' == entry['level']:
                item.setForeground(Qt.darkYellow)
            self.addItem(item)
        problems = sum(entry['level'] in ('warning', 'error')
                       for entry in entries)
        self.entries_changed.emit(len(entries), problems)

    def texts(self):
        return [self.item(it).text() for it in range(self.count())]

    def _on_clicked(self, item):
        joint = item.data(Qt.UserRole)
        if joint and self.picked is not None:
            self.picked(joint)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
