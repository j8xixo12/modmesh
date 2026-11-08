# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Inspect static GLB scenes in a pilot sub-window."""

from pathlib import Path
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QLabel, QMessageBox, QPushButton,
                               QHBoxLayout, QSplitter, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ... import core
from ...pilot import base
from . import _gltf, _scene


class ModelViewerApp(base.PilotFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = None
        self.subwin = None
        self.tree = None
        self.model = None
        self.items = []
        self.node_items = {}
        self.path = None
        self.elapsed = 0.0
        self._mdi = None

    def populate_menu(self):
        self.add_action('App', text='Model Viewer',
                        tip='Inspect static GLB models', func=self.open,
                        id='app.modelview', weight=25)

    def open(self):
        path, _ = QFileDialog.getOpenFileName(
            self._mainWindow, 'Open GLB', '', 'Binary glTF (*.glb)')
        if path:
            try:
                self.load_path(path)
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self._mainWindow, 'Cannot open GLB',
                                    str(exc))

    def _open_window(self):
        if self.widget is not None:
            self._mdi.setActiveSubWindow(self.subwin)
            return
        self.widget = self._mgr.add3DWidget()
        self.widget.showAxis(True)
        self._mdi = self._mgr.mdiArea
        self.subwin = self._mdi.activeSubWindow()
        self.subwin.setAttribute(Qt.WA_DeleteOnClose, True)
        self._close_filter = base.SubWindowCloseFilter(
            self._closed, self.subwin)
        self.subwin.installEventFilter(self._close_filter)
        host = self.subwin.widget()
        handle = next((c for c in host.findChildren(QWidget)
                       if c.metaObject().className()
                       == 'solvcon::RDomainWidget'), None)
        if handle is None:
            raise ValueError('Pilot 3D widget is unavailable')
        layout = host.layout()
        layout.removeWidget(handle)
        toolbar = QWidget()
        buttons = QHBoxLayout(toolbar)
        buttons.setContentsMargins(0, 0, 0, 0)
        for label, callback in [
                ('Open GLB', self.open),
                ('Fit', lambda: self.widget.fitCameraToScene()),
                *[(label, lambda checked=False, view=view:
                   self.widget.setView(view))
                  for label, view in [('Front', '+x'), ('Side', '-y'),
                                      ('Top', '+z'), ('Iso', 'iso')]]]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Part', 'Triangles / material'])
        self.tree.setMinimumWidth(260)
        self.tree.itemChanged.connect(self._visibility)
        self.tree.currentItemChanged.connect(self._selection)
        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(handle)
        splitter.setStretchFactor(1, 1)
        self.summary = QLabel()
        self.selection = QLabel()
        self.notices = QLabel()
        self.notices.setWordWrap(True)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        for label in (self.summary, self.selection, self.notices):
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            panel_layout.addWidget(label)
        layout.addWidget(toolbar)
        layout.addWidget(splitter, 1)
        layout.addWidget(panel)
        self.subwin.resize(1100, 760)

    def _closed(self):
        self.widget = None
        self.subwin = None
        self.tree = None
        self.node_items = {}

    @staticmethod
    def _arrays(item):
        return (core.SimpleArrayFloat32(array=item.positions),
                core.SimpleArrayFloat32(array=item.normals),
                core.SimpleArrayUint32(array=item.indices))

    def _draw(self, items):
        self.widget.clearObjects()
        for item in items:
            if item.texture is not None:
                pos, normal, indices = self._arrays(item)
                uv = core.SimpleArrayFloat32(array=item.texcoords)
                self.widget.addTexturedSurfaceObject(
                    item.key, pos, normal, uv, indices, item.texture,
                    item.alpha_mode, item.alpha_cutoff, item.color[3],
                    item.wrap_s, item.wrap_t)
                self.widget.setObjectColor(item.key, *item.color[:3])
            else:
                self.widget.addSurfaceObject(item.key, *self._arrays(item))
                # The legacy lit shader takes display-space color values.
                color = [12.92 * v if v <= 0.0031308 else
                         1.055 * v**(1 / 2.4) - 0.055 for v in item.color[:3]]
                self.widget.setObjectColor(item.key, *color)
                if item.alpha_mode == 'BLEND':
                    self.widget.setObjectOpacity(item.key, item.color[3])

    def load_path(self, path):
        """Prepare a scene completely before replacing the displayed model."""
        start = time.perf_counter()
        model = _gltf.load_glb(path)
        items = _scene.surfaces(model)
        lo, hi = _scene.bounds(items)
        self._open_window()
        try:
            self._draw(items)
        except Exception:
            self._draw(self.items)
            raise
        self.model, self.items, self.path = model, items, Path(path)
        self.elapsed = time.perf_counter() - start
        self._fill_tree()
        self.subwin.setWindowTitle(f'Model Viewer: {self.path.name}')
        self.summary.setText(
            f'{len(model.nodes)} nodes | {len(items)} surfaces | '
            f'{sum(len(s.indices) for s in items):,} triangles | '
            f'Size [m]: {hi[0] - lo[0]:.4f} x {hi[1] - lo[1]:.4f} x '
            f'{hi[2] - lo[2]:.4f} | Prepared in {self.elapsed:.3f} s')
        self.selection.setText('Select a part to inspect its geometry.')
        self.notices.setText('Base-color materials; simplified lighting. '
                             + ' | '.join(model.notices))
        self.widget.setView('iso')
        return model

    def _fill_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        self.node_items = {}
        grouped = {}
        for surface in self.items:
            grouped.setdefault(surface.node, []).append(surface)

        def insert(index, parent):
            node = self.model.nodes[index]
            own = grouped.get(index, [])
            total = sum(len(s.indices) for s in own)
            text = ''
            if own:
                names = dict.fromkeys(s.material for s in own)
                text += ' / ' + ', '.join(names)
            item = QTreeWidgetItem(parent, [node.name, text])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked)
            item.setData(0, Qt.UserRole, index)
            self.node_items[index] = item
            for child in node.children:
                total += insert(child, item)
            item.setText(1, f'{total:,}' + text)
            return total

        for index in self.model.roots:
            insert(index, self.tree)
        self.tree.expandToDepth(1)
        self.tree.blockSignals(False)

    def _visibility(self, item, column):
        self.tree.blockSignals(True)
        stack = [item]
        while stack:
            current = stack.pop()
            current.setCheckState(0, item.checkState(0))
            stack.extend(current.child(i) for i in range(current.childCount()))
        self.tree.blockSignals(False)
        for surface in self.items:
            current = self.node_items[surface.node]
            visible = True
            while current is not None:
                visible = visible and current.checkState(0) != Qt.Unchecked
                current = current.parent()
            self.widget.setObjectVisible(surface.key, visible)

    def _selection(self, item, previous):
        if item is None:
            return
        index = item.data(0, Qt.UserRole)
        descendants = set()
        stack = [index]
        while stack:
            current = stack.pop()
            descendants.add(current)
            stack.extend(self.model.nodes[current].children)
        group = [s for s in self.items if s.node in descendants]
        if not group:
            self.selection.setText(f'{item.text(0)}: hierarchy node')
            return
        lo, hi = _scene.bounds(group)
        self.selection.setText(
            f'{item.text(0)} | {sum(len(s.positions) for s in group):,} '
            f'vertices | {sum(len(s.indices) for s in group):,} triangles | '
            f'Extent [m]: {hi[0] - lo[0]:.4f}, {hi[1] - lo[1]:.4f}, '
            f'{hi[2] - lo[2]:.4f}')


def build_app(controller):
    return ModelViewerApp(mgr=controller.mgr)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
