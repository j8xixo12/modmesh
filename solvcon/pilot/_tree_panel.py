# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Unified inspector dock that stacks the pilot trees."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDockWidget, QStackedWidget

from . import _gui_common
from ._mesh_info import MeshInfoTree
from ._entity_tree import EntityTreeWidget

__all__ = [  # noqa: F822
    'TreePanel',
]


class TreePanel(_gui_common.PilotFeature):
    """Unified inspector dock that follows the active sub-window.

    One dock holds both trees in a stack. A 3D mesh viewer shows the mesh
    information tree; the 2D canvas shows the world entity tree. The active
    sub-window's type selects which tree is shown, so the two panels that
    used to be separate now share one widget.
    """

    def __init__(self, *args, **kw):
        self._status = kw.pop('style_status')
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._stack = None
        self._mesh_tree = None
        self._entity_tree = None

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Inspector", "Toggle the inspector panel",
            None, id="panel.inspector", weight=10, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        """Show or hide the panel."""
        if checked:
            self._ensure_panel()
            self._sync()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        """Build the dock lazily and follow sub-window activation."""
        if self._stack is not None:
            return
        self._mesh_tree = MeshInfoTree(self._status)
        self._mesh_tree.boundary_toggled = self._on_boundary_toggled
        self._mesh_tree.edges_toggled = self._on_edges_toggled
        self._mesh_tree.normals_toggled = self._on_normals_toggled
        self._entity_tree = EntityTreeWidget()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._mesh_tree)
        self._stack.addWidget(self._entity_tree)
        self._dock = QDockWidget("inspector")
        self._dock.setWidget(self._stack)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           self._dock)
        self._dock.visibilityChanged.connect(self._action.setChecked)
        mdi = self._mdi_area()
        if mdi is not None:
            mdi.subWindowActivated.connect(self._on_subwindow_activated)

    def _on_subwindow_activated(self, _subwin):
        """Re-select the tree when the active sub-window changes."""
        if self._dock is not None and self._action.isChecked():
            QTimer.singleShot(0, self._sync)

    def _sync(self):
        """Select the tree that matches the active sub-window.

        A 3D viewer shows the mesh tree; the 2D canvas shows the world
        tree. Detection keys on which viewer the manager reports active.
        """
        widget3d = self._mgr.currentR3DWidget()
        if widget3d is not None:
            self._stack.setCurrentWidget(self._mesh_tree)
            self._mesh_tree.set_mesh(widget3d.mesh)
            return
        widget2d = self._mgr.currentR2DWidget()
        if widget2d is not None:
            self._stack.setCurrentWidget(self._entity_tree)
            self._entity_tree.set_world(widget2d.world)

    def _on_boundary_toggled(self, ibc, checked):
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showBoundary(ibc, checked)

    def _on_edges_toggled(self, checked):
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showFeatureEdges(checked)

    def _on_normals_toggled(self, checked):
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showNormals(checked)

    def _mdi_area(self):
        return self._mainWindow.centralWidget()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
