# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Shared control styling for pilot workspaces."""

WORKSPACE_STYLE = (
    "QTabWidget::pane { border: 0; } "
    "QTabBar::tab { padding: 7px 12px; border: 0; "
    "border-bottom: 2px solid transparent; background: transparent; } "
    "QTabBar::tab:selected { border-bottom: 2px solid #579bd4; "
    "font-weight: 600; } "
    "QTabBar::tab:hover { background: rgba(128,128,128,25); } "
    "QGroupBox { border: 1px solid rgba(128,128,128,70); "
    "border-radius: 6px; margin-top: 12px; padding: 8px; } "
    "QGroupBox::title { subcontrol-origin: margin; left: 10px; } "
    "QPushButton { padding: 5px 10px; } "
    "QPushButton[primary=\"true\"] { background: #266cad; "
    "color: white; border: 1px solid #266cad; border-radius: 4px; "
    "font-weight: 600; } "
    "QPushButton[primary=\"true\"]:hover { background: #327fbe; } "
    "QPushButton[primary=\"true\"]:pressed { background: #1b5187; } "
    "QPushButton[primary=\"true\"]:disabled { "
    "background: palette(button); color: palette(mid); "
    "border-color: palette(mid); } "
    "QPushButton:focus, QToolButton:focus { "
    "border: 1px solid #579bd4; } "
    "QComboBox, QSpinBox, QDoubleSpinBox { padding: 3px; } "
    "QToolButton { padding: 3px 6px; } "
    "QTreeWidget, QListWidget { "
    "border: 1px solid rgba(128,128,128,70); } "
    "QLineEdit { padding: 4px; } "
    "QTreeView::item, QListView::item { padding: 3px 2px; }")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
