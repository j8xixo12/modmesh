"""Synchronized GLB and Suspension K&C views of multibody step steer."""

import argparse
import time
from pathlib import Path
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from solvcon.apps.golf_r.body_dynamics import run, channels, export_csv
from solvcon.apps.golf_r.multibody_pose import transforms
from solvcon.apps.golf_r.visual_pose import apply, wheel_groups
from solvcon import apps, pilot  # noqa: E402,F401
from solvcon.apps.modelview import world_matrices  # noqa: E402
from solvcon.apps.suspension import (
    CornerView,
    vehicle_geometries,
    vehicle_view_pose,
    vehicle_body_segments,
)


class History(QWidget):
    def __init__(self, times, values):
        super().__init__()
        self.times, self.values = times, values
        self.index = 0
        self.setMinimumHeight(160)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#131c29'))
        w, h = self.width(), self.height()
        margin = 25
        series = self.values[:, 3:7] * 1000
        limit = max(1.0, float(np.max(np.abs(series)))) * 1.1

        def point(t, value):
            return QPointF(
                margin + t / self.times[-1] * (w - 2 * margin),
                h / 2 - value / limit * (h / 2 - margin),
            )

        painter.setPen(QPen(QColor('#526070'), 1))
        painter.drawLine(point(0, 0), point(self.times[-1], 0))
        colors = ('#f39c60', '#60b9ff', '#d1a4ff', '#75d59d')
        for col, color in enumerate(colors):
            painter.setPen(QPen(QColor(color), 2))
            path = QPainterPath(point(self.times[0], series[0, col]))
            for t, value in zip(self.times[1:], series[1:, col]):
                path.lineTo(point(t, value))
            painter.drawPath(path)
        x = point(self.times[self.index], 0).x()
        painter.setPen(QPen(QColor('#ffffff'), 1, Qt.DashLine))
        painter.drawLine(QPointF(x, margin / 2), QPointF(x, h - margin / 2))
        painter.setPen(QColor('#dbe6ef'))
        painter.drawText(
            8, 14, f'Wheel travel vs chassis (mm), +/- {limit:.1f}'
        )
        painter.drawText(
            8, h - 5, 'FL orange   FR blue   RL purple   RR green'
        )


class Playback:
    def __init__(
        self, viewer, model, suspension, data, speed=12.0, rack=-0.004
    ):
        self.viewer, self.suspension, self.data = viewer, suspension, data
        self.input_speed, self.input_rack = speed, rack
        self.times, self.poses = data['pose_times'], data['poses']
        self.values = channels(self.poses)
        self.speed = np.interp(self.times, data['time'], data['speed'])
        self.yaw_rate = np.interp(self.times, data['time'], data['yaw_rate'])
        self.steer = np.interp(self.times, data['time'], data['steer'][:, 0])
        self.spin = np.zeros(len(self.times), dtype='float64')
        self.spin[1:] = np.cumsum(
            np.diff(self.times) * (self.speed[:-1] + self.speed[1:]) / 0.64
        )
        self.groups, self.pivots = wheel_groups(model, world_matrices(model))
        self.geometries = vehicle_geometries(data['models'])
        suspension.build(self.geometries)
        self.elapsed, self.started, self.playing = 0.0, time.monotonic(), True
        self._controls()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.render(0)

    def _controls(self):
        v = self.viewer
        v.tree.hide()
        v.selection.hide()
        v.summary.setText(
            f'MULTIBODY STEP STEER | {self.input_speed:.1f} m/s | '
            f'rack: {self.input_rack * 1000:+.1f} mm'
        )
        v.notices.setText(
            'Experimental candidate geometry and parameters; '
            'wheel spin is illustrative, not a solved state.'
        )
        v.subwin.setWindowTitle('Golf R | Body dynamics')
        layout = v.subwin.widget().layout()
        row = QHBoxLayout()
        self.button = QPushButton('Pause')
        self.button.clicked.connect(self.toggle)
        restart = QPushButton('Restart')
        restart.clicked.connect(self.restart)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(self.times) - 1)
        self.slider.sliderPressed.connect(self.pause)
        self.slider.valueChanged.connect(self.seek)
        for item in (self.button, restart, self.slider):
            row.addWidget(item)
        layout.addLayout(row)
        self.telemetry = QLabel()
        self.telemetry.setWordWrap(True)
        layout.addWidget(self.telemetry)
        self.history = History(self.times, self.values)
        layout.addWidget(self.history)

    def pause(self):
        if self.playing:
            self.elapsed = min(
                self.elapsed + time.monotonic() - self.started,
                float(self.times[-1]),
            )
            self.playing = False
            self.button.setText('Play')

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            if self.elapsed >= self.times[-1]:
                self.elapsed = 0.0
            self.started, self.playing = time.monotonic(), True
            self.button.setText('Pause')

    def restart(self):
        self.elapsed, self.started, self.playing = 0.0, time.monotonic(), True
        self.button.setText('Pause')
        self.render(0)

    def seek(self, value):
        self.elapsed, self.started = float(self.times[value]), time.monotonic()
        self.render(value)

    def tick(self):
        if self.viewer.widget is None:
            self.timer.stop()
            return
        if not self.playing:
            return
        elapsed = self.elapsed + time.monotonic() - self.started
        index = min(
            int(np.searchsorted(self.times, elapsed, side='right') - 1),
            len(self.times) - 1,
        )
        self.render(max(0, index))
        if elapsed >= self.times[-1]:
            self.pause()

    def render(self, i):
        pose, values = self.poses[i], self.values[i]
        apply(
            self.viewer.widget,
            transforms(
                self.viewer.items,
                self.groups,
                self.pivots,
                pose,
                self.poses[0],
                self.spin[i],
            ),
        )
        if self.suspension.widget is not None:
            model_pose = vehicle_view_pose(pose)
            body = vehicle_body_segments(self.geometries, model_pose)
            self.suspension.show_pose(
                self.geometries, model_pose, overlays={'vehicle_body': body}
            )
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.history.index = i
        self.history.update()
        z, roll, pitch, fl, fr, rl, rr = values
        self.telemetry.setText(
            f't={self.times[i]:.2f} s   speed={
                self.speed[i] * 3.6:.1f} km/h   '
            f'front steer={np.degrees(self.steer[i]):+.2f} deg   '
            f'yaw rate={np.degrees(self.yaw_rate[i]):+.2f} deg/s\n'
            f'Body heave={
                z * 1000:+.2f} mm   roll={np.degrees(roll):+.3f} deg   '
            f'pitch={np.degrees(pitch):+.3f} deg\n'
            f'Wheel bump vs body (mm): FL {
                fl * 1000:+.2f}   FR {fr * 1000:+.2f}   '
            f'RL {rl * 1000:+.2f}   RR {rr * 1000:+.2f}'
        )


def main():
    parser = argparse.ArgumentParser(
        description='Golf R multibody step-steer demo'
    )
    parser.add_argument('--speed-kmh', type=float, default=43.2)
    parser.add_argument('--rack-mm', type=float, default=-4.0)
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='directory containing suspensions.ini',
    )
    parser.add_argument(
        '--model', type=Path, required=True, help='car GLB file'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('body_dynamics.csv')
    )
    args = parser.parse_args()
    if (
        not np.isfinite([args.speed_kmh, args.rack_mm]).all()
        or args.speed_kmh <= 0
    ):
        parser.error('speed must be positive and inputs must be finite')
    print(
        'Running full-vehicle multibody step-steer simulation...', flush=True
    )
    speed, rack = args.speed_kmh / 3.6, args.rack_mm / 1000
    data = run(args.data_dir, speed=speed, rack=rack)
    export_csv(data, args.output)
    controller = pilot.controller
    controller.build()
    viewer = controller.apps['modelview']
    model = viewer.load_path(args.model)
    viewer.widget.projection = 'perspective'
    viewer.widget.cameraTarget = [0.0, 0.0, 0.55]
    viewer.widget.cameraPosition = [5.8, -7.0, 3.0]
    viewer.widget.cameraUp = [0.0, 0.0, 1.0]
    suspension = CornerView(controller.mgr)
    suspension.open('Golf R | Suspension K&C multibody geometry')
    playback = Playback(viewer, model, suspension, data, speed, rack)
    controller.mgr.mdiArea.tileSubWindows()
    controller.launch()
    return playback


if __name__ == '__main__':
    main()
