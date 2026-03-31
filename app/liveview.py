"""Live 3D view of Pandar 40P LiDAR stream."""

import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from record_lidar import (
    BLOCK_SIZE,
    BLOCKS_PER_PKT,
    CHANNELS,
    VALID_SIZES,
    load_calibration,
)

# ── Colors by elevation (channel index) ─────────────────────────────────

def _make_channel_colors():
    """Create a color map: low channels (ground) = blue, high = red."""
    cmap = np.zeros((CHANNELS, 4), dtype=np.float32)
    for i in range(CHANNELS):
        t = i / (CHANNELS - 1)  # 0 = top (ch0, +15°), 1 = bottom (ch39, -25°)
        # Blue -> Cyan -> Green -> Yellow -> Red
        if t < 0.25:
            r, g, b = 0, t * 4, 1
        elif t < 0.5:
            r, g, b = 0, 1, 1 - (t - 0.25) * 4
        elif t < 0.75:
            r, g, b = (t - 0.5) * 4, 1, 0
        else:
            r, g, b = 1, 1 - (t - 0.75) * 4, 0
        cmap[i] = [r, g, b, 0.8]
    return cmap


CHANNEL_COLORS = _make_channel_colors()


# ── Packet parser (single packet -> Nx3 + Nx4 colors) ───────────────────

def parse_packet(raw, cal):
    """Parse one UDP packet -> (points_Nx3, colors_Nx4)."""
    cos_elev = cal['cos_elev']
    sin_elev = cal['sin_elev']
    az_off_rad = cal['az_off_rad']

    pts = []
    cols = []

    for b in range(BLOCKS_PER_PKT):
        off = b * BLOCK_SIZE
        if len(raw) < off + 4 or raw[off] != 0xFF or raw[off + 1] != 0xEE:
            continue

        az_raw = struct.unpack_from("<H", raw, off + 2)[0]
        az_base = np.float64(az_raw) * 0.01 * (np.pi / 180.0)

        if len(raw) < off + 4 + CHANNELS * 3:
            continue

        ch_data = np.frombuffer(raw, dtype=np.uint8,
                                count=CHANNELS * 3, offset=off + 4)
        ch = ch_data.reshape(CHANNELS, 3)
        dist_raw = (ch[:, 0].astype(np.uint16) |
                    (ch[:, 1].astype(np.uint16) << 8))

        mask = dist_raw > 0
        if not mask.any():
            continue

        indices = np.where(mask)[0]
        dist_m = dist_raw[mask].astype(np.float64) * 0.004
        az = az_base + az_off_rad[mask]
        cos_e = cos_elev[mask]
        sin_e = sin_elev[mask]

        x = dist_m * cos_e * np.sin(az)
        y = dist_m * cos_e * np.cos(az)
        z = dist_m * sin_e

        block_pts = np.column_stack([x, y, z]).astype(np.float32)
        block_cols = CHANNEL_COLORS[indices]

        pts.append(block_pts)
        cols.append(block_cols)

    if pts:
        return np.vstack(pts), np.vstack(cols)
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)


# ── Capture thread ───────────────────────────────────────────────────────

class LidarCapture(threading.Thread):
    """Captures UDP packets and parses them into point buffers."""

    def __init__(self, cal, udp_port=2368, acc_time=0.1):
        super().__init__(daemon=True)
        self.cal = cal
        self.udp_port = udp_port
        self.acc_time = acc_time  # accumulation time per frame (seconds)
        self._stopped = False

        # Double buffer: _front is read by display, _back is written by capture
        self._front_pts = np.empty((0, 3), dtype=np.float32)
        self._front_cols = np.empty((0, 4), dtype=np.float32)
        self._lock = threading.Lock()
        self._new_frame = False
        self.pps = 0

    def request_stop(self):
        self._stopped = True

    def get_frame(self):
        """Return (pts, colors, is_new). Non-blocking."""
        with self._lock:
            new = self._new_frame
            self._new_frame = False
            return self._front_pts, self._front_cols, new

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        sock.bind(("0.0.0.0", self.udp_port))
        sock.settimeout(1.0)

        while not self._stopped:
            pts_list = []
            cols_list = []
            t0 = time.monotonic()
            n_pkts = 0

            # Accumulate for one frame period
            while not self._stopped and (time.monotonic() - t0) < self.acc_time:
                try:
                    data, _ = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                if len(data) not in VALID_SIZES:
                    continue
                n_pkts += 1
                p, c = parse_packet(data, self.cal)
                if len(p) > 0:
                    pts_list.append(p)
                    cols_list.append(c)

            elapsed = time.monotonic() - t0
            self.pps = n_pkts / max(elapsed, 0.001)

            if pts_list:
                all_pts = np.vstack(pts_list)
                all_cols = np.vstack(cols_list)
                with self._lock:
                    self._front_pts = all_pts
                    self._front_cols = all_cols
                    self._new_frame = True

        sock.close()


# ── 3D View Window ───────────────────────────────────────────────────────

class LiveViewWindow(QWidget):
    """Separate window with 3D OpenGL live LiDAR view."""

    def __init__(self, lidar_ip="192.168.20.13", parent=None):
        super().__init__(parent)
        self.setWindowTitle("LiDAR Live View")
        self.resize(1000, 700)

        self._capture = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_view)

        self._build_ui()
        self._lidar_ip = lidar_ip

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Toolbar ──
        toolbar = QHBoxLayout()

        self.btnStart = QPushButton("Start")
        self.btnStart.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 4px; }")
        self.btnStart.clicked.connect(self._start)
        toolbar.addWidget(self.btnStart)

        self.btnStop = QPushButton("Stop")
        self.btnStop.setEnabled(False)
        self.btnStop.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #9E9E9E; }")
        self.btnStop.clicked.connect(self._stop)
        toolbar.addWidget(self.btnStop)

        toolbar.addWidget(QLabel("  Acc. (ms):"))
        self.spinAcc = QSpinBox()
        self.spinAcc.setRange(50, 2000)
        self.spinAcc.setValue(100)
        self.spinAcc.setSingleStep(50)
        toolbar.addWidget(self.spinAcc)

        toolbar.addWidget(QLabel("  Point size:"))
        self.spinSize = QSpinBox()
        self.spinSize.setRange(1, 10)
        self.spinSize.setValue(2)
        self.spinSize.valueChanged.connect(self._update_point_size)
        toolbar.addWidget(self.spinSize)

        toolbar.addWidget(QLabel("  Max dist (m):"))
        self.spinMaxDist = QSpinBox()
        self.spinMaxDist.setRange(1, 200)
        self.spinMaxDist.setValue(50)
        toolbar.addWidget(self.spinMaxDist)

        self.checkDecimate = QCheckBox("Decimate x4")
        toolbar.addWidget(self.checkDecimate)

        toolbar.addStretch()

        self.labelStats = QLabel("Arrete")
        self.labelStats.setStyleSheet("color: #757575; font-style: italic;")
        toolbar.addWidget(self.labelStats)

        layout.addLayout(toolbar)

        # ── 3D View ──
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('k')
        self.view.setCameraPosition(distance=20, elevation=30, azimuth=45)
        self.view.setAttribute(pg.QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.view.setAttribute(pg.QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, False)

        # Grid
        grid = gl.GLGridItem()
        grid.setSize(50, 50)
        grid.setSpacing(5, 5)
        grid.setColor((0.3, 0.3, 0.3, 0.5))
        self.view.addItem(grid)

        # Axes
        ax = gl.GLAxisItem()
        ax.setSize(3, 3, 3)
        self.view.addItem(ax)

        # Scatter plot item
        self.scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            size=2,
            pxMode=True
        )
        self.view.addItem(self.scatter)

        layout.addWidget(self.view)

    def _start(self):
        cal = load_calibration(self._lidar_ip)
        acc = self.spinAcc.value() / 1000.0

        self._capture = LidarCapture(cal, acc_time=acc)
        self._capture.start()

        self._timer.start(int(acc * 1000))

        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.labelStats.setText("Demarrage...")

    def _stop(self):
        self._timer.stop()
        if self._capture:
            self._capture.request_stop()
            self._capture.join(timeout=3)
            self._capture = None

        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.labelStats.setText("Arrete")

    def _update_view(self):
        if not self._capture:
            return

        pts, cols, is_new = self._capture.get_frame()
        if not is_new or len(pts) == 0:
            return

        # Distance filter
        max_dist = self.spinMaxDist.value()
        dist_sq = pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2
        mask = dist_sq <= max_dist**2
        pts = pts[mask]
        cols = cols[mask]

        # Decimate
        if self.checkDecimate.isChecked() and len(pts) > 0:
            pts = pts[::4]
            cols = cols[::4]

        self.scatter.setData(pos=pts, color=cols, size=self.spinSize.value())

        self.labelStats.setText(
            f"{len(pts):,} pts | {self._capture.pps:.0f} pkt/s")

    def _update_point_size(self, val):
        self.scatter.setData(size=val)

    def closeEvent(self, event):
        self._stop()
        # Don't destroy GL items — just hide the window so it can be reused
        event.ignore()
        self.hide()
