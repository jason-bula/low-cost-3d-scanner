#!/usr/bin/env python3
"""LiDAR Scanner GUI — Pandar 40P + SimpleFOC motor control."""

import glob
import os
import sys
import time

# Add parent directory to path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PySide6.QtCore import QThread, QTimer, Signal, Slot
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog

from calibration import CalibrationWindow
from liveview import LiveViewWindow

import socket

import numpy as np
import serial
import serial.tools.list_ports

from record_lidar import VALID_SIZES, load_calibration, write_pcd
from scan_lidar import (
    load_calibration_file,
    parse_packets_with_motor,
)


# ── Scan Worker (runs in QThread) ────────────────────────────────────────

class ScanWorker(QThread):
    """Executes the full scan sequence in a background thread."""

    log = Signal(str)
    status = Signal(str)
    progress = Signal(int)          # 0-100
    finished_ok = Signal(str)       # output path
    finished_err = Signal(str)      # error message

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        self._ser = None
        try:
            self._run_scan()
        except Exception as e:
            self.finished_err.emit(str(e))
        finally:
            # Safety: always try to stop the motor
            if self._ser is not None:
                try:
                    self._ser.write(b"X\n")
                    self._ser.close()
                except Exception:
                    pass

    def _run_scan(self):
        p = self.p

        # ── 1. Calibration LiDAR ──
        self.status.emit("Chargement calibration...")
        self.log.emit(f"Connexion PTC {p['ip']}:9347 ...")
        cal = load_calibration(p["ip"])
        self.log.emit("Calibration OK")
        self.progress.emit(5)

        # ── 2. Calibration de montage (optionnel) ──
        R_calib, t_calib = None, None
        if p["apply_calibration"]:
            self.log.emit(f"Chargement {p['calibration_file']} ...")
            R_calib, t_calib = load_calibration_file(p["calibration_file"])
            if R_calib is None:
                self.finished_err.emit(
                    f"Impossible de charger {p['calibration_file']}")
                return
            self.log.emit("Correction de montage chargee")

        # ── 3. Ouvrir port serie + boot Arduino ──
        self.status.emit("Connexion moteur...")
        self.log.emit(f"Ouverture {p['serial']} ...")
        ser = serial.Serial(p["serial"], 115200, timeout=1)
        self._ser = ser          # expose for finally block
        time.sleep(2.0)
        while ser.in_waiting:
            ser.readline()
        self.log.emit("Port serie OK")
        self.progress.emit(10)

        if self._stop_requested:
            self.finished_err.emit("Annule")
            return

        # ── 4. Demarrer moteur ──
        cmd = f"R{p['rpm']}\n"
        self.status.emit("Stabilisation moteur...")
        self.log.emit(f"Envoi: {cmd.strip()}")
        ser.write(cmd.encode())
        # Wait 3s for stabilization, checking stop every 0.5s
        for _ in range(6):
            if self._stop_requested:
                self.finished_err.emit("Annule")
                return
            time.sleep(0.5)
        self.progress.emit(15)

        # ── 5. Capture UDP (inline, no signal.signal) ──
        self.status.emit("Capture en cours...")
        duration = p["duration"]
        self.log.emit(f"Capture UDP :{p['udp_port']} pour {duration}s ...")

        self._capture_start = time.monotonic()
        self._capture_duration = duration

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        sock.bind(("0.0.0.0", p["udp_port"]))
        sock.settimeout(2.0)

        packets_ts = []
        t_start = time.monotonic()
        deadline = t_start + duration

        while not self._stop_requested and time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(2048)
                t_pkt = time.monotonic()
            except socket.timeout:
                continue
            if len(data) in VALID_SIZES:
                packets_ts.append((t_pkt, data))

        sock.close()
        elapsed = time.monotonic() - t_start
        pps = len(packets_ts) / max(elapsed, 0.001)
        self.log.emit(f"Capture: {len(packets_ts):,} paquets en {elapsed:.1f}s ({pps:.0f} pkt/s)")
        self.progress.emit(70)

        # ── 6. Stop moteur ──
        self.status.emit("Arret moteur...")
        ser.write(b"X\n")
        time.sleep(0.5)

        # ── 7. Retour (optionnel) ──
        if p["return_rpm"] > 0:
            scan_angle = p["rpm"] * 6.0 * duration
            return_time = scan_angle / (p["return_rpm"] * 6.0)
            cmd_ret = f"R-{p['return_rpm']}\n"
            self.status.emit("Retour en cours...")
            self.log.emit(
                f"Retour: {cmd_ret.strip()} pour {return_time:.1f}s")
            ser.write(cmd_ret.encode())
            time.sleep(return_time + 1.0)
            ser.write(b"X\n")
            time.sleep(0.5)

        ser.close()
        self._ser = None  # already closed, prevent finally double-close
        self.progress.emit(75)

        if not packets_ts:
            self.finished_err.emit("Aucun paquet capture")
            return

        # ── 8. Parsing ──
        self.status.emit("Traitement des points...")
        self.log.emit("Parsing avec rotation moteur...")
        motor_data = []  # open-loop
        need_channels = p.get("central_bands", False)
        result = parse_packets_with_motor(
            packets_ts, t_start, motor_data, cal, p["rpm"],
            block_interp=p["block_interp"],
            R_calib=R_calib, t_calib=t_calib,
            split_lidar_coords=need_channels,
            return_channels=need_channels,
        )

        if need_channels:
            # result = (pts_front, ch_front, pts_back, ch_back)
            pts_front, ch_front, _, _ = result
            # Keep only channels 10-30
            mask = (ch_front >= 10) & (ch_front <= 30)
            pts = pts_front[mask]
            self.log.emit(
                f"Parse: {len(pts_front):,} pts avant, "
                f"filtre ch 10-30 -> {len(pts):,} points")
        else:
            pts = result
            self.log.emit(f"Parse: {len(pts):,} points")
        self.progress.emit(90)

        # ── 9. Decimation ──
        if p["decimate"] > 1:
            pts = pts[:: p["decimate"]]
            self.log.emit(
                f"Decimation 1/{p['decimate']} -> {len(pts):,} points")

        # ── 10. Ecriture PCD ──
        self.status.emit("Ecriture fichier...")
        output = p["output"]
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        write_pcd(output, pts, p["ascii"])
        sz_mb = os.path.getsize(output) / (1024 * 1024)
        self.log.emit(f"Ecrit: {output} ({len(pts):,} pts, {sz_mb:.1f} MB)")
        self.progress.emit(100)
        self.finished_ok.emit(output)


# ── Main Window ──────────────────────────────────────────────────────────

class ScanApp:
    """Loads the .ui and wires up all signals."""

    def __init__(self):
        loader = QUiLoader()
        ui_path = os.path.join(os.path.dirname(__file__), "scan_gui.ui")
        self.w = loader.load(ui_path)
        self.worker = None
        self._calib_window = None
        self._liveview_window = None
        self._progress_timer = QTimer()
        self._progress_timer.setInterval(500)
        self._progress_timer.timeout.connect(self._update_capture_progress)
        self._connect()
        self._refresh_serial()
        self._update_angle()

    # ── Setup ──

    def _connect(self):
        w = self.w
        w.btnLiveView.clicked.connect(self._open_liveview)
        w.btnCalibration.clicked.connect(self._open_calibration)
        w.btnStart.clicked.connect(self._start_scan)
        w.btnStop.clicked.connect(self._stop_scan)
        w.btnBrowse.clicked.connect(self._browse_output)
        w.btnBrowseCalib.clicked.connect(self._browse_calib)
        w.btnRefreshSerial.clicked.connect(self._refresh_serial)
        w.checkCalibration.toggled.connect(self._toggle_calib)
        w.checkReturn.toggled.connect(self._toggle_return)
        w.spinDuration.valueChanged.connect(self._update_angle)
        w.spinRPM.valueChanged.connect(self._update_angle)

    def _refresh_serial(self):
        combo = self.w.comboSerial
        combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in sorted(ports, key=lambda x: x.device):
            combo.addItem(p.device)
        # Default to /dev/ttyUSB0 if present
        idx = combo.findText("/dev/ttyUSB0")
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _open_liveview(self):
        if self._liveview_window is None:
            ip = self.w.editIP.text().strip()
            self._liveview_window = LiveViewWindow(lidar_ip=ip)
        self._liveview_window.show()
        self._liveview_window.raise_()
        self._liveview_window.activateWindow()

    def _open_calibration(self):
        self._calib_window = CalibrationWindow(self.w)
        self._calib_window.show()

    def _toggle_return(self, checked):
        self.w.spinReturnRPM.setEnabled(checked)

    def _toggle_calib(self, checked):
        self.w.editCalibrationFile.setEnabled(checked)
        self.w.btnBrowseCalib.setEnabled(checked)

    def _update_angle(self):
        rpm = self.w.spinRPM.value()
        dur = self.w.spinDuration.value()
        angle = rpm * 6.0 * dur
        self.w.labelAngleValue.setText(f"{angle:.1f}\u00b0")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self.w, "Fichier de sortie", self.w.editOutput.text(),
            "Point Cloud (*.pcd);;Tous (*)")
        if path:
            self.w.editOutput.setText(path)

    def _browse_calib(self):
        path, _ = QFileDialog.getOpenFileName(
            self.w, "Fichier de calibration",
            self.w.editCalibrationFile.text(),
            "JSON (*.json);;Tous (*)")
        if path:
            self.w.editCalibrationFile.setText(path)

    # ── Scan lifecycle ──

    def _start_scan(self):
        w = self.w
        params = {
            "ip": w.editIP.text().strip(),
            "udp_port": w.spinUDPPort.value(),
            "serial": w.comboSerial.currentText().strip(),
            "duration": w.spinDuration.value(),
            "rpm": w.spinRPM.value(),
            "return_rpm": w.spinReturnRPM.value() if w.checkReturn.isChecked() else 0,
            "decimate": w.spinDecimate.value(),
            "block_interp": w.checkBlockInterp.isChecked(),
            "output": w.editOutput.text().strip(),
            "ascii": w.checkAscii.isChecked(),
            "apply_calibration": w.checkCalibration.isChecked(),
            "calibration_file": w.editCalibrationFile.text().strip(),
            "central_bands": w.checkCentralBands.isChecked(),
        }

        # Resolve output path relative to app/
        if not os.path.isabs(params["output"]):
            params["output"] = os.path.join(
                os.path.dirname(__file__), params["output"])

        w.btnStart.setEnabled(False)
        w.btnStop.setEnabled(True)
        w.progressBar.setValue(0)
        w.textLog.clear()

        self.worker = ScanWorker(params)
        self.worker.log.connect(self._on_log)
        self.worker.status.connect(self._on_status)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.finished_err.connect(self._on_error)
        self.worker.start()
        self._progress_timer.start()

    def _stop_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._on_log("Arret demande...")

    def _cleanup(self):
        self._progress_timer.stop()
        self.w.btnStart.setEnabled(True)
        self.w.btnStop.setEnabled(False)
        self.worker = None

    # ── Slots ──

    @Slot(str)
    def _on_log(self, msg):
        self.w.textLog.appendPlainText(msg)

    @Slot(str)
    def _on_status(self, msg):
        self.w.labelStatus.setText(msg)

    @Slot(int)
    def _on_progress(self, val):
        self.w.progressBar.setValue(val)

    @Slot(str)
    def _on_done(self, path):
        self._on_status(f"Termine : {os.path.basename(path)}")
        self._on_log(f"\nScan termine avec succes !")
        self._cleanup()

    @Slot(str)
    def _on_error(self, msg):
        self._on_status(f"Erreur : {msg}")
        self._on_log(f"\nERREUR: {msg}")
        self._cleanup()

    def _update_capture_progress(self):
        """Update progress bar during capture phase."""
        if self.worker is None:
            return
        t_start = getattr(self.worker, "_capture_start", None)
        t_dur = getattr(self.worker, "_capture_duration", None)
        if t_start is None or t_dur is None:
            return
        elapsed = time.monotonic() - t_start
        # Capture is progress 15-70
        pct = 15 + min(55, int(55 * elapsed / t_dur))
        current = self.w.progressBar.value()
        if 15 <= pct <= 70 and pct > current:
            self.w.progressBar.setValue(pct)
            remaining = max(0, t_dur - elapsed)
            self.w.labelStatus.setText(
                f"Capture en cours... {remaining:.0f}s restantes")

    def show(self):
        self.w.show()


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    scanner = ScanApp()
    scanner.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
