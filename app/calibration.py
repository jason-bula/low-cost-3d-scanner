"""Calibration window — runs optimize_*.py scripts as subprocesses."""

import os
import socket
import struct
import sys
import time

from PySide6.QtCore import QProcess, QThread, QTimer, Signal, Slot
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QFileDialog

import serial

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration")

METHODS = {
    "v1": "optimize_angles.py",
    "v2": "optimize_roughness.py",
    "v3": "optimize_planes.py",
    "v4": "optimize_voxels.py",
    "v5": "optimize_p2plane.py",
}


PCAP_MAGIC = 0xA1B2C3D4
PCAP_VERSION_MAJOR = 2
PCAP_VERSION_MINOR = 4
PCAP_LINKTYPE_UDP = 228  # LINKTYPE_IPV4 (raw)
# We use LINKTYPE_USER0 (147) which is simpler — just raw UDP payload
# Actually use DLT_RAW (228 = raw IPv4) — but simplest is just raw data
# The optimize scripts use dpkt/scapy to read, let's check what they expect.
# Safest: write standard Ethernet/IP/UDP pcap. But since the optimize scripts
# likely just extract UDP payload, we use a minimal approach with raw link type.

VALID_SIZES = (1262, 1266)
UDP_PORT = 2368
RPM_RECORD = 0.1667  # 1°/s -> 360° in 6 min


class PcapRecordWorker(QThread):
    """Record a 360° slow scan to PCAP file for calibration."""

    log = Signal(str)
    status = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(str)
    finished_err = Signal(str)

    def __init__(self, serial_port, output_path, lidar_ip, udp_port=2368):
        super().__init__()
        self._serial_port = serial_port
        self._output_path = output_path
        self._lidar_ip = lidar_ip
        self._udp_port = udp_port
        self._stop_requested = False
        self._rpm = RPM_RECORD
        self._duration = 360.0 / (self._rpm * 6.0)  # seconds for 360°

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        self._ser = None
        try:
            self._run_record()
        except Exception as e:
            self.finished_err.emit(str(e))
        finally:
            if self._ser is not None:
                try:
                    self._ser.write(b"X\n")
                    self._ser.close()
                except Exception:
                    pass

    def _run_record(self):
        duration = self._duration

        # 1. Open serial
        self.status.emit("Connexion moteur...")
        self.log.emit(f"Ouverture {self._serial_port} ...")
        ser = serial.Serial(self._serial_port, 115200, timeout=1)
        self._ser = ser
        time.sleep(2.0)
        while ser.in_waiting:
            ser.readline()

        if self._stop_requested:
            self.finished_err.emit("Annule")
            return

        # 2. Start motor
        cmd = f"R{self._rpm}\n"
        self.status.emit("Stabilisation moteur...")
        self.log.emit(f"Moteur: {cmd.strip()} ({self._rpm * 6:.1f} deg/s)")
        ser.write(cmd.encode())
        for _ in range(6):
            if self._stop_requested:
                self.finished_err.emit("Annule")
                return
            time.sleep(0.5)

        # 3. Capture UDP and write PCAP
        self.status.emit(f"Enregistrement PCAP ({duration:.0f}s)...")
        self.log.emit(f"Capture {duration:.0f}s -> {self._output_path}")

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        sock.bind(("0.0.0.0", self._udp_port))
        sock.settimeout(2.0)

        with open(self._output_path, "wb") as f:
            # PCAP global header
            f.write(struct.pack(
                "<IHHiIII",
                PCAP_MAGIC,
                PCAP_VERSION_MAJOR, PCAP_VERSION_MINOR,
                0,          # thiszone
                0,          # sigfigs
                65535,      # snaplen
                PCAP_LINKTYPE_UDP,  # network (raw link type)
            ))

            t_start = time.monotonic()
            deadline = t_start + duration
            n_pkts = 0

            while not self._stop_requested and time.monotonic() < deadline:
                try:
                    data, _ = sock.recvfrom(2048)
                    t_now = time.time()
                except socket.timeout:
                    continue

                if len(data) not in VALID_SIZES:
                    continue

                # PCAP packet header
                ts_sec = int(t_now)
                ts_usec = int((t_now - ts_sec) * 1_000_000)
                f.write(struct.pack(
                    "<IIII", ts_sec, ts_usec, len(data), len(data)))
                f.write(data)
                n_pkts += 1

                # Progress update every ~1800 packets (~1s)
                if n_pkts % 1800 == 0:
                    elapsed = time.monotonic() - t_start
                    pct = min(95, int(100 * elapsed / duration))
                    self.progress.emit(pct)
                    remaining = max(0, duration - elapsed)
                    self.status.emit(
                        f"Enregistrement... {remaining:.0f}s restantes "
                        f"({n_pkts:,} paquets)")

        sock.close()

        # 4. Stop motor + return
        self.status.emit("Retour moteur...")
        ser.write(b"X\n")
        time.sleep(0.5)
        return_rpm = 4.0
        return_time = 360.0 / (return_rpm * 6.0)
        self.log.emit(f"Retour: R-{return_rpm} pour {return_time:.0f}s")
        ser.write(f"R-{return_rpm}\n".encode())
        time.sleep(return_time + 1.0)
        ser.write(b"X\n")
        time.sleep(0.5)
        ser.close()
        self._ser = None

        sz_mb = os.path.getsize(self._output_path) / (1024 * 1024)
        self.log.emit(
            f"PCAP ecrit: {self._output_path} "
            f"({n_pkts:,} paquets, {sz_mb:.1f} MB)")
        self.progress.emit(100)
        self.finished_ok.emit(self._output_path)


class CalibrationWindow:
    """Calibration dialog — launches optimize scripts via QProcess."""

    def __init__(self, parent=None):
        loader = QUiLoader()
        ui_path = os.path.join(os.path.dirname(__file__), "calibration_gui.ui")
        self.w = loader.load(ui_path, parent)
        self._process = None
        self._record_worker = None
        self._workflow_queue = []  # list of (method, params) for chained runs
        self._connect()

    def _connect(self):
        w = self.w
        w.btnBrowsePcap.clicked.connect(self._browse_pcap)
        w.btnBrowseOutput.clicked.connect(self._browse_output)
        w.btnRecordPcap.clicked.connect(self._record_pcap)
        w.btnRun.clicked.connect(self._run)
        w.btnStop.clicked.connect(self._stop)
        w.btnWorkflowFull.clicked.connect(self._workflow_full)
        w.btnWorkflowQuick.clicked.connect(self._workflow_quick)

    # ── PCAP recording ──

    def _record_pcap(self):
        """Start a 360° slow scan and record raw packets to PCAP."""
        w = self.w
        output = w.editPcap.text().strip()
        if not os.path.isabs(output):
            output = os.path.join(SCRIPTS_DIR, output)

        # Find serial port from main window (parent), or default
        serial_port = "/dev/ttyUSB0"
        parent = w.parent()
        if parent and hasattr(parent, "comboSerial"):
            serial_port = parent.comboSerial.currentText().strip()

        w.textLog.clear()
        w.progressBar.setValue(0)
        self._log(f"=== Enregistrement PCAP calibration ===")
        self._log(f"0.1667 RPM (1 deg/s) -> 360 deg en ~6 min")

        self._record_worker = PcapRecordWorker(
            serial_port, output, w.editIP.text().strip())
        self._record_worker.log.connect(self._log)
        self._record_worker.status.connect(
            lambda msg: w.labelStatus.setText(msg))
        self._record_worker.progress.connect(
            lambda val: w.progressBar.setValue(val))
        self._record_worker.finished_ok.connect(self._on_record_done)
        self._record_worker.finished_err.connect(self._on_record_error)

        w.btnRecordPcap.setEnabled(False)
        w.btnRun.setEnabled(False)
        w.btnStop.setEnabled(True)
        self._record_worker.start()

    def _on_record_done(self, path):
        self.w.editPcap.setText(path)
        self.w.labelStatus.setText("PCAP enregistre")
        self._log(f"\nPCAP pret pour calibration : {path}")
        self._record_cleanup()

    def _on_record_error(self, msg):
        self.w.labelStatus.setText(f"Erreur : {msg}")
        self._log(f"\nERREUR: {msg}")
        self._record_cleanup()

    def _record_cleanup(self):
        self.w.btnRecordPcap.setEnabled(True)
        self.w.btnRun.setEnabled(True)
        self.w.btnStop.setEnabled(False)
        self._record_worker = None

    def _browse_pcap(self):
        path, _ = QFileDialog.getOpenFileName(
            self.w, "Fichier PCAP",
            self.w.editPcap.text(),
            "PCAP (*.pcap);;Tous (*)")
        if path:
            self.w.editPcap.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self.w, "Dossier de sortie", self.w.editOutputDir.text())
        if path:
            self.w.editOutputDir.setText(path)

    # ── Method selection ──

    def _selected_method(self):
        w = self.w
        if w.radioV1.isChecked():
            return "v1"
        if w.radioV2.isChecked():
            return "v2"
        if w.radioV3.isChecked():
            return "v3"
        if w.radioV4.isChecked():
            return "v4"
        if w.radioV5.isChecked():
            return "v5"
        return "v1"

    # ── Build command line ──

    def _read_params(self):
        """Read all UI fields into a dict."""
        w = self.w
        return {
            "pcap": w.editPcap.text().strip(),
            "total_angle": w.spinTotalAngle.value(),
            "min_dist": w.spinMinDist.value(),
            "ip": w.editIP.text().strip(),
            "subsample": w.spinSubsample.value(),
            "max_shift": w.spinMaxShift.value() / 1000.0,  # mm -> m
            "output_dir": w.editOutputDir.text().strip(),
            "init_roll": w.spinInitRoll.value(),
            "init_pitch": w.spinInitPitch.value(),
            "init_yaw": w.spinInitYaw.value(),
            "init_tx": w.spinInitTx.value() / 1000.0,  # mm -> m
            "init_ty": w.spinInitTy.value() / 1000.0,
            "init_tz": w.spinInitTz.value() / 1000.0,
        }

    def _build_args(self, method, params):
        """Build command-line arguments for a given method."""
        script = os.path.join(SCRIPTS_DIR, METHODS[method])
        # Resolve paths relative to 01_test
        pcap = params["pcap"]
        if not os.path.isabs(pcap):
            pcap = os.path.join(SCRIPTS_DIR, pcap)
        output_dir = params["output_dir"]
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(SCRIPTS_DIR, output_dir)

        args = [
            sys.executable, "-u", script,
            "--pcap", pcap,
            "--output-dir", output_dir,
            "--total-angle", str(params["total_angle"]),
            "--min-dist", str(params["min_dist"]),
            "--ip", "0.0.0.0",
            "--subsample", str(params["subsample"]),
            "--max-shift", str(params["max_shift"]),
            "--init-angles",
            str(params["init_roll"]),
            str(params["init_pitch"]),
            str(params["init_yaw"]),
            "--init-translation",
            str(params["init_tx"]),
            str(params["init_ty"]),
            str(params["init_tz"]),
        ]
        return args

    # ── Run ──

    def _run_method(self, method, params):
        """Start a single optimization method."""
        args = self._build_args(method, params)

        self.w.labelStatus.setText(f"Calibration {method} en cours...")
        self._log(f"--- {METHODS[method]} ---")
        self._log(f"Commande: {' '.join(args[-8:])}")  # show tail

        self._process = QProcess(self.w)
        # Working dir = project root so scripts can import record_lidar, replay_scan, etc.
        self._process.setWorkingDirectory(
            os.path.join(os.path.dirname(__file__), ".."))
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)

        self._process.start(args[0], args[1:])
        self.w.btnRun.setEnabled(False)
        self.w.btnStop.setEnabled(True)

    def _run(self):
        """Run selected method with current params."""
        self.w.textLog.clear()
        self.w.progressBar.setValue(0)
        self._clear_results()
        self._workflow_queue.clear()
        method = self._selected_method()
        params = self._read_params()
        self._run_method(method, params)

    def _stop(self):
        self._workflow_queue.clear()
        if self._record_worker and self._record_worker.isRunning():
            self._record_worker.request_stop()
            self._log("Arret enregistrement demande")
            return
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._log("Arret demande")

    # ── Workflows ──

    def _workflow_full(self):
        """v1 cold start -> v2 warm start."""
        self.w.textLog.clear()
        self.w.progressBar.setValue(0)
        self._clear_results()
        self.w.radioV1.setChecked(True)
        params = self._read_params()
        # Force cold start for v1
        params["init_roll"] = 0.0
        params["init_pitch"] = 0.0
        params["init_yaw"] = 0.0
        params["init_tx"] = 0.0
        params["init_ty"] = 0.0
        params["init_tz"] = 0.0
        self._workflow_queue = [("v2", None)]  # v2 will use v1 result
        self._log("=== Workflow complet: v1 + v2 ===")
        self._run_method("v1", params)

    def _workflow_quick(self):
        """v1 cold start -> v5 warm start."""
        self.w.textLog.clear()
        self.w.progressBar.setValue(0)
        self._clear_results()
        self.w.radioV1.setChecked(True)
        params = self._read_params()
        params["init_roll"] = 0.0
        params["init_pitch"] = 0.0
        params["init_yaw"] = 0.0
        params["init_tx"] = 0.0
        params["init_ty"] = 0.0
        params["init_tz"] = 0.0
        self._workflow_queue = [("v5", None)]
        self._log("=== Workflow rapide: v1 + v5 ===")
        self._run_method("v1", params)

    # ── QProcess slots ──

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode(
            errors="replace")
        for line in data.splitlines():
            self._log(line)
            self._parse_result(line)

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode(
            errors="replace")
        for line in data.splitlines():
            self._log(line)

    def _on_finished(self, exit_code, exit_status):
        self._process = None

        if self._workflow_queue:
            # Chain next method with results from previous
            next_method, _ = self._workflow_queue.pop(0)
            params = self._read_params()  # picks up updated init values from UI
            self._log(f"\n--- Enchainement {next_method} (warm start) ---")
            self.w.progressBar.setValue(50)
            self._run_method(next_method, params)
            return

        self.w.btnRun.setEnabled(True)
        self.w.btnStop.setEnabled(False)
        self.w.progressBar.setValue(100)
        if exit_code == 0:
            self.w.labelStatus.setText("Calibration terminee")
            self._log("\nCalibration terminee avec succes !")
        else:
            self.w.labelStatus.setText(f"Erreur (code {exit_code})")
            self._log(f"\nProcessus termine avec code {exit_code}")

    # ── Output parsing ──

    def _clear_results(self):
        w = self.w
        for lbl in (w.labelResRoll, w.labelResPitch, w.labelResYaw,
                     w.labelResTx, w.labelResTy, w.labelResTz,
                     w.labelResImprovement, w.labelResFile):
            lbl.setText("\u2014")

    def _parse_result(self, line):
        """Parse optimization output to update init fields and results panel."""
        w = self.w
        line_s = line.strip()

        # Roll/Pitch/Yaw
        if "Roll=" in line_s and "Pitch=" in line_s and "Yaw=" in line_s:
            try:
                parts = line_s.replace("\u00b0", "").split()
                for part in parts:
                    if part.startswith("Roll="):
                        val = float(part.split("=")[1])
                        w.spinInitRoll.setValue(val)
                        w.labelResRoll.setText(f"{val:+.4f}\u00b0")
                    elif part.startswith("Pitch="):
                        val = float(part.split("=")[1])
                        w.spinInitPitch.setValue(val)
                        w.labelResPitch.setText(f"{val:+.4f}\u00b0")
                    elif part.startswith("Yaw="):
                        val = float(part.split("=")[1])
                        w.spinInitYaw.setValue(val)
                        w.labelResYaw.setText(f"{val:+.4f}\u00b0")
            except (ValueError, IndexError):
                pass

        # tx/ty/tz
        if "tx=" in line_s and "ty=" in line_s and "tz=" in line_s:
            try:
                parts = line_s.replace("mm", "").split()
                for part in parts:
                    if part.startswith("tx="):
                        val = float(part.split("=")[1])
                        w.spinInitTx.setValue(val)
                        w.labelResTx.setText(f"{val:+.1f} mm")
                    elif part.startswith("ty="):
                        val = float(part.split("=")[1])
                        w.spinInitTy.setValue(val)
                        w.labelResTy.setText(f"{val:+.1f} mm")
                    elif part.startswith("tz="):
                        val = float(part.split("=")[1])
                        w.spinInitTz.setValue(val)
                        w.labelResTz.setText(f"{val:+.1f} mm")
            except (ValueError, IndexError):
                pass

        # Improvement percentage (e.g. "Improvement: 81.9%" or "improvement 81.9%")
        if "mprovement" in line_s and "%" in line_s:
            try:
                idx = line_s.index("%")
                # Walk backwards to find the number
                num_start = idx - 1
                while num_start > 0 and (line_s[num_start - 1].isdigit()
                                          or line_s[num_start - 1] in ".+-"):
                    num_start -= 1
                val = line_s[num_start:idx].strip()
                w.labelResImprovement.setText(f"{val}%")
            except (ValueError, IndexError):
                pass

        # Output file (e.g. "Wrote scans/opt_front_xxx.pcd" or "Wrote scans/opt_xxx.pcd")
        if line_s.startswith("Wrote ") and ".pcd" in line_s:
            try:
                path = line_s.split("Wrote ")[1].split(" ")[0]
                w.labelResFile.setText(os.path.basename(path))
            except (ValueError, IndexError):
                pass

    def _log(self, msg):
        self.w.textLog.appendPlainText(msg)

    def show(self):
        self.w.show()
