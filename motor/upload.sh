#!/bin/bash

# ESP32 Compile & Upload Script
# Usage: ./upload.sh <sketch_dir> [port] [baudrate]

if [ -z "$1" ]; then
    echo "Usage: ./upload.sh <sketch_dir> [port] [baudrate]"
    echo "Example: ./upload.sh motor_test /dev/ttyUSB0 460800"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKETCH_DIR="$SCRIPT_DIR/$1"
PORT=${2:-/dev/ttyUSB0}
BAUD=${3:-460800}
CLI="${ARDUINO_CLI:-arduino-cli}"
SKETCH_NAME=$(basename "$SKETCH_DIR")

echo "=== ESP32 Compiler & Uploader ==="
echo "Sketch: $SKETCH_NAME"
echo "Port: $PORT"
echo "Baud: $BAUD"
echo ""

echo "[1/2] Compiling sketch..."
$CLI compile --fqbn esp32:esp32:esp32 "$SKETCH_DIR"

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful"
echo ""

echo "[2/2] Uploading to ESP32..."
$CLI upload --fqbn esp32:esp32:esp32 --port $PORT "$SKETCH_DIR"

if [ $? -ne 0 ]; then
    echo "Upload failed!"
    exit 1
fi

echo "Upload successful!"
echo ""
echo "Monitoring serial output (Ctrl+C to stop):"
sleep 2
stty -F $PORT 115200 raw -echo
timeout 15 cat $PORT
