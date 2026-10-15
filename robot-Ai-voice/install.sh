#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install --user --break-system-packages -r requirements.txt

# openWakeWord 0.6.0 declares tflite-runtime even when only its ONNX backend is
# used. tflite-runtime has no Python 3.12 Linux wheel, so install openWakeWord
# without that optional backend. All dependencies needed by the ONNX path are
# explicitly listed in requirements.txt.
"${PYTHON_BIN}" -m pip install --user --break-system-packages \
  --no-deps openwakeword==0.6.0

"${PYTHON_BIN}" -c \
  "import numpy, onnxruntime, openwakeword, requests, scipy, sklearn, sounddevice; print('Python dependencies installed successfully')"
