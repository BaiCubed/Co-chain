#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/opt/pangu/openPangu-Embedded-1B-V1.1}"
LOG_FILE="${2:-run_pangu_full.log}"

python run_pangu.py --model_dir "${MODEL_DIR}" 2>&1 | tee "${LOG_FILE}"
echo "[OK] log saved to: ${LOG_FILE}"
