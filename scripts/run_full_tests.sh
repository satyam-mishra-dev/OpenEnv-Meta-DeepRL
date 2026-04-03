#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT_DIR}/outputs"
REPORT_FILE="${REPORT_DIR}/test_report_full.txt"

mkdir -p "${REPORT_DIR}"

PYTHON_BIN="${ROOT_DIR}/../venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" -m pytest "${ROOT_DIR}/tests" -vv -rA --show-capture=all > "${REPORT_FILE}"

echo "Wrote test report to ${REPORT_FILE}"
