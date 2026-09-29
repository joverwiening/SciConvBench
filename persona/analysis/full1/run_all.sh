#!/bin/bash
# Usage (from the SciConvBench repo root): run_all.sh RUN_DIR ANALYSIS_DIR OUT [REPORT_OUT]; env PYTHON, DATASET override defaults.
set -euo pipefail
RUN=$1; ANALYSIS=$2; OUT=$3; REPORT=${4:-}
PYTHON=${PYTHON:-python}
DATASET=${DATASET:-dataset/foam/disambiguation_foam.json}
HERE=$(cd "$(dirname "$0")" && pwd)
ARGS=(--run-dir "$RUN" --analysis-dir "$ANALYSIS" --dataset "$DATASET" --out "$OUT")
mkdir -p "$OUT"
"$PYTHON" "$HERE/content.py" "${ARGS[@]}"
"$PYTHON" "$HERE/decode_stages.py" "${ARGS[@]}"
"$PYTHON" "$HERE/figs_mechanism.py" "${ARGS[@]}"
"$PYTHON" "$HERE/figs_routes.py" "${ARGS[@]}"
"$PYTHON" "$HERE/plot_poles.py" "${ARGS[@]}"
"$PYTHON" "$HERE/lencheck.py" "${ARGS[@]}" | tee "$OUT/lencheck.txt"
if [ -n "$REPORT" ]; then
  "$PYTHON" "$HERE/export_report.py" --run-dir "$RUN" --analysis-dir "$ANALYSIS" --deep-dir "$OUT" --out "$REPORT"
fi
