#!/bin/bash
# Run all benchmark conditions sequentially, then generate the summary.
# Usage: bash simulation/experiments/run_all.bash [--overwrite]
#   --overwrite  Delete all previous output before starting fresh.
#                Without this flag, completed runs are resumed automatically.
# Run from the project root with the mogp-waveome-sim conda env active.

set -e  # abort on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/output"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Parse --overwrite flag
OVERWRITE=0
for arg in "$@"; do
    if [ "$arg" = "--overwrite" ]; then OVERWRITE=1; fi
done

if [ "$OVERWRITE" = "1" ]; then
    log "--overwrite set: removing previous output"
    rm -rf "$OUTPUT_DIR/results_easy" \
           "$OUTPUT_DIR/results_medium" \
           "$OUTPUT_DIR/results_difficult" \
           "$OUTPUT_DIR/results_annotation" \
           "$OUTPUT_DIR/results_group_covariate" \
           "$OUTPUT_DIR/summary"
    log "Previous output removed"
fi

log "Starting full benchmark run"
START=$(date +%s)

log "=== Easy condition ==="
bash "$SCRIPT_DIR/run_easy.bash"

log "=== Medium condition ==="
bash "$SCRIPT_DIR/run_medium.bash"

log "=== Difficult condition ==="
bash "$SCRIPT_DIR/run_difficult.bash"

log "=== Annotation sweep ==="
bash "$SCRIPT_DIR/run_annotation.bash"

log "=== Group covariate condition ==="
bash "$SCRIPT_DIR/run_group_covariate.bash"

log "=== Generating summary ==="
python3 "$SCRIPT_DIR/generate_summary.py"

END=$(date +%s)
ELAPSED=$(( END - START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
log "All done. Total wall time: ${HOURS}h ${MINS}m"
