#!/bin/bash
# ============================================================================
# Setup cron job for the bridge data pipeline
# ============================================================================
# Usage: bash setup_cron.sh [interval]
#   interval: "6h" (default), "12h", "daily"
#
# This adds a cron job that runs data_pipeline.py on the chosen schedule.
# Logs go to pipeline_logs/ directory.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
PIPELINE="${SCRIPT_DIR}/data_pipeline.py"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $PYTHON"
    echo "Create it first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

if [ ! -f "$PIPELINE" ]; then
    echo "ERROR: data_pipeline.py not found at $PIPELINE"
    exit 1
fi

INTERVAL="${1:-6h}"

case "$INTERVAL" in
    6h)
        CRON_EXPR="0 */6 * * *"
        ;;
    12h)
        CRON_EXPR="0 */12 * * *"
        ;;
    daily)
        CRON_EXPR="0 3 * * *"
        ;;
    *)
        echo "Unknown interval: $INTERVAL (use: 6h, 12h, daily)"
        exit 1
        ;;
esac

CRON_CMD="$CRON_EXPR cd $SCRIPT_DIR && $PYTHON $PIPELINE >> ${SCRIPT_DIR}/pipeline_logs/cron.log 2>&1"

# Check if already installed
if crontab -l 2>/dev/null | grep -q "data_pipeline.py"; then
    echo "Existing pipeline cron job found. Replacing..."
    crontab -l 2>/dev/null | grep -v "data_pipeline.py" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo "Cron job installed successfully!"
echo "  Schedule: $CRON_EXPR ($INTERVAL)"
echo "  Command:  cd $SCRIPT_DIR && $PYTHON $PIPELINE"
echo "  Logs:     ${SCRIPT_DIR}/pipeline_logs/cron.log"
echo ""
echo "To verify:  crontab -l"
echo "To remove:  crontab -l | grep -v data_pipeline | crontab -"
