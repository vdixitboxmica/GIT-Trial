#!/usr/bin/env bash
set -euxo pipefail

LOGFILE="/workspaces/start.log"

if ! pgrep -f "uvicorn server:app" >/dev/null 2>&1; then
  nohup python start.py --host 0.0.0.0 --port 8000 > "$LOGFILE" 2>&1 &
  sleep 2 || true
  tail -n 50 "$LOGFILE" || true
fi
