#!/usr/bin/env bash
set -euxo pipefail

# Git LFS is already installed in the image; set it up without touching repo hooks
git lfs install --skip-repo || true

# Let start.py handle venv + requirements + server
LOGFILE="/workspaces/start.log"
nohup python start.py --host 0.0.0.0 --port 8000 --upgrade > "$LOGFILE" 2>&1 &

# small wait to show early logs in creation output
sleep 3 || true
tail -n +1 "$LOGFILE" || true
