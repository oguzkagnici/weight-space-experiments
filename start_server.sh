#!/bin/bash
# (re)start the forward-pass server detached; requeue any job that was running.
# For code changes prefer a graceful restart between jobs: touch queue/RESTART
cd /home/user/weight-space-experiments
if [ -f queue/server.pid ] && kill -0 "$(cat queue/server.pid)" 2>/dev/null; then kill "$(cat queue/server.pid)"; sleep 1; fi
for f in queue/running/*.json; do [ -e "$f" ] && mv "$f" queue/pending/; done
PYTHONPATH=/home/user/weight-space-experiments setsid nohup /home/user/venv/bin/python -m wsp.server >> queue/server.log 2>&1 < /dev/null &
echo $! > queue/server.pid
echo "server started pid $!"
