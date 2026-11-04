#!/bin/bash
# (re)start the forward-pass server detached; requeue any job that was running
cd /home/user/weight-space-experiments
pkill -f "wsp.server" 2>/dev/null; sleep 1
for f in queue/running/*.json; do [ -e "$f" ] && mv "$f" queue/pending/; done
PYTHONPATH=/home/user/weight-space-experiments setsid nohup /home/user/venv/bin/python -m wsp.server >> queue/server.log 2>&1 < /dev/null &
echo "server started pid $!"
