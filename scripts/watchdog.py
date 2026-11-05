"""Restart the forward-pass server if its process died (e.g. out-of-memory kill). The interrupted job is requeued;
a job that was running during 2 crashes is moved to failed/ (not charged) so a crashing request cannot loop.
Run detached:  setsid nohup /home/user/venv/bin/python scripts/watchdog.py >> queue/watchdog.log 2>&1 < /dev/null &"""
import json, os, subprocess, time
ROOT = "/home/user/weight-space-experiments"; Q = os.path.join(ROOT, "queue")
def alive():
    try:
        os.kill(int(open(os.path.join(Q, "server.pid")).read().strip()), 0); return True
    except Exception:
        return False
while True:
    if not alive():
        for fn in os.listdir(os.path.join(Q, "running")):
            p = os.path.join(Q, "running", fn); j = json.load(open(p))
            j["crashes"] = j.get("crashes", 0) + 1
            if j["crashes"] >= 2:
                j["error"] = ("the server crashed twice while running this request (most likely out of memory: request "
                              "fewer or shorter texts, fewer layers, or 'last' positions); not charged")
                json.dump(j, open(os.path.join(Q, "failed", fn), "w")); os.remove(p)
            else:
                json.dump(j, open(p, "w"))
        print(time.strftime("%H:%M:%S"), "server down: restarting", flush=True)
        subprocess.run(["bash", os.path.join(ROOT, "start_server.sh")], cwd=ROOT)
        time.sleep(120)
    time.sleep(30)
