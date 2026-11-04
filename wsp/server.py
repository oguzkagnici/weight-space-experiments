"""Forward-pass server: one process, model in memory, processes queue/pending/*.json in priority order.

Job types
  score  (editor pass)  model original|edited|both on practice questions; optional activations at the answer position
  read   (editor pass)  teacher-forced read of texts / rendered practice questions
  eval   (GA internal)  score a weight state on hidden questions (selection / heldout); no budget
Budget accounting: a successful editor job increments passes_used in the editor's state.json; failures do not count."""
import fcntl, json, math, os, shutil, sys, time, traceback
import numpy as np

from .config import QUEUE, CACHE
from .engine import Engine, weights_key
from . import data

HEART = os.path.join(QUEUE, "heartbeat.json")


def locked_update(path, fn):
    lock = path + ".lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        st = json.load(open(path)) if os.path.exists(path) else {}
        st = fn(st) or st
        tmp = path + ".tmp"; json.dump(st, open(tmp, "w"), indent=1); os.replace(tmp, path)
        return st


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    acc = np.mean([r["correct"] for r in rows])
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r["correct"])
    dist = np.bincount([r["pred"] for r in rows], minlength=10)
    adist = np.bincount([r["answer"] for r in rows], minlength=10)
    return {"n": n, "acc": round(100 * acc, 2), "se": round(100 * math.sqrt(acc * (1 - acc) / n), 2),
            "mean_logp_correct": round(float(np.mean([r["logp_correct"] for r in rows])), 4),
            "mean_margin": round(float(np.mean([r["margin"] for r in rows])), 3),
            "mean_letter_mass": round(float(np.mean([r["letter_mass"] for r in rows])), 4),
            "by_category": {c: [round(100 * np.mean(v), 1), len(v)] for c, v in sorted(cats.items())},
            "pred_letter_counts": dict(zip("ABCDEFGHIJ", map(int, dist))),
            "answer_letter_counts": dict(zip("ABCDEFGHIJ", map(int, adist)))}


def paired(ro, re_):
    o = {r["id"]: r for r in ro}
    pairs = [(o[r["id"]], r) for r in re_ if r["id"] in o]
    n = len(pairs)
    d = np.array([b["correct"] - a["correct"] for a, b in pairs], float)
    dl = np.array([b["logp_correct"] - a["logp_correct"] for a, b in pairs], float)
    return {"n": n, "acc_diff": round(100 * d.mean(), 2), "acc_diff_se": round(100 * d.std(ddof=1) / math.sqrt(n), 2) if n > 1 else None,
            "wrong_to_right": int((d > 0).sum()), "right_to_wrong": int((d < 0).sum()),
            "pred_changed": int(sum(a["pred"] != b["pred"] for a, b in pairs)),
            "logp_correct_diff": round(float(dl.mean()), 4),
            "logp_correct_diff_se": round(float(dl.std(ddof=1) / math.sqrt(n)), 4) if n > 1 else None}


class Server:
    def __init__(self):
        self.e = Engine()
        for d in ("pending", "running", "done", "failed"):
            os.makedirs(os.path.join(QUEUE, d), exist_ok=True)
        os.makedirs(CACHE, exist_ok=True)

    def beat(self, status):
        json.dump({"time": time.time(), "pid": os.getpid(), "status": status}, open(HEART + ".tmp", "w"))
        os.replace(HEART + ".tmp", HEART)

    def weights(self, job, which):
        if which == "original":
            return [d for d in [job.get("parent_weights")] if d]
        return [d for d in [job.get("parent_weights"), job["snapshot"]] if d]

    def run_score(self, job):
        models = ["original", "edited"] if job["model"] == "both" else [job["model"]]
        acts = job.get("activations")
        out_dir = job["out_dir"]; os.makedirs(out_dir, exist_ok=True)
        res, rows = {}, {}
        for m in models:
            dirs = self.weights(job, m)
            key = self.e.set_weights(dirs)
            t = time.time()
            rows[m], A = self.e.score(job["questions"], acts=acts, cache_key=None if acts else key)
            with open(os.path.join(out_dir, f"score_{job['k']}_{m}.jsonl"), "w") as f:
                for r in rows[m]: f.write(json.dumps(r) + "\n")
            res[m] = summarize(rows[m]); res[m]["seconds"] = round(time.time() - t, 1)
            if A:
                ad = os.path.join(out_dir, f"score_{job['k']}_acts", m); os.makedirs(ad, exist_ok=True)
                idx = {"ids": job["questions"], "correct": [r["correct"] for r in rows[m]], "files": {}}
                for k, v in A.items():
                    fn = k.replace(":", "_") + ".npy"; np.save(os.path.join(ad, fn), v); idx["files"][k] = [fn, list(v.shape)]
                json.dump(idx, open(os.path.join(ad, "index.json"), "w"))
        if len(models) == 2:
            res["paired_edited_minus_original"] = paired(rows["original"], rows["edited"])
        return res

    def run_read(self, job):
        dirs = self.weights(job, job["model"])
        self.e.set_weights(dirs)
        texts = job["texts_resolved"]
        return self.e.read(texts, job.get("return", {}), job["out_dir"])

    def run_eval(self, job):
        key = self.e.set_weights(job["weights"])
        t = time.time()
        rows, _ = self.e.score(job["questions"], cache_key=key)
        os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        with open(job["out"], "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        s = summarize(rows); s["seconds"] = round(time.time() - t, 1); s["weights_key"] = key
        return s

    def next_job(self):
        pend = os.path.join(QUEUE, "pending")
        jobs = []
        for fn in os.listdir(pend):
            if fn.endswith(".json"):
                try:
                    j = json.load(open(os.path.join(pend, fn)))
                    jobs.append((j.get("priority", 5), j.get("created", 0), fn))
                except Exception:
                    pass
        return sorted(jobs)[0][2] if jobs else None

    def loop(self):
        while True:
            flag = os.path.join(QUEUE, "RESTART")
            if os.path.exists(flag):          # graceful restart between jobs (picks up code changes)
                os.remove(flag); self.beat("restarting")
                os.execv(sys.executable, [sys.executable, "-m", "wsp.server"])
            fn = self.next_job()
            if not fn:
                self.beat("idle"); time.sleep(2); continue
            src = os.path.join(QUEUE, "pending", fn); run = os.path.join(QUEUE, "running", fn)
            try:
                os.replace(src, run)
            except FileNotFoundError:
                continue
            job = json.load(open(run))
            self.beat(f"running {job['id']} ({job['type']})")
            t = time.time()
            try:
                if job["type"] == "score":
                    result = self.run_score(job)
                elif job["type"] == "read":
                    result = self.run_read(job)
                elif job["type"] == "eval":
                    result = self.run_eval(job)
                else:
                    raise ValueError(f"unknown job type {job['type']}")
                job["result"] = result; job["seconds"] = round(time.time() - t, 1); job["finished"] = time.time()
                if job.get("state_file"):
                    def inc(st, job=job):
                        st["passes_used"] = st.get("passes_used", 0) + 1
                        st.setdefault("passes", []).append({"k": job["k"], "id": job["id"], "type": job["type"],
                                                            "seconds": job["seconds"]})
                        st["pending"] = [p for p in st.get("pending", []) if p != job["id"]]
                    locked_update(job["state_file"], inc)
                    with open(os.path.join(os.path.dirname(job["state_file"]), "passes.jsonl"), "a") as f:
                        f.write(json.dumps({k: job.get(k) for k in ("id", "k", "type", "model", "request", "result",
                                                                     "seconds", "created", "finished")}) + "\n")
                dst = os.path.join(QUEUE, "done", fn)
            except Exception as ex:
                job["error"] = f"{type(ex).__name__}: {ex}"; job["traceback"] = traceback.format_exc()[-3000:]
                if job.get("state_file"):
                    locked_update(job["state_file"], lambda st, job=job: st.update(
                        pending=[p for p in st.get("pending", []) if p != job["id"]]))
                dst = os.path.join(QUEUE, "failed", fn)
            json.dump(job, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
            os.remove(run)


if __name__ == "__main__":
    Server().loop()
