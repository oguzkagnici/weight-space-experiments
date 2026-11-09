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


def gold_rank(r):
    """rank of the correct option among the question's options (0 = top), from the stored letter probabilities"""
    p = r["probs"]; return int(sum(x > p[r["answer"]] for x in p))


def entropy(r):
    p = np.array(r["probs"]); p = p[p > 0]; return float(-(p * np.log(p)).sum())


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
            "mean_gold_rank": round(float(np.mean([gold_rank(r) for r in rows])), 4),
            "mean_entropy": round(float(np.mean([entropy(r) for r in rows])), 4),
            "by_category": {c: [round(100 * np.mean(v), 1), len(v)] for c, v in sorted(cats.items())},
            "pred_letter_counts": dict(zip("ABCDEFGHIJ", map(int, dist))),
            "answer_letter_counts": dict(zip("ABCDEFGHIJ", map(int, adist)))}


def paired(ro, re_):
    o = {r["id"]: r for r in ro}
    pairs = [(o[r["id"]], r) for r in re_ if r["id"] in o]
    n = len(pairs)
    d = np.array([b["correct"] - a["correct"] for a, b in pairs], float)
    dl = np.array([b["logp_correct"] - a["logp_correct"] for a, b in pairs], float)
    dr = np.array([gold_rank(b) - gold_rank(a) for a, b in pairs], float)
    return {"n": n, "acc_diff": round(100 * d.mean(), 2), "acc_diff_se": round(100 * d.std(ddof=1) / math.sqrt(n), 2) if n > 1 else None,
            "wrong_to_right": int((d > 0).sum()), "right_to_wrong": int((d < 0).sum()),
            "pred_changed": int(sum(a["pred"] != b["pred"] for a, b in pairs)),
            "logp_correct_diff": round(float(dl.mean()), 4),
            "logp_correct_diff_se": round(float(dl.std(ddof=1) / math.sqrt(n)), 4) if n > 1 else None,
            "gold_rank_diff": round(float(dr.mean()), 4),
            "gold_rank_diff_se": round(float(dr.std(ddof=1) / math.sqrt(n)), 4) if n > 1 else None}


class Server:
    def __init__(self):
        self.e = Engine()
        self.llama = None          # thinking-mode generation backend (llama-server), started on demand
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

    def _gen(self, dirs, items, cap=None):
        from . import gen
        if self.llama is None:
            self.llama = gen.LlamaServer()
        key = weights_key(dirs)
        self.beat(f"building gguf {key}" if dirs else "base gguf")
        path = gen.build_gguf(dirs, key)
        self.beat(f"generating {len(items)} answers with {key or 'base'}")
        cap = cap or gen.THINK_CAP
        return gen.generate(self.llama, path, items, cache_path=os.path.join(CACHE, f"gen_{key}_cap{cap}" + (f"_{gen.ENGINE_TAG}" if gen.ENGINE_TAG else "") + ".jsonl"), cap=cap), key

    def run_generate(self, job):
        from . import gen
        models = ["original", "edited"] if job["model"] == "both" else [job["model"]]
        out_dir = job["out_dir"]; os.makedirs(out_dir, exist_ok=True)
        items = [(q, s) for q in job["questions"] for s in range(job.get("n", 1))]
        res, rows = {}, {}
        for m in models:
            t = time.time()
            rows[m], key = self._gen(self.weights(job, m), items)
            with open(os.path.join(out_dir, f"gen_{job['k']}_{m}.jsonl"), "w") as f:
                for r in rows[m]: f.write(json.dumps(r) + "\n")
            res[m] = gen.summarize_gen(rows[m]); res[m]["seconds"] = round(time.time() - t, 1)
        if len(models) == 2:
            res["paired_edited_minus_original"] = gen.paired_gen(rows["original"], rows["edited"])
        return res

    def run_eval_gen(self, job):
        from . import gen
        t = time.time()
        items = [(q, s) for q in job["questions"] for s in range(job.get("n", 1))]
        rows, key = self._gen(job["weights"], items, job.get("cap"))
        os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        with open(job["out"], "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        s = gen.summarize_gen(rows); s["seconds"] = round(time.time() - t, 1); s["weights_key"] = key
        return s

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
                if self.llama is not None:
                    self.llama.stop()     # D52: execv keeps child processes; an orphaned llama-server kept the port
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
                if job["type"] in ("score", "read", "eval") and self.llama is not None:
                    # free llama-server's memory (about 5 GB) before a PyTorch pass; it restarts on the next
                    # generation job (about 10 s). Two models plus long-text activations exceeded the RAM once.
                    self.llama.stop()
                if job["type"] in ("generate", "eval_gen"):
                    # the fp32 PyTorch model (6.8 GB) is not needed while llama-server generates; reloaded on demand
                    self.e.unload()
                if job["type"] == "score":
                    result = self.run_score(job)
                elif job["type"] == "read":
                    result = self.run_read(job)
                elif job["type"] == "generate":
                    result = self.run_generate(job)
                elif job["type"] == "eval_gen":
                    result = self.run_eval_gen(job)
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
