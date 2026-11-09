"""Thinking-mode generation backend (phase 2): edited weights -> GGUF (patched official BF16, quantised to Q8_0) ->
llama-server with continuous batching -> graded answers.

Protocol (fixed): the user message is the question, the options as "(A) ..." lines, and THINK_INSTR; the chat template
is applied with thinking enabled (the model's default); sampling temperature 0.6, top_p 0.95, top_k 20; at most
THINK_CAP new tokens for thinking and answer together. Graded: the last "answer is (X)" after the thinking. An answer
cut at the cap or without that phrase is wrong."""
import hashlib, json, os, re, subprocess, time, urllib.request, zlib, concurrent.futures as cf
import numpy as np

from .config import ROOT, LETTERS
from . import data

# Generation engine (D40). "det1": llama.cpp built with -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF, separate KV per slot,
# no flash attention, no prompt-prefix reuse -> byte-identical outputs regardless of batch neighbours, and 1.6-2x faster on
# this host than the default build. The engine tag is part of every generation cache file name, so results of different
# engines are never mixed or paired. Generations 1-4 used the default build (tag ""), which is NOT deterministic with 16 slots.
ENGINE_TAG = "det1"
LLAMA = "/home/user/llama.cpp/build-det/bin"
GGUF_CACHE = "/home/user/gguf_cache"
BASE_Q8 = "/home/user/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf"
PORT = 8091
PARALLEL = 16
THINK_CAP = 3072
THINK_INSTR = 'Think step by step, then finish with "The answer is (X)" where X is the letter of the correct option.'
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MAX_GGUF_CACHE = 3            # keep at most this many edited Q8_0 files
EXTRA_ARGS = ["-no-kvu", "-fa", "off", "-cram", "0"]   # extra llama-server arguments; -cram 0 (D44): no host prompt cache (OOM)
CACHE_PROMPT = False          # llama-server prompt-prefix reuse across requests


def user_message(q):
    opts = "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(q["options"]))
    return f"{q['question']}\n\n{opts}\n\n{THINK_INSTR}"


def render_prompt(q, tok):
    """The exact prompt string (chat template, thinking enabled), for teacher-forced analysis of traces."""
    return tok.apply_chat_template([{"role": "user", "content": user_message(q)}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


def seed_of(qid, sample=0):
    return zlib.crc32(f"{qid}:{sample}".encode()) % 2_000_000_000


def grade(content, gold_letter):
    m = re.findall(r"answer is \(?\*?\*?([A-J])\)?", content or "")
    pred = m[-1] if m else None
    return pred, int(pred == gold_letter)


# ---------------------------------------------------------------- weights -> gguf
def build_gguf(weight_dirs, key):
    """Q8_0 GGUF for a weight state (list of edit dirs, later overriding earlier); base -> official Q8_0."""
    if not weight_dirs:
        return BASE_Q8
    os.makedirs(GGUF_CACHE, exist_ok=True)
    out = os.path.join(GGUF_CACHE, f"{key}.q8_0.gguf")
    if os.path.exists(out):
        os.utime(out); return out
    from safetensors.torch import load_file
    from .gguf_patch import patch
    edits = {}
    for d in weight_dirs:
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors"):
                edits[fn[:-12]] = load_file(os.path.join(d, fn))["tensor"]
    # evict before building (a build needs ~5.3 GB of scratch: bf16 GGUF + Q8_0) so a full disk cannot fail it
    files = sorted((f for f in os.listdir(GGUF_CACHE) if f.endswith(".q8_0.gguf")),
                   key=lambda f: os.path.getmtime(os.path.join(GGUF_CACHE, f)))
    for f in files[:-(MAX_GGUF_CACHE - 1)]:
        os.remove(os.path.join(GGUF_CACHE, f))
    tmp = os.path.join(GGUF_CACHE, f"{key}.bf16.gguf")
    try:
        patch(edits, tmp)
        subprocess.run([f"{LLAMA}/llama-quantize", tmp, out + ".tmp", "Q8_0", "4"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(out + ".tmp", out)
    finally:   # never leave multi-GB scratch files behind
        for f in (tmp, out + ".tmp"):
            if os.path.exists(f):
                os.remove(f)
    return out


# ---------------------------------------------------------------- llama-server management
def kill_strays():
    """SIGKILL every llama-server on our port that this process does not track (D52), and wait until they are gone."""
    import signal
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            args = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if args and args[0].endswith(b"llama-server") and b"--port" in args and str(PORT).encode() in args:
            pids.append(int(d))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for _ in range(100):
        alive = []
        for pid in pids:
            try:
                st = open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]
            except OSError:
                continue
            if st != "Z":
                alive.append(pid)
            else:
                try:
                    os.waitpid(pid, os.WNOHANG)   # reap our own zombie children
                except ChildProcessError:
                    pass
        if not alive:
            return pids
        time.sleep(0.1)
    raise RuntimeError(f"could not kill stray llama-server(s) {pids}")


class LlamaServer:
    def __init__(self):
        self.proc = None; self.model = None; self.cap = None; self.room = None

    def ensure(self, gguf_path, cap=THINK_CAP, room=1024):
        if self.proc and self.proc.poll() is None and self.model == gguf_path and self.cap == cap and self.room == room:
            return
        self.stop()
        kill_strays()   # D52: a llama-server left over from a worker re-exec kept the port and answered for every model
        ctx = PARALLEL * (cap + room)   # per-slot context = cap + prompt room (D60: caps other than THINK_CAP; D62: room)
        self.cap = cap; self.room = room
        log = open(os.path.join(ROOT, "queue", "llama_server.log"), "a")
        self.proc = subprocess.Popen([f"{LLAMA}/llama-server", "-m", gguf_path, "-t", "4", "-np", str(PARALLEL),
                                      "-c", str(ctx), "--port", str(PORT), "--jinja", "--reasoning-format", "none"] + list(EXTRA_ARGS),
                                     stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        self.model = gguf_path
        for _ in range(300):
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during start-up (see queue/llama_server.log)")
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read():
                    time.sleep(1)   # a server that failed to bind exits within milliseconds; make sure ours is the one answering
                    if self.proc.poll() is not None:
                        raise RuntimeError("llama-server exited during start-up (port taken? see queue/llama_server.log)")
                    served = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5)).get("model_path")
                    if served != gguf_path:
                        raise RuntimeError(f"port {PORT} serves {served}, expected {gguf_path}")
                    return
            except RuntimeError:
                self.stop(); raise
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except Exception:
                self.proc.kill(); self.proc.wait(30)
        self.proc = None; self.model = None; self.cap = None; self.room = None

    def ask_bench(self, item, cap):
        """D62: one free-form benchmark item {id, messages[, tools]} -> raw thinking-mode output (graded offline)."""
        body = {"messages": item["messages"], "max_tokens": cap, "seed": seed_of(item["id"]),
                "cache_prompt": CACHE_PROMPT, **SAMPLING}
        if item.get("tools"):
            body["tools"] = item["tools"]
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=48 * 3600))
        if r.get("model") not in (None, self.model):
            raise RuntimeError(f"answer served by {r.get('model')}, expected {self.model}")
        c = r["choices"][0]; m = c["message"]; raw = m.get("content") or ""
        think, content = raw.split("</think>", 1) if "</think>" in raw else (raw, "")
        return {"id": item["id"], "cut": int(c["finish_reason"] == "length"), "finished_think": int("</think>" in raw),
                "tokens": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"].get("prompt_tokens"),
                "think": think, "content": content, "raw": raw, "tool_calls": m.get("tool_calls")}

    def ask(self, qid, sample=0, cap=THINK_CAP):
        q = data.questions()[qid]
        body = {"messages": [{"role": "user", "content": user_message(q)}], "max_tokens": cap,
                "seed": seed_of(qid, sample), "cache_prompt": CACHE_PROMPT, **SAMPLING}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=48 * 3600))
        if r.get("model") not in (None, self.model):   # D52: every answer must come from the requested weights
            raise RuntimeError(f"answer served by {r.get('model')}, expected {self.model}")
        c = r["choices"][0]; raw = c["message"].get("content") or ""
        # raw = exactly what the model generated after the prompt (which ends with "<think>")
        if "</think>" in raw:
            think, content = raw.split("</think>", 1)
        else:
            think, content = raw, ""
        pred, ok = grade(content, q["answer"])
        cut = c["finish_reason"] == "length"
        if cut:
            ok = 0   # protocol (D30, D49): an answer cut at the cap is wrong, even if "answer is (X)" was already written
        return {"id": qid, "sample": sample, "category": q["category"], "gold": q["answer"], "pred": pred,
                "correct": ok, "cut": int(cut), "finished_think": int("</think>" in raw),
                "tokens": r["usage"]["completion_tokens"], "think": think, "content": content, "raw": raw}


def generate(server, gguf_path, items, cache_path=None, cap=THINK_CAP):
    """items: list of (qid, sample). Returns rows (with full texts). Cached per weight state."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line); cache[(r["id"], r["sample"])] = r
    todo = [it for it in items if tuple(it) not in cache]
    if todo:
        server.ensure(gguf_path, cap)
        with cf.ThreadPoolExecutor(PARALLEL) as ex:
            futs = {ex.submit(server.ask, q, s, cap): (q, s) for q, s in todo}
            for f in cf.as_completed(futs):
                r = f.result(); cache[(r["id"], r["sample"])] = r
                if cache_path:
                    with open(cache_path, "a") as fh:
                        fh.write(json.dumps(r) + "\n")
    return [cache[tuple(it)] for it in items]


def generate_bench(server, gguf_path, items, cache_path, cap, room=1024):
    """D62: items are dicts {id, messages[, tools]}; rows cached per weight state, benchmark and cap."""
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line); cache[r["id"]] = r
    todo = [it for it in items if it["id"] not in cache]
    if todo:
        server.ensure(gguf_path, cap, room)
        with cf.ThreadPoolExecutor(PARALLEL) as ex:
            futs = [ex.submit(server.ask_bench, it, cap) for it in todo]
            for f in cf.as_completed(futs):
                r = f.result(); cache[r["id"]] = r
                with open(cache_path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
    return [cache[it["id"]] for it in items]


def summarize_gen(rows):
    n = len(rows)
    if not n:
        return {}
    acc = np.mean([r["correct"] for r in rows]); cut = np.mean([r["cut"] for r in rows])
    fin = [r for r in rows if not r["cut"]]
    return {"n": n, "acc": round(100 * acc, 2), "se": round(100 * float(np.sqrt(acc * (1 - acc) / n)), 2),
            "cut_share": round(100 * cut, 1), "acc_if_finished": round(100 * np.mean([r["correct"] for r in fin]), 1) if fin else None,
            "no_answer_finished": int(sum(1 for r in fin if r["pred"] is None)),
            "mean_tokens": round(float(np.mean([r["tokens"] for r in rows])), 1),
            "mean_tokens_finished": round(float(np.mean([r["tokens"] for r in fin])), 1) if fin else None}


def paired_gen(ro, re_):
    o = {(r["id"], r["sample"]): r for r in ro}
    pairs = [(o[(r["id"], r["sample"])], r) for r in re_ if (r["id"], r["sample"]) in o]
    n = len(pairs)
    d = np.array([b["correct"] - a["correct"] for a, b in pairs], float)
    dc = np.array([b["cut"] - a["cut"] for a, b in pairs], float)
    dt = np.array([b["tokens"] - a["tokens"] for a, b in pairs], float)
    se = lambda x: round(float(x.std(ddof=1) / np.sqrt(n)), 3) if n > 1 else None
    return {"n": n, "acc_diff": round(100 * d.mean(), 2), "acc_diff_se": round(100 * se(d), 2) if n > 1 else None,
            "wrong_to_right": int((d > 0).sum()), "right_to_wrong": int((d < 0).sum()),
            "cut_share_diff": round(100 * dc.mean(), 2), "cut_share_diff_se": round(100 * se(dc), 2) if n > 1 else None,
            "tokens_diff": round(float(dt.mean()), 1), "tokens_diff_se": se(dt)}
