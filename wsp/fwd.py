"""Forward-pass client (the editor's ```forward block).

  fwd request.json         validate, enqueue, wait up to ~9 min, print the summary
  fwd wait <job_id>        keep waiting for a job
  fwd status               queue state, your budget

Request JSON (one pass each, whatever the size; failed requests do not count):
 {"type": "score", "model": "original" | "edited" | "both",
  "questions": ["<practice id>", ...]  or  "sample": {"n": 256, "seed": 0, "categories": [...] (optional)},
  "activations": {"hidden_states": [0..24], "mlp": [0..23], "inputs": [module names], "outputs": [module names]}}
     at most 512 questions per model (256 with "both"). Activations are taken at the answer position (last token).
 {"type": "read", "model": "original" | "edited",
  "texts": ["raw text", ...]  or  "questions": ["<practice id>", ...] (rendered evaluation strings),
  "suffixes": ["text appended to each rendered question", ...] (optional, with "questions"),
  "return": {"token_logprobs": true, "topk": k<=50, "hidden_states": [...], "mlp": [...], "inputs": [...],
             "outputs": [...], "positions": "last" | "mean" | "all", "attn_layers": [attention layer idx]}}
     at most 1,024 texts, 200,000 tokens in total, 4,096 per text; attention probabilities for the first 4 texts."""
import json, os, random, shutil, sys, time, uuid

from .config import QUEUE, MAX_SCORE_ITEMS, MAX_SCORE_ITEMS_BOTH, MAX_READ_TEXTS, MAX_READ_TOKENS, MAX_TEXT_TOKENS
from .server import locked_update
from . import data

WAIT = int(os.environ.get("WSP_WAIT", "540"))


def fail(msg):
    print(f"REQUEST REFUSED (does not count): {msg}"); sys.exit(1)


def editor_ctx():
    edir = os.environ.get("WSP_EDITOR_DIR")
    if not edir:
        fail("WSP_EDITOR_DIR not set")
    return edir, json.load(open(os.path.join(edir, "meta.json"))), os.path.join(edir, "state.json")


def submit(req):
    edir, meta, sfile = editor_ctx()
    st = json.load(open(sfile)) if os.path.exists(sfile) else {}
    used, pend = st.get("passes_used", 0), st.get("pending", [])
    if used + len(pend) >= meta["budget"]:
        fail(f"budget exhausted: {used} used + {len(pend)} pending of {meta['budget']}")
    practice = data.splits()["practice"]; pset = set(practice)
    Q = data.questions()
    t = req.get("type")
    if t not in ("score", "read"):
        fail("type must be 'score' or 'read'")
    model = req.get("model")
    k = used + len(pend) + 1
    jid = f"{meta['gen_name']}_{meta['editor']}_p{k:02d}_{uuid.uuid4().hex[:6]}"
    job = {"id": jid, "k": k, "type": t, "model": model, "priority": 5, "created": time.time(),
           "editor": meta["editor"], "gen": meta["gen"], "parent_weights": meta.get("parent_weights"),
           "state_file": sfile, "out_dir": os.path.join(edir, "work", f"fwd_{k:02d}"), "request": req}
    if t == "score":
        if model not in ("original", "edited", "both"):
            fail("model must be original, edited or both")
        if "questions" in req:
            qs = list(req["questions"])
        elif "sample" in req:
            s = req["sample"]; pool = [q for q in practice if not s.get("categories") or Q[q]["category"] in s["categories"]]
            qs = random.Random(s.get("seed", 0)).sample(pool, min(int(s["n"]), len(pool)))
        else:
            fail("give 'questions' or 'sample'")
        bad = [q for q in qs if q not in pset]
        if bad:
            fail(f"{len(bad)} ids are not practice questions, e.g. {bad[:3]}")
        lim = MAX_SCORE_ITEMS_BOTH if model == "both" else MAX_SCORE_ITEMS
        if not qs or len(qs) > lim:
            fail(f"1..{lim} questions per model")
        job["questions"] = qs; job["activations"] = req.get("activations")
    else:
        if model not in ("original", "edited"):
            fail("model must be original or edited")
        if "texts" in req:
            texts = list(req["texts"])
        elif "questions" in req:
            bad = [q for q in req["questions"] if q not in pset]
            if bad:
                fail(f"{len(bad)} ids are not practice questions")
            from transformers import AutoTokenizer
            from .config import MODEL_DIR
            tk = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
            suf = req.get("suffixes") or [""] * len(req["questions"])
            texts = [data.render(Q[q], tk) + s for q, s in zip(req["questions"], suf)]
        else:
            fail("give 'texts' or 'questions'")
        from transformers import AutoTokenizer
        from .config import MODEL_DIR
        tk = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        lens = [len(tk(x, add_special_tokens=False).input_ids) for x in texts]
        if not texts or len(texts) > MAX_READ_TEXTS or sum(lens) > MAX_READ_TOKENS or max(lens) > MAX_TEXT_TOKENS:
            fail(f"read limits: <= {MAX_READ_TEXTS} texts, <= {MAX_READ_TOKENS} tokens total, <= {MAX_TEXT_TOKENS} per text "
                 f"(got {len(texts)} texts, {sum(lens)} tokens, max {max(lens) if lens else 0})")
        job["texts_resolved"] = texts; job["return"] = req.get("return", {})
    if model in ("edited", "both"):
        src = os.path.join(edir, "edited")
        if not os.listdir(src):
            fail("no edits staged in edited/")
        snap = os.path.join(edir, "work", f"fwd_{k:02d}_snapshot")
        if os.path.exists(snap):
            shutil.rmtree(snap)
        shutil.copytree(src, snap)
        job["snapshot"] = snap
    locked_update(sfile, lambda s: s.setdefault("pending", []).append(jid))
    path = os.path.join(QUEUE, "pending", jid + ".json")
    json.dump(job, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    with open(os.path.join(edir, "requests.jsonl"), "a") as f:
        f.write(json.dumps({"id": jid, "k": k, "time": time.time(), "request": req}) + "\n")
    print(f"queued {jid} (pass {k} of {meta['budget']})")
    return jid


def show(job):
    if "error" in job:
        print(f"FAILED (does not count): {job['error']}"); return
    print(f"DONE {job['id']} pass {job['k']} in {job['seconds']}s; files in {job['out_dir']}")
    out = []
    for k, v in job["result"].items():
        if isinstance(v, dict):
            out.append(f"{k}:")
            out += [f"  {kk}: {json.dumps(vv)}" for kk, vv in v.items()]
        else:
            out.append(f"{k}: {json.dumps(v)}")
    print("\n".join(out)[:9000])


def wait(jid, timeout=WAIT):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for d in ("done", "failed"):
            p = os.path.join(QUEUE, d, jid + ".json")
            if os.path.exists(p):
                show(json.load(open(p))); return True
        time.sleep(3)
    where = "running" if os.path.exists(os.path.join(QUEUE, "running", jid + ".json")) else "queued"
    ahead = len(os.listdir(os.path.join(QUEUE, "pending")))
    print(f"STILL {where.upper()} after {timeout}s ({ahead} jobs pending in queue). Run: fwd wait {jid}")
    return False


def status():
    hb = json.load(open(os.path.join(QUEUE, "heartbeat.json"))) if os.path.exists(os.path.join(QUEUE, "heartbeat.json")) else {}
    age = time.time() - hb.get("time", 0)
    print(f"server: {hb.get('status')} (heartbeat {age:.0f}s ago){'  <-- SERVER DOWN?' if age > 120 else ''}")
    for d in ("pending", "running"):
        print(d, sorted(os.listdir(os.path.join(QUEUE, d))))
    if os.environ.get("WSP_EDITOR_DIR"):
        edir, meta, sfile = editor_ctx()
        st = json.load(open(sfile)) if os.path.exists(sfile) else {}
        print(f"budget: {st.get('passes_used', 0)} used, {len(st.get('pending', []))} pending, of {meta['budget']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "status":
        status()
    elif a[0] == "wait":
        wait(a[1])
    else:
        req = json.load(open(a[0])) if a[0] != "-" else json.load(sys.stdin)
        jid = submit(req)
        wait(jid)
