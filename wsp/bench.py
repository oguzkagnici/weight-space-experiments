"""Generalisation benchmarks beyond MMLU-Pro / GPQA (D62).

  python -m wsp.bench prep  <bench>                       build runs/bench/<bench>/items.jsonl (fixed subsample)
  python -m wsp.bench queue <bench> <base|champion|name> [--cap 8192] [--priority 2]
  python -m wsp.bench grade <bench> [--who base,champion]  score both runs, paired comparison

Generation is the thinking-mode protocol of D30/D40 (det1 llama.cpp, Q8_0, temperature 0.6 / top-p 0.95 / top-k 20, a
fixed seed per item shared by every model), run as queue job type "bench". Scoring is offline, with each benchmark's
own checker. A trace cut at the cap scores 0 (no answer), as everywhere else in this project.
"""
import argparse, json, math, os, random, re, sys, time

from .config import ROOT, QUEUE

BDIR = os.path.join(ROOT, "runs", "bench")
LET = "ABCDEFGHIJ"
MC_INSTR = 'Think step by step, then finish with "The answer is (X)" where X is the letter of the correct option.'
NUM_INSTR = 'Think step by step, then finish with "The answer is N" where N is the final number.'
BOX_INSTR = "Think step by step, then put your final answer within \\boxed{}."
IFEVAL_SRC = "/home/user/bench_src/gr"          # google-research checkout with instruction_following_eval/

# bench -> (description, sample size or None for all)
BENCHES = {
    "ifeval": ("IFEval, all 541 prompts; official Google checker, strict and loose", None),
    "gsm8k": ("GSM8K test, 250 sampled; last 'answer is N' (else last number) vs gold", 250),
    "medqa": ("MedQA-USMLE 4-option test, 256 sampled; same MC protocol as MMLU-Pro", 256),
    "math500": ("MATH-500, 200 sampled; last \\boxed{} vs gold with math-verify", 200),
}


def _token():
    p = "/home/user/.hf_token"
    return open(p).read().strip() if os.path.exists(p) else None


def _sample(rows, n, bench):
    idx = list(range(len(rows)))
    if n and n < len(rows):
        idx = sorted(random.Random(f"bench:{bench}").sample(idx, n))
    return [(i, rows[i]) for i in idx]


def prep(bench):
    from datasets import load_dataset
    tok = _token(); n = BENCHES[bench][1]; items = []
    if bench == "ifeval":
        d = load_dataset("google/IFEval", split="train", token=tok)
        for i, r in _sample(d, n, bench):
            items.append({"id": f"ifeval:{r['key']}", "messages": [{"role": "user", "content": r["prompt"]}],
                          "meta": {"key": r["key"], "instruction_id_list": r["instruction_id_list"],
                                   "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in r["kwargs"]]}})
    elif bench == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split="test", token=tok)
        for i, r in _sample(d, n, bench):
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"id": f"gsm8k:{i}", "messages": [{"role": "user", "content": f"{r['question']}\n\n{NUM_INSTR}"}],
                          "meta": {"gold": gold}})
    elif bench == "medqa":
        d = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", token=tok)
        for i, r in _sample(d, n, bench):
            opts = "\n".join(f"({k}) {v}" for k, v in sorted(r["options"].items()))
            items.append({"id": f"medqa:{i}", "messages": [{"role": "user", "content": f"{r['question']}\n\n{opts}\n\n{MC_INSTR}"}],
                          "meta": {"gold": r["answer_idx"], "step": r["meta_info"]}})
    elif bench == "math500":
        d = load_dataset("HuggingFaceH4/MATH-500", split="test", token=tok)
        for i, r in _sample(d, n, bench):
            items.append({"id": f"math500:{r['unique_id']}", "messages": [{"role": "user", "content": f"{r['problem']}\n\n{BOX_INSTR}"}],
                          "meta": {"gold": r["answer"], "subject": r["subject"], "level": r["level"]}})
    else:
        sys.exit(f"unknown bench {bench}")
    os.makedirs(os.path.join(BDIR, bench), exist_ok=True)
    p = os.path.join(BDIR, bench, "items.jsonl")
    with open(p, "w") as f:
        for it in items: f.write(json.dumps(it) + "\n")
    print(f"{bench}: {len(items)} items -> {p}")


def _who(who):
    if who == "champion":
        who = json.load(open(os.path.join(ROOT, "runs", "ga_state.json")))["champion"]
    w = [] if who == "base" else [os.path.join(ROOT, "runs", "champions", who, "weights")]
    return who, w


def out_path(bench, who, cap):
    return os.path.join(BDIR, bench, f"{who}_cap{cap}.jsonl")


def queue(bench, who, cap=8192, priority=2, room=1024):
    who, weights = _who(who)
    items_file = os.path.join(BDIR, bench, "items.jsonl")
    if not os.path.exists(items_file):
        prep(bench)
    out = out_path(bench, who, cap)
    if os.path.exists(out):
        print("exists:", out); return
    jid = f"bench_{bench}_{who}_cap{cap}"
    job = {"id": jid, "type": "bench", "bench": bench, "priority": priority, "created": time.time(),
           "weights": weights, "items_file": items_file, "cap": cap, "room": room, "out": out}
    p = os.path.join(QUEUE, "pending", jid + ".json")
    json.dump(job, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
    print("queued", jid)


# ---------------------------------------------------------------- scoring
def _num(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def score_gsm8k(row, meta):
    c = row["content"]
    m = re.findall(r"answer is[:\s]*\**\s*\$?\\?\(?\s*(-?[\d,]*\.?\d+)", c)
    pred = m[-1] if m else (re.findall(r"-?[\d,]*\.?\d+", c) or [None])[-1]
    v = _num(pred) if pred else None
    return {"pred": pred, "correct": int(v is not None and abs(v - float(meta["gold"])) < 1e-6)}


def score_mc(row, meta):
    m = re.findall(r"answer is \(?\*?\*?([A-J])\)?", row["content"])
    pred = m[-1] if m else None
    return {"pred": pred, "correct": int(pred == meta["gold"])}


def score_math(row, meta):
    from math_verify import parse, verify
    c = row["content"]
    if "\\boxed" not in c:
        return {"pred": None, "correct": 0}
    try:
        g = parse(f"${meta['gold']}$"); p = parse(c)
        ok = bool(p) and verify(g, p)
    except Exception:
        p, ok = None, False
    return {"pred": str(p[-1]) if p else None, "correct": int(ok)}


def score_ifeval(rows, items):
    os.environ.setdefault("NLTK_DATA", "/home/user/nltk_data")   # punkt_tab fetched by hand (D62)
    sys.path.insert(0, IFEVAL_SRC)
    from instruction_following_eval import evaluation_lib as ev
    out = {}
    for it in items:
        r = rows[it["id"]]; m = it["meta"]
        inp = ev.InputExample(key=int(m["key"]), instruction_id_list=m["instruction_id_list"],
                              prompt=it["messages"][0]["content"], kwargs=m["kwargs"])
        resp = "" if r["cut"] else r["content"].strip()
        p2r = {inp.prompt: resp}
        s = ev.test_instruction_following_strict(inp, p2r); l = ev.test_instruction_following_loose(inp, p2r)
        out[it["id"]] = {"correct": int(s.follow_all_instructions), "loose": int(l.follow_all_instructions),
                         "inst_strict": s.follow_instruction_list, "inst_loose": l.follow_instruction_list}
    return out


def score(bench, rows, items):
    if bench == "ifeval":
        return score_ifeval(rows, items)
    f = {"gsm8k": score_gsm8k, "medqa": score_mc, "math500": score_math}[bench]
    res = {}
    for it in items:
        r = rows[it["id"]]
        s = f(r, it["meta"])
        if r["cut"]:
            s["correct"] = 0
        res[it["id"]] = s
    return res


def _paired(a, b, key="correct"):
    ids = sorted(set(a) & set(b)); n = len(ids)
    d = [b[i][key] - a[i][key] for i in ids]; m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    return {"n": n, "diff": round(100 * m, 2), "se": round(100 * se, 2), "z": round(m / se, 2) if se else None,
            "plus": sum(x > 0 for x in d), "minus": sum(x < 0 for x in d)}


def grade(bench, whos=("base", "champion"), cap=8192, write=True):
    items = [json.loads(l) for l in open(os.path.join(BDIR, bench, "items.jsonl"))]
    res = {}
    for w in whos:
        name, _ = _who(w); p = out_path(bench, name, cap)
        if not os.path.exists(p):
            print(f"{w}: not run yet ({p})"); continue
        rows = {r["id"]: r for r in map(json.loads, open(p))}
        sc = score(bench, rows, items)
        n = len(items); acc = sum(s["correct"] for s in sc.values()) / n
        summ = {"n": n, "acc": round(100 * acc, 2), "se": round(100 * math.sqrt(acc * (1 - acc) / n), 2),
                "cut": round(100 * sum(rows[i]["cut"] for i in rows) / n, 1),
                "mean_tokens": round(sum(rows[i]["tokens"] for i in rows) / n)}
        if bench == "ifeval":
            summ["prompt_loose"] = round(100 * sum(s["loose"] for s in sc.values()) / n, 2)
            inst = [x for s in sc.values() for x in s["inst_strict"]]; il = [x for s in sc.values() for x in s["inst_loose"]]
            summ["inst_strict"] = round(100 * sum(inst) / len(inst), 2); summ["inst_loose"] = round(100 * sum(il) / len(il), 2)
        for i in sc:
            sc[i]["cut"] = rows[i]["cut"]; sc[i]["tokens"] = rows[i]["tokens"]
        res[w] = (summ, sc)
        print(f"{w} ({name}): {json.dumps(summ)}")
    if "base" in res and "champion" in res:
        pr = _paired(res["base"][1], res["champion"][1])
        print("champion vs base (paired):", json.dumps(pr))
        if bench == "ifeval":
            print("  loose:", json.dumps(_paired(res["base"][1], res["champion"][1], "loose")))
        if write:
            json.dump({"bench": bench, "cap": cap, "summary": {w: v[0] for w, v in res.items()}, "paired": pr,
                       "scores": {w: v[1] for w, v in res.items()}},
                      open(os.path.join(BDIR, bench, f"graded_cap{cap}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "queue", "grade"]); ap.add_argument("bench", choices=list(BENCHES))
    ap.add_argument("who", nargs="?", default="base,champion"); ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--priority", type=int, default=2); ap.add_argument("--room", type=int, default=1024)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.bench)
    elif a.cmd == "queue":
        for w in a.who.split(","):
            queue(a.bench, w, a.cap, a.priority, a.room)
    else:
        grade(a.bench, a.who.split(","), a.cap)
