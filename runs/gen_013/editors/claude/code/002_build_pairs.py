# Hypothesis: the network registers a just-written FALSE step. Build matched clean / corrupted prefixes from the parent's own RIGHT, finished
# practice traces (excluding mixed pass@4 questions, kept for testing): (a) antonym/negation swap in one sentence, (b) a correct "a op b = c"
# result changed. Each text ends right after the step's sentence and its original separator, so the next token is where a re-check would start.
import json, re, random, collections
from wsp.editor import *
random.seed(13)
S = "/home/user/weight-space-experiments/runs/shared/"
pk = [json.loads(l) for l in open(S + "passk_gen008champ_cap8192_det1.jsonl")]
byq = collections.defaultdict(list)
for r in pk: byq[r["id"]].append(r)
mixed = {q for q, v in byq.items() if 0 < sum(x["correct"] for x in v) < 4}
rows = [r for r in pk if r["correct"] and not r["cut"] and r["id"] not in mixed]
rows += [json.loads(l) for l in open(S + "trace_bank_gen008champ_cap8192_det1.jsonl")]
rows = [r for r in rows if r["correct"] and not r["cut"] and r["id"] not in mixed]
SW = [("increases", "decreases"), ("increase", "decrease"), ("higher", "lower"), ("larger", "smaller"), ("more", "less"), ("greater", "smaller"),
      ("true", "false"), ("correct", "incorrect"), ("can", "cannot"), ("positive", "negative"), ("before", "after"), ("always", "never"),
      ("is not", "is"), ("does not", "does"), ("rise", "fall"), ("maximum", "minimum"), ("inside", "outside"), ("valid", "invalid"),
      ("consistent", "inconsistent"), ("allowed", "prohibited"), ("legal", "illegal"), ("first", "last"), ("strong", "weak")]
pairs = []
for r in rows:
    tt = trace_text(r); p0 = tt.index("<think>") + len("<think>") if "<think>" in tt else 0
    body = tt[p0:]
    # candidate sentence ends: ". " or ".\n"
    sents = [m for m in re.finditer(r"[^.\n]{25,240}?\.(\s+)", body)]
    random.shuffle(sents)
    got = 0
    for m in sents:
        if m.end() > 4500 or m.start() < 150: continue
        s = m.group(0); sep = m.group(1); core = s[: len(s) - len(sep)]
        for a, b in SW:
            for x, y in ((a, b), (b, a)):
                mm = re.search(rf"\b{re.escape(x)}\b", core)
                if mm and not (x == "is" and re.search(r"\bis not\b", core)):
                    bad = core[: mm.start()] + y + core[mm.end():]
                    pairs.append(dict(kind="swap", id=r["id"], s=r.get("sample", 0), w=f"{x}->{y}", clean=tt[: p0 + m.start()] + core + sep,
                                      corrupt=tt[: p0 + m.start()] + bad + sep, sent=core[-160:]))
                    got += 1; break
            if got: break
        if got >= 2: break
    for m in re.finditer(r"(?:^|(?<=[\s(:;]))(\d+(?:\.\d+)?)\s*([+*×-])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)([.,;]?\s)", body):
        if m.end() > 4500 or m.start() < 150: continue
        a, op, c, res = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        v = {"+": a + c, "-": a - c, "*": a * c, "×": a * c}[op]
        if abs(v - float(res)) > 1e-6 * max(1, abs(v)) or abs(v) < 4: continue
        d = res.split(".")[1] if "." in res else ""
        wrong = float(res) * random.choice([1.2, 0.8, 1.5, 0.7]) + (1 if not d else 0)
        wr = f"{wrong:.{len(d)}f}" if d else str(int(round(wrong)))
        if wr == res: continue
        st = p0 + m.start(4)
        pairs.append(dict(kind="arith", id=r["id"], s=r.get("sample", 0), w=f"{res}->{wr}", clean=tt[: p0 + m.end()], corrupt=tt[:st] + wr + tt[p0 + m.end(4): p0 + m.end()], sent=m.group(0)))
        break
n = lambda t: len(tok(t, add_special_tokens=False).input_ids)
random.shuffle(pairs)
sel, tot, cnt = [], 0, collections.Counter()
for p in sorted(pairs, key=lambda p: (p["kind"] != "arith", random.random())):
    if cnt[p["kind"]] >= (40 if p["kind"] == "arith" else 100): continue
    k = n(p["clean"]) + n(p["corrupt"])
    if k > 2 * 3000 or tot + k > 199000: continue
    sel.append(p); tot += k; cnt[p["kind"]] += 1
print("candidate pairs", len(pairs), "selected", dict(cnt), "tokens", tot, "questions", len({p["id"] for p in sel}))
for p in sel[:3] + [q for q in sel if q["kind"] == "swap"][:6]: print(p["kind"], p["w"], "|", p["sent"][-140:].encode("ascii", "replace").decode())
texts = []
for p in sel: texts += [p["clean"], p["corrupt"]]
json.dump(sel, open("work/pairs1.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"token_logprobs": True, "topk": 50, "positions": "last",
           "hidden_states": [8, 12, 14, 16, 18, 20, 22, 24]}}, open("work/req_read1.json", "w"))
