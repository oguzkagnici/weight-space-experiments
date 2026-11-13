# H16: EARLY CLOSURE: within the same mixed question, wrong samples lock onto their pick earlier (and onto the no-thinking prior) than right samples;
# text marker: an early hypothesis statement ("sounds like / likely / probably / I think / answer is") before any systematic option sweep.
import json, re, numpy as np, collections
from wsp.editor import *
ix = json.load(open("work/probe2_index.json")); tr = json.load(open("work/probe2_traj.json"))
LET = "ABCDEFGHIJ"; CK = [25, 50, 100, 200, 400, 800]
meta = {(m["id"], m["sample"]): m for m in ix if m["sample"] is not None}
by = collections.defaultdict(list)
for key, d in tr.items():
    q, s = key.split("|"); m = meta[(q, int(s))]; f = LET.index(m["pred"])
    tops = [int(np.argmax(p)) for p in d]
    lock = next((CK[i] for i in range(len(CK)) if all(t == f for t in tops[i:])), 1600)
    by[q].append((m["correct"], lock))
d = []
for q, v in by.items():
    R = [l for c, l in v if c]; W = [l for c, l in v if not c]
    if R and W: d.append(np.mean(np.log2(R)) - np.mean(np.log2(W)))
d = np.array(d); print("within-question log2(lock) right - wrong:", d.round(2).tolist(), "mean", round(d.mean(), 2), "SE", round(d.std(ddof=1) / np.sqrt(len(d)), 2), "n", len(d))
# text marker across ALL finished passk samples
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
HYP = re.compile(r"(sounds like|seems like|this is (likely|probably)|likely (the|answer|option)|probably (the|answer|option)|i think (it'?s|the answer)|answer is (likely|probably)|answer (would|should) be)", re.I)
SWEEP = re.compile(r"(let'?s (examine|evaluate|analy[sz]e|go through|consider|check) (each|the|all)|examine each option|evaluate each option|option by option)", re.I)
tab = collections.Counter(); byq = collections.defaultdict(list)
for r in rows:
    if r["cut"]: continue
    t = r["think"]; full = trace_text(r); pr = render_think(r["id"])
    gen = full[len(pr):]
    head = tok.decode(tok(gen, add_special_tokens=False).input_ids[:250])
    h = HYP.search(head); sw = SWEEP.search(head)
    early = bool(h) and (not sw or h.start() < sw.start())
    tab[(early, r["correct"])] += 1; byq[r["id"]].append((early, r["correct"]))
for e in (True, False):
    a, b = tab[(e, 1)], tab[(e, 0)]; print(f"early hypothesis in first 250 tokens = {e}: n {a+b} acc {100*a/max(1,a+b):.1f}")
pairs = []
for q, v in byq.items():
    A = [c for e, c in v if e]; B = [c for e, c in v if not e]
    if A and B: pairs.append(np.mean(A) - np.mean(B))
p = np.array(pairs); print("within-question acc(early hyp) - acc(no early hyp):", round(p.mean(), 3), "SE", round(p.std(ddof=1) / np.sqrt(len(p)), 3), "questions", len(p))
