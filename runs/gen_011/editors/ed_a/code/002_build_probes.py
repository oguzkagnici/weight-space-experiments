# Hypothesis: we can fit ~60 real swap probes (both orders) in one 200k-token read pass if we keep prefixes <= 3,900 tokens.
import json, re, random
from wsp.editor import *
P = json.load(open("work/pairs_all.json"))
SH = "/home/user/weight-space-experiments/runs/shared/"
raws = {}
for f in ["passk_gen008champ_cap8192_det1.jsonl", "trace_bank_gen008champ_cap8192_det1.jsonl"]:
    for l in open(SH + f):
        r = json.loads(l); raws[(r["id"], r["sample"])] = r
SUF = "\n\nThus, the answer is ("
print([tok.decode([t]) for t in tok.encode("Thus, the answer is (B", add_special_tokens=False)])
print(LETTER_IDS, [tok.encode("is (" + c, add_special_tokens=False)[-1] for c in "ABCDEFGHIJ"])
def swap_span(s, a, b):
    # swap letters a and b inside the matched span only
    return re.sub(r"(?<![A-Za-z])([%s%s])(?![A-Za-z])" % (a, b), lambda m: b if m.group(1) == a else a, s)
cands = []
for p in P:
    r = raws.get((p["id"], p["sample"]))
    if r is None: continue
    raw = r["raw"][len("<think>"):] if r["raw"].startswith("<think>") else r["raw"]
    pre = render_think(p["id"])
    t0 = raw[:p["ls"]]; span = raw[p["ls"]:p["le"]]; rest = raw[p["le"]:p["sent_end"]]
    # skip if the rest of the sentence names either letter again (swap would be inconsistent)
    if re.search(r"(?<![A-Za-z])[%s%s](?![A-Za-z])" % (p["a"], p["b"]), rest): continue
    orig = pre + t0 + span + rest.rstrip() + SUF
    sw = pre + t0 + swap_span(span, p["a"], p["b"]) + rest.rstrip() + SUF
    n = len(tok.encode(orig, add_special_tokens=False))
    if n > 3900: continue
    p2 = dict(p); p2.update(n_tok=n, orig=orig, swap=sw, span=span, rest=rest[:80])
    cands.append(p2)
print("candidates <=3900 tok", len(cands), "traces", len({(c['id'], c['sample']) for c in cands}), "questions", len({c['id'] for c in cands}))
# keep at most 2 per trace (the last two), and at most 4 per question
random.seed(0)
by = {}
for c in cands: by.setdefault((c["id"], c["sample"]), []).append(c)
sel = []
for k, v in by.items(): sel += v[-2:]
byq = {}
for c in sel: byq.setdefault(c["id"], []).append(c)
sel = []
for k, v in byq.items(): sel += v[-4:]
tot = sum(2 * c["n_tok"] for c in sel)
print("selected", len(sel), "tokens", tot, "questions", len(byq))
while tot > 190000:
    sel.sort(key=lambda c: -c["n_tok"]); c = sel.pop(0); tot -= 2 * c["n_tok"]
print("final", len(sel), "tokens", tot)
json.dump(sel, open("work/probes_real.json", "w"))
for c in sel[:5]: print(c["id"], c["a"], c["b"], c["gold"], repr(c["span"]), repr(c["rest"][:50]))
