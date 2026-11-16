# Hypothesis: quoting the decisive stem cue inside a wrong trace (vs quoting a non-decisive stem sentence, vs nothing) raises P(gold) over P(wrong pick) at a forced conclusion; build the read request.
import json, re, sys, collections
sys.path.insert(0, "code")
from wsp.editor import *
from importlib import import_module
cm = import_module("005_cues"); CUES, CTRL = cm.CUES, cm.CTRL
Q = questions()
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pkids = {r["id"] for r in pk}
allr = pk + [dict(r, sample=0) for r in bk if r["id"] not in pkids]
NEUTRAL = "Let me take a moment and consider every option carefully before I decide on the final answer."
FRAME = '\n\nLet me re-read the key part of the question: "%s"'
PROBE = "\n\nThe answer is ("
def ntok(s): return len(tok.encode(s))
def cuts(think, pred):
    # think excludes leading <think>; paragraph boundaries = positions right after "\n\n"
    b = [m.end() for m in re.finditer(r"\n\n", think)]
    if not b: return None, None
    L = len(think)
    early = min(b, key=lambda x: abs(x - 0.3 * L))
    pat = re.compile(r"(answer|choose|pick|go with|select|correct)[^\n]{0,40}\(?\b%s\b\)?" % pred, re.I)
    late = None
    for m in pat.finditer(think):
        if m.start() > 0.5 * L:
            prev = [x for x in b if x <= m.start()]
            late = prev[-1] if prev else None
            break
    return early, late
texts, meta = [], []
for r in allr:
    q = r["id"]
    if q not in CUES or r["cut"]: continue
    think = r["raw"][len("<think>"):] if r["raw"].startswith("<think>") else r["raw"]
    think = think.split("</think>")[0]
    e, l = cuts(think, r["pred"])
    ctrl = CTRL[q] or NEUTRAL
    for cname, c in (("early", e), ("late", l)):
        if c is None: continue
        pre = render_think(q) + think[:c].rstrip("\n")
        for v, ins in (("none", ""), ("cue", FRAME % CUES[q]), ("ctrl", FRAME % ctrl)):
            texts.append(pre + ins + PROBE)
            meta.append(dict(id=q, sample=r["sample"], correct=r["correct"], pred=r["pred"], gold=Q[q]["answer"], cut=cname, variant=v, frac=c / len(think)))
tot = sum(ntok(t) for t in texts)
print("texts", len(texts), "tokens", tot, "max", max(ntok(t) for t in texts))
c = collections.Counter((m["cut"], m["correct"]) for m in meta if m["variant"] == "none"); print(c)
json.dump(meta, open("work/insert_meta.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts, "return": {"topk": 50, "positions": "last"}}, open("work/req_p1.json", "w"))
for m, t in list(zip(meta, texts))[:0]: pass
if len(sys.argv) > 1:
    for m, t in zip(meta, texts):
        if m["variant"] == "cue" and m["id"] == sys.argv[1]:
            i = t.rfind("Let me re-read"); print(m["cut"], round(m["frac"], 2), repr(t[i-250:i].encode("ascii","replace").decode()))
