# H1b: re-verification after an in-think answer statement costs many tokens and rarely changes the answer (so it is a wasteful episode).
import json, re, numpy as np, collections
from wsp.editor import *
TB = "/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl"
rows = [json.loads(l) for l in open(TB)]
PAT = re.compile(r"(?:answer is|answer should be|correct answer is|so the answer)\s*\(?\*?\*?([A-J])\b", re.I)
stats = []
nxt = collections.Counter()
for r in rows:
    ms = list(PAT.finditer(r["think"]))
    if not ms: continue
    m = ms[0]
    ntok_before = len(tok(render_think(r["id"]) + r["think"][:m.end()], add_special_tokens=False)["input_ids"]) - len(tok(render_think(r["id"]), add_special_tokens=False)["input_ids"])
    after = r["think"][m.end():]
    # first word after the statement's closing punctuation
    w = re.match(r"[\)\*\.\"'\s,]*(\w+)", after)
    first_word = w.group(1) if w else "<end>"
    reopened = first_word in ("But", "However", "Let", "Wait", "Actually", "Alternatively", "Hmm", "or", "unless")
    final = r["pred"] if not r["cut"] else None
    stats.append(dict(id=r["id"], cut=r["cut"], gold=r["gold"], first=m.group(1), final=final, tok_before=ntok_before,
                      tok_after=r["tokens"] - ntok_before, first_word=first_word, reopened=reopened))
    nxt[(first_word, r["cut"])] += 1
S = stats
print("traces with an in-think statement:", len(S))
for c in (0, 1):
    g = [s for s in S if s["cut"] == c]
    print(f"{'cut' if c else 'finished'}: n {len(g)}, tokens before first statement median {np.median([s['tok_before'] for s in g]):.0f}, after {np.median([s['tok_after'] for s in g]):.0f}, mean after {np.mean([s['tok_after'] for s in g]):.0f}")
fin = [s for s in S if not s["cut"]]
ch = [s for s in fin if s["final"] != s["first"]]
print(f"finished: final answer != first stated in {len(ch)} of {len(fin)}; first correct {sum(s['first']==s['gold'] for s in fin)}, final correct {sum(s['final']==s['gold'] for s in fin)}")
for s in ch: print("   changed", s["id"], "first", s["first"], "final", s["final"], "gold", s["gold"], "tok_after", s["tok_after"])
cutg = [s for s in S if s["cut"]]
print(f"cut: first stated correct {sum(s['first']==s['gold'] for s in cutg)} of {len(cutg)} (lost to the cap after a statement)")
print("first word after the statement (word, cut):", nxt.most_common(25))
big = [s for s in fin if s["tok_after"] > 200]
print(f"finished with >200 tokens after the first statement: {len(big)}; those tokens sum {sum(s['tok_after'] for s in big)} of all finished tokens {sum(r['tokens'] for r in rows if not r['cut'])}")
json.dump(S, open("work/post_commit_cost.json", "w"), indent=1)
