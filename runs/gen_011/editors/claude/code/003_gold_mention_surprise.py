# H3: in wrong samples, the judgement of the GOLD option is a low-probability sampled continuation (the model's own preferred verdict was different);
# measured with gen-10 pass-1 token log-probs (50 finished samples of the 15 mixed questions, first 4,090 tokens).
import json, numpy as np, re, collections
from wsp.editor import *
G10 = "/home/user/weight-space-experiments/runs/gen_010/editors/claude/work/"
idx = json.load(open(G10 + "read1_index.json")); req = json.load(open(G10 + "req_read1.json"))
meta = json.load(open(G10 + "fwd_01/meta.json")); lp = np.load(G10 + "fwd_01/token_logprobs.npy")
Q = questions(); off = 0; stats = {0: collections.Counter(), 1: collections.Counter()}; ex = []
for m, text, n in zip(idx, req["texts"], meta["n_tokens"]):
    enc = tok(text, add_special_tokens=False); ids = enc.input_ids; assert len(ids) == n
    l = lp[off:off + n - 1]; off += n - 1
    pl = len(tok(render_think(m["id"]), add_special_tokens=False).input_ids)
    # char offsets of tokens
    offs = enc.offset_mapping if hasattr(enc, "offset_mapping") and enc.offset_mapping else None
    if offs is None:
        cur = 0; offs = []
        for i in ids:
            s = tok.decode([i]); offs.append((cur, cur + len(s))); cur += len(s)
    plen = len(tok.decode(ids[:pl]))
    gold = m["gold"]; q = Q[m["id"]]; gtext = q["options"][q["answer_index"]][:25].lower()
    think = text[plen:]
    # mentions of gold: "(X)" or "Option X" or option text
    pat = re.compile(r"\(%s\)|[Oo]ption %s\b|\b%s\)" % (gold, gold, gold))
    ments = [mm.start() + plen for mm in pat.finditer(think)]
    k = m["correct"]
    for c in ments:
        # tokens within the next 250 chars after the mention
        js = [j for j in range(pl, n) if c <= offs[j][0] < c + 250]
        ps = np.exp(l[[j - 1 for j in js]]) if js else np.array([1.0])
        stats[k]["mentions"] += 1; stats[k]["low10"] += int((ps < 0.1).sum()); stats[k]["toks"] += len(js)
        if len(js) and ps.min() < 0.05:
            j = js[int(ps.argmin())]
            ctx = text[max(c, offs[j][0] - 120):offs[j][1] + 40].replace("\n", " | ")
            ex.append((k, m["id"][-5:], m["sample"], round(float(ps.min()), 3), ctx.encode("ascii", "replace").decode()))
for k in (1, 0):
    s = stats[k]; print("right" if k else "wrong", dict(s), "low<.1 per 1k tok after gold mention:", round(1000 * s["low10"] / max(1, s["toks"]), 1),
                         "mentions/trace", round(s["mentions"] / sum(1 for m in idx if m["correct"] == k), 1))
import random; random.seed(0)
for e in [e for e in ex if e[0] == 0][:14]: print(e)
