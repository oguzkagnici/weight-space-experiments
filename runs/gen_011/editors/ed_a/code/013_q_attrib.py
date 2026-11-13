# Hypothesis: the first-named letter's identity in L22.H12's query (path 2: retrieving that letter's option line / mentions) is written by a few upstream heads that attend first>second (L20 H9/H13, L14 H15, L12 H0 ...); linear attribution of H12's query (through operator_norm, W_q, q_norm, rope) to recorded component writes finds them.
import json, re
import numpy as np
from wsp.editor import *
D1, D2 = "work/fwd_01", "work/fwd_02"
probes = json.load(open("work/p1_probes.json")); rows = json.load(open("work/p1_letterpos.json"))
off = np.load(D1 + "/offsets_tok.npy")
K = np.load(D2 + "/out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
Qrec = np.load("work/p2_q.npy")[0]  # H12 recovered post-rope query per text
inv = 1.0 / (1e6 ** (np.arange(0, 64, 2) / 128.0)); inv = np.concatenate([inv, np.zeros(32)])
def rope(k, pos):
    fr = np.atleast_1d(pos)[:, None] * inv[None, :]; emb = np.concatenate([fr, fr], 1)
    rh = np.concatenate([-k[:, 64:], k[:, :64]], 1); return k * np.cos(emb) + rh * np.sin(emb)
Wq = load("model.layers.22.self_attn.q_proj.weight").numpy()[12 * 128:13 * 128]
gq = load("model.layers.22.self_attn.q_norm.weight").numpy(); gop = load("model.layers.22.operator_norm.weight").numpy()
h22 = np.load(D1 + "/h22.npy").astype(np.float32); r22 = 1 / np.sqrt((h22 ** 2).mean(1) + 1e-5)
qpre = (h22 * r22[:, None] * gop) @ Wq.T; rq = 1 / np.sqrt((qpre ** 2).mean(1) + 1e-5)
# check reconstruction vs recovered
T_ = np.array([r[2] for r in rows])
qhat = rope(gq * qpre * rq[:, None], T_ - 1.0) if False else np.array([rope((gq * qpre[t] * rq[t])[None], np.array([T_[t] - 1.0]))[0] for t in range(len(rows))])
cs = [qhat[t] @ Qrec[t] / np.linalg.norm(qhat[t]) / np.linalg.norm(Qrec[t]) for t in range(len(rows))]
print("cos(reconstructed q, recovered q) median %.4f min %.4f" % (np.median(cs), np.min(cs)))
# letter keys: the prompt option-letter token and option-end token of letters a,b, plus pair tokens
def letter_key_diff(i, t):
    p = probes[i]; text = p["orig"] if t % 2 == 0 else p["swap"]; pr = render_think(p["id"])
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); om = enc["offset_mapping"]
    k = K[off[t]:off[t + 1], 3].astype(np.float64)
    idx = {}
    for m in re.finditer(r"\n\(([A-J])\) ", pr):
        c = m.start(1); idx[m.group(1)] = [j for j, (a, b) in enumerate(om) if a <= c < b][0]
    ka = rope(k[[idx[p["a"]]]], np.array([idx[p["a"]] * 1.0]))[0]; kb = rope(k[[idx[p["b"]]]], np.array([idx[p["b"]] * 1.0]))[0]
    return ka - kb
# components: per-layer writes (post norms) and per-head writes of attention layers < 22
comps = {}
for l in range(22):
    for nm in ["post_operator_norm", "post_ffn_norm"]:
        comps[f"L{l}.{'mix' if nm[5]=='o' else 'mlp'}"] = np.load(f"{D1}/out_model.layers.{l}.{nm}.npy").astype(np.float32)
for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    x = np.load(f"{D1}/in_model.layers.{l}.self_attn.o_proj.npy").astype(np.float32)
    Wo = load(f"model.layers.{l}.self_attn.o_proj.weight").numpy(); gp = load(f"model.layers.{l}.post_operator_norm.weight").numpy()
    y = x @ Wo.T; ry = 1 / np.sqrt((y ** 2).mean(1) + 1e-5)
    for h in range(16):
        comps[f"L{l}H{h}"] = gp * (x[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T) * ry[:, None]
kd = {}
res = {}
for nm, c in comps.items():
    qc = (c * r22[:, None] * gop) @ Wq.T * rq[:, None] * gq   # pre-rope contribution
    vals = []
    for i in range(len(probes)):
        v2 = []
        for t in (2 * i, 2 * i + 1):
            if t not in kd: kd[t] = letter_key_diff(i, t)
            qr = rope(qc[t][None], np.array([T_[t] - 1.0]))[0]
            v2.append(qr @ kd[t] / np.sqrt(128))   # score preference for letter a's option slot over b's
        vals.append((v2[0] - v2[1]) / 2)          # first-named advantage
    res[nm] = np.array(vals)
kind = np.array([p["kind"] for p in probes])
tot = np.array([(Qrec[2*i] @ kd[2*i] - Qrec[2*i+1] @ kd[2*i+1]) / 2 / np.sqrt(128) for i in range(len(probes))])
print("H12 query: first-named advantage in score at the prompt option-letter token: real %.3f syn %.3f" % (tot[kind=='real'].mean(), tot[kind=='syn'].mean()))
layer_only = {k: v for k, v in res.items() if "." in k}
print("sum of layer writes: real %.3f syn %.3f" % (sum(layer_only.values())[kind=='real'].mean(), sum(layer_only.values())[kind=='syn'].mean()))
for k, v in sorted(res.items(), key=lambda kv: -abs(kv[1].mean()))[:22]:
    print(f"{k:9s} real {v[kind=='real'].mean():+.3f} syn {v[kind=='syn'].mean():+.3f}")
json.dump({k: [float(v[kind=='real'].mean()), float(v[kind=='syn'].mean())] for k, v in res.items()}, open("work/p2_qattrib.json", "w"))
