# Analyse pass 1: after a corrupted step vs the clean one: (1) surprisal of the corrupted token, (2) P(re-check opener) next, (3) a linear
# corrupt-vs-clean direction fit on half the questions and tested (paired accuracy, AUC) on the other half, per layer and per kind.
import json, sys, numpy as np
from wsp.editor import *
D = sys.argv[1] if len(sys.argv) > 1 else "work/fwd_01/"
P = json.load(open("work/pairs1.json")); n = len(P)
meta = json.load(open(D + "meta.json"))
tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy")
def ids(ws): return {tok(w, add_special_tokens=False).input_ids[0] for w in ws}
NARROW = ids(["Wait", " Wait", "Actually", " Actually", "Hmm", " Hmm", "Hold", " Hold", "Oops", " Oops", "No", " No", "Hmm,", "wait", " wait", "Hmm.", "Let", " Let", "Check", " Check", "Double", " Double"])
BROAD = NARROW | ids(["But", " But", "However", " However"])
def pm(t, S): return sum(np.exp(b) for a, b in zip(tk[t], tl[t]) if int(a) in S)
kinds = np.array([p["kind"] for p in P])
for nm, S in (("narrow re-check", NARROW), ("broad (+But/However)", BROAD)):
    c = np.array([pm(2 * i, S) for i in range(n)]); x = np.array([pm(2 * i + 1, S) for i in range(n)])
    for k in ("swap", "arith"):
        m = kinds == k
        print(f"{nm:22s} {k:5s}: P clean {c[m].mean():.3f}  corrupt {x[m].mean():.3f}  ratio {x[m].mean()/max(c[m].mean(),1e-9):.2f}  corrupt>clean {np.mean(x[m] > c[m]):.2f} (n {m.sum()})")
# surprisal of the first differing token
tlp = np.load(D + "token_logprobs.npy") if "token_logprobs" in str(meta.get("files")) else None
if tlp is not None:
    nt = json.loads(meta["n_tokens"]) if isinstance(meta["n_tokens"], str) else meta["n_tokens"]
    off = np.concatenate([[0], np.cumsum(np.array(nt) - 1)])
    print("server tokens vs local (first text):", nt[0], len(tok(P[0]["clean"], add_special_tokens=False).input_ids))
    s_c, s_x = [], []
    for i, p in enumerate(P):
        a = tok(p["clean"], add_special_tokens=False).input_ids; b = tok(p["corrupt"], add_special_tokens=False).input_ids
        j = next(k for k in range(min(len(a), len(b))) if a[k] != b[k])
        A = tlp[off[2 * i]: off[2 * i + 1]]; B = tlp[off[2 * i + 1]: off[2 * i + 2]]
        # token_logprobs[k] = logprob of token k given prefix (assume aligned so that index k is token k; check len)
        sh = len(A) - (len(a) - 1)  # extra leading tokens on the server side (e.g. BOS)
        s_c.append(A[j - 1 + sh]); s_x.append(B[j - 1 + sh])
    s_c, s_x = np.array(s_c), np.array(s_x)
    for k in ("swap", "arith"):
        m = kinds == k
        print(f"logprob of first changed token {k}: clean {np.median(s_c[m]):.2f}  corrupt {np.median(s_x[m]):.2f}  (median); corrupt lower in {np.mean(s_x[m] < s_c[m]):.2f}")
    np.save("work/surpr1.npy", np.stack([s_c, s_x]))
# probe
qs = sorted({p["id"] for p in P}); rng = np.random.default_rng(0); res = {}
for L in [8, 12, 14, 16, 18, 20, 22, 24]:
    H = np.load(D + f"h{L}.npy"); C = H[0::2]; X = H[1::2]; d = X - C
    accs = {k: [] for k in ("swap", "arith", "all")}; aucs = []
    for rep in range(20):
        tr = set(rng.choice(qs, len(qs) // 2, replace=False)); m = np.array([p["id"] in tr for p in P])
        for k in ("swap", "arith", "all"):
            mk = (kinds == k) if k != "all" else np.ones(n, bool)
            w = d[m & mk].mean(0) if (m & mk).sum() else d[m].mean(0)
            te = (~m) & mk
            if te.sum() == 0: continue
            accs[k].append(np.mean((X[te] @ w) > (C[te] @ w)))
    print(f"h{L}: paired held-out acc  swap {np.mean(accs['swap']):.2f}  arith {np.mean(accs['arith']):.2f}  all {np.mean(accs['all']):.2f};  |mean diff|/|h| {np.linalg.norm(d.mean(0))/np.linalg.norm(C,axis=1).mean():.4f}")
