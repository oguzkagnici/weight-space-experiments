# Hypothesis: H12's option selection carries a slot (position) bias that over-picks B/C; it lives in the slot-mean structure of the group-3 keys read by the (nearly constant) query. Removing the slot-mean key subspace from the query (rank-k, CV) should flatten H12's picks; measure the ceiling via per-slot constant correction.
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
H = 12; G = 3
data = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    K = np.asarray(Kn[a:b, G]).astype(np.float64)
    data.append(dict(K=K, q=np.asarray(Qn[b - 1, H]).astype(np.float64), toks=[list(range(s0, s1 + 2)) for (s0, s1) in o["spans"]], spans=o["spans"], gold=o["gold"], n=o["n"]))
n = len(data)

def scores(qv, d):
    lg = d["K"] @ qv / np.sqrt(128)
    allz = np.logaddexp.reduce(lg)
    return np.array([np.logaddexp.reduce(lg[t]) for t in d["toks"]]), allz

S0 = [scores(d["q"], d)[0] for d in data]
ten = [i for i in range(n) if data[i]["n"] == 10]
# slot bias of the H12 option score (centered within question), 10-option questions
B = np.array([S0[i] - S0[i].mean() for i in ten])
print("H12 centered option score by slot (10-opt): ", np.round(B.mean(0), 2))
print("  of which gold-adjusted (exclude gold):", np.round(np.array([np.where(np.arange(10) == data[i]['gold'], np.nan, S0[i] - S0[i].mean()) for i in ten]).__array__().__iter__().__next__()[:0], 2) if False else "")
# CV per-slot correction ceiling
folds = [ten[0::2], ten[1::2]]
acc0 = np.mean([np.argmax(S0[i]) == data[i]["gold"] for i in ten])
accc = []
for f in range(2):
    bias = np.mean([S0[i] - S0[i].mean() for i in folds[1 - f]], 0)
    for i in folds[f]:
        accc.append(np.argmax(S0[i] - bias) == data[i]["gold"])
print("10-opt H12 argmax acc %.3f -> CV slot-debiased %.3f (n=%d)" % (acc0, np.mean(accc), len(ten)))

# slot-mean keys: per question, mean key over option-span tokens per slot, centered over slots
def slot_keys(idx):
    M = []
    for i in idx:
        d = data[i]
        if d["n"] != 10: continue
        km = np.array([d["K"][t].mean(0) for t in d["toks"]])
        M.append(km - km.mean(0))
    return np.mean(M, 0)  # 10 x 128

for kdim in [1, 2, 3, 5, 9]:
    accs, lps, cnt, inmass = [], [], np.zeros(10), []
    for f in range(2):
        Sm = slot_keys(folds[1 - f])
        U, s, Vt = np.linalg.svd(Sm, full_matrices=False)
        P = Vt[:kdim]  # directions in normalized-key space
        for i in folds[f]:
            d = data[i]; q = d["q"]
            qq = q - P.T @ (P @ q)
            qq *= np.linalg.norm(q) / np.linalg.norm(qq)
            s_, z = scores(qq, d)
            accs.append(np.argmax(s_) == d["gold"]); lps.append(s_[d["gold"]] - np.logaddexp.reduce(s_)); cnt[np.argmax(s_)] += 1
            inmass.append(np.exp(np.logaddexp.reduce(s_) - z))
    print("remove top-%d slot dirs from q: acc %.3f lshare %.3f optmass %.3f counts %s" % (kdim, np.mean(accs), np.mean(lps), np.mean(inmass), cnt.astype(int)))
print("gold counts", np.bincount([data[i]["gold"] for i in ten], minlength=10))
print("orig counts", np.bincount([np.argmax(S0[i]) for i in ten], minlength=10))
Sm = slot_keys(ten); U, s, Vt = np.linalg.svd(Sm, full_matrices=False)
print("slot-mean key SVs", np.round(s, 2))
print("qbar . slot means", np.round(Sm @ np.mean([d['q'] for d in data], 0) / np.sqrt(128), 2))
