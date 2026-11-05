# Hypothesis check: where does H12's 17% non-option attention go (BOS, question, instruction, assistant prefix), and is it content-free?
import numpy as np, pickle
from wsp.editor import render, tok
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb")); off = np.load(D + "offsets_tok.npy")
A = np.load(D + "attn_last_L22.npy").astype(np.float64)
acc = {"bos": 0, "question": 0, "after_options": 0, "last5": 0}
tokmass = {}
for i, o in enumerate(sp):
    att = A[off[i]:off[i + 1], 12]; T = len(att)
    s_first = o["spans"][0][0]; s_last = o["spans"][-1][1] + 1
    acc["bos"] += att[0]; acc["question"] += att[1:s_first].sum(); acc["after_options"] += att[s_last + 1:T - 5].sum(); acc["last5"] += att[T - 5:].sum()
    ids = tok(render(o["id"]), add_special_tokens=False)["input_ids"]
    for t in list(range(s_last + 1, T)):
        tokmass[ids[t]] = tokmass.get(ids[t], 0) + att[t]
print({k: round(v / len(sp), 3) for k, v in acc.items()})
top = sorted(tokmass.items(), key=lambda x: -x[1])[:8]
print([(repr(tok.decode([k])), round(v / len(sp), 3)) for k, v in top])
