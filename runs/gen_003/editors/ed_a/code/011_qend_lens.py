# Hypothesis: at the end of the question (before the option list) the residual already "predicts" the answer text: the
# logit lens there (h14/h21) scores the gold option's text tokens above the others, i.e. generative knowledge exists that
# the option-end belief may not use.
import json, numpy as np, torch
from wsp.editor import *
D1, D2 = 'work/fwd_01/', 'work/fwd_02/'
ids = json.load(open('work/pass2_ids.json')); SP = json.load(open('work/spans_p1.json'))
N = len(ids); off2 = np.load(D2 + 'offsets.npy')
gold = np.load('work/p1_gold.npy')[:N]; nopt = np.load('work/p1_nopt.npy')[:N]; LP = np.load('work/p1_letterlogp.npy')[:N]
mask = np.arange(10)[None] < nopt[:, None]
final = np.where(mask, LP, -1e9).argmax(1); wrong = final != gold
E = load_base('model.embed_tokens.weight'); wn = load_base('model.embedding_norm.weight')
for layer in [14, 21]:
    H = np.load(D2 + f'h{layer}.npy', mmap_mode='r')
    sc_first = np.full((N, 10), -1e9); sc_mean = np.full((N, 10), -1e9)
    for i, q in enumerate(ids):
        s = render(q); enc = tok(s, add_special_tokens=False)['input_ids']
        sp = SP[q]['spans']
        qend = sp[0][0] - 1  # token before "(A"
        h = torch.tensor(H[off2[i] + qend].astype(np.float32))
        h = h / h.pow(2).mean().add(1e-5).sqrt() * wn
        for k, (t0, t1) in enumerate(sp):
            tt = enc[t0 + 2:t1 + 1] or [enc[t1]]
            lg = E[tt] @ h
            sc_first[i, k] = float(lg[0]); sc_mean[i, k] = float(lg.mean())
    for nm, S in [('first', sc_first), ('mean', sc_mean)]:
        p = S.argmax(1)
        ex = [np.argmax(np.where((np.arange(10) < nopt[i]) & (np.arange(10) != final[i]), S[i], -1e9)) == gold[i] for i in np.where(wrong)[0]]
        print(f'h{layer} q-end lens, option {nm}-token logit: acc {(p==gold).mean():.3f}; on final-wrong excluding final pick {np.mean(ex):.3f} (chance {np.mean(1/(nopt[wrong]-1)):.3f})')
