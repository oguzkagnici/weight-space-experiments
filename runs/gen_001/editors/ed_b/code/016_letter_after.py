# Hypothesis: after E1 the remaining per-letter preference on unseen questions is small; check mean centered log-prob per letter (10-option q) for original vs edited in passes 2,4,5.
import json, numpy as np, sys
for k in [2, 3, 4, 5] + [int(a) for a in sys.argv[1:]]:
    for m in ['original', 'edited']:
        try: rows = [json.loads(l) for l in open(f'work/fwd_{k:02d}/score_{k}_{m}.jsonl')]
        except FileNotFoundError: continue
        P = np.array([r['probs'] for r in rows if len(r['probs']) == 10]); lp = np.log(P + 1e-9); lp -= lp.mean(1, keepdims=True)
        pr = np.array([r['pred'] for r in rows]); an = np.array([r['answer'] for r in rows])
        print(k, m[:4], 'mean centered logp', np.round(lp.mean(0), 2), 'acc by gold A-E vs F-J: %.3f %.3f' % ((pr == an)[an < 5].mean(), (pr == an)[an >= 5].mean()))
