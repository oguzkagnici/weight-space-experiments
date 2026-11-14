# Hypothesis: the early-letter prior (A predicted 44/160 at mid-trace probes vs gold 7) also tilts FINAL picks in generated traces: compare pred vs gold letter counts among finished-wrong answers.
import json, numpy as np, collections
S='/home/user/weight-space-experiments/runs/shared/'
L='ABCDEFGHIJ'
for f in ('passk_gen008champ_cap8192_det1.jsonl','trace_bank_gen008champ_cap8192_det1.jsonl','trace_bank_gen008champ_cap3072_det1.jsonl','trace_bank_base_cap3072_det1.jsonl'):
    R=[json.loads(l) for l in open(S+f)]; fin=[r for r in R if not r['cut'] and r['pred'] in L]
    g=np.bincount([L.index(r['gold']) for r in fin],minlength=10); p=np.bincount([L.index(r['pred']) for r in fin],minlength=10)
    w=[r for r in fin if not r['correct']]
    gw=np.bincount([L.index(r['gold']) for r in w],minlength=10); pw=np.bincount([L.index(r['pred']) for r in w],minlength=10)
    acc=[np.mean([r['correct'] for r in fin if r['gold']==c]) if g[i] else np.nan for i,c in enumerate(L)]
    print(f[:34],'finished',len(fin)); print('  gold ',g,'\n  pred ',p,'\n  wrong: gold',gw,' pred',pw)
    print('  acc by gold letter',np.round(acc,2), ' mean pos gold %.2f pred %.2f; among wrong: gold %.2f pred %.2f'%(np.dot(g,range(10))/g.sum(),np.dot(p,range(10))/p.sum(),np.dot(gw,range(10))/gw.sum(),np.dot(pw,range(10))/pw.sum()))
