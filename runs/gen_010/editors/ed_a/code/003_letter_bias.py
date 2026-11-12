# Hypothesis: thinking-mode finished answers still over-pick early letters (as direct mode did); check pred vs gold counts and wrong-answer letter offsets.
import json, collections, re
L='ABCDEFGHIJ'
for f in ['passk_gen008champ_cap8192_det1.jsonl','trace_bank_gen008champ_cap3072_det1.jsonl']:
    rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/'+f)]
    fin=[r for r in rows if not r['cut'] and r['pred']]
    pc=collections.Counter(r['pred'] for r in fin); gc=collections.Counter(r['gold'] for r in fin)
    wr=[r for r in fin if not r['correct']]
    print(f, len(rows), 'finished', len(fin), 'wrong', len(wr))
    print(' pred', [pc[c] for c in L]); print(' gold', [gc[c] for c in L])
    off=collections.Counter(L.index(r['pred'])-L.index(r['gold']) for r in wr)
    print(' wrong pred-gold offset', sorted(off.items()))
    print(' wrong pred earlier than gold', sum(1 for r in wr if r['pred']<r['gold']), 'later', sum(1 for r in wr if r['pred']>r['gold']))
