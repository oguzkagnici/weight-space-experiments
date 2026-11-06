# Hypothesis: edited traces diverge from the original at a gated churn break, where the edited text opens with Hence/Therefore and then commits; cut traces become finished answers, finished traces keep their answer.
import json, sys, glob, os
K=sys.argv[1]
D=f'work/fwd_{K}'
f_o=glob.glob(f'{D}/gen_*_original.jsonl')[0]; f_e=glob.glob(f'{D}/gen_*_edited.jsonl')[0]
O={json.loads(l)['id']:json.loads(l) for l in open(f_o)}; E={json.loads(l)['id']:json.loads(l) for l in open(f_e)}
up=dn=0; s=[]
for q in O:
    o,e=O[q],E[q]
    a,b=o['raw'],e['raw']; i=0
    while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
    same=(a==b)
    tag='' if o['correct']==e['correct'] else ('W->R' if e['correct'] else 'R->W')
    print(f"{q:14s} {o['category'][:10]:10s} gold {o['gold']} | orig {str(o['pred']):4s} cut {o['cut']} tok {o['tokens']:4d} | edit {str(e['pred']):4s} cut {e['cut']} tok {e['tokens']:4d} | {'IDENTICAL' if same else 'diverge@%d (%.2f)'%(i,i/max(1,len(a)))} {tag}")
    if not same and len(sys.argv)>2:
        ctx=b[max(0,i-120):i+160].replace('\n','/')
        print('      ...',ctx.encode('ascii','replace').decode())
