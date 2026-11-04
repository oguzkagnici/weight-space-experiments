# Hypothesis: none (extract option-span positions, parent predictions from h24 readout, L22 attention to options; save compact arrays).
from wsp.editor import *
import numpy as np, json, torch
s=json.load(open('work/split.json')); qids=s['dev'][:160]; qs=questions()
off=np.load('work/fwd_01/offsets_tok.npy'); meta=json.load(open('work/fwd_01/meta.json'))
OPT=[17175,20370,16384,21639,32642,25413,21010,28899,28687,44007]
# letter readout
E=load('model.embed_tokens.weight')[LETTER_IDS].numpy(); wn=load('model.embedding_norm.weight').numpy()
np.save('work/E_letters.npy',E); np.save('work/wn.npy',wn); del E
E=np.load('work/E_letters.npy')
h24=np.load('work/fwd_01/h24.npy',mmap_mode='r')
recs=[]
for i,q in enumerate(qids):
    ids=tok.encode(render(q)); n=len(ids); assert n==off[i+1]-off[i], (n,off[i+1]-off[i])
    k=len(qs[q]['options'])
    # find last occurrence of option sequence: search backwards for OPT[k-1], then OPT[k-2] ...
    pos=[None]*k; j=n-1
    for L in range(k-1,-1,-1):
        while ids[j]!=OPT[L]: j-=1
        pos[L]=j; j-=1
    ans_tok=[t for t in range(pos[-1],n) if tok.decode([ids[t]]).startswith('Answer')][0]
    ends=[pos[L+1]-1 for L in range(k-1)]+[ans_tok-1]
    x=np.asarray(h24[off[i+1]-1],dtype=np.float64); xn=x/np.sqrt((x**2).mean()+1e-5)*wn
    lg=(E[:k]@xn)
    recs.append(dict(q=q,k=k,gold=qs[q]['answer_index'],pred=int(lg.argmax()),lg=lg.tolist(),start=int(off[i]),pos=pos,ends=ends,n=n,cat=qs[q]['category']))
json.dump(recs,open('work/recs1.json','w'))
acc=np.mean([r['pred']==r['gold'] for r in recs]); print('parent acc on DEV160',acc)
# attention of L22 heads: mass on letter tokens, ')' tokens, option ends, other
for l in [14,16,18,20,22]:
    A=np.load(f'work/fwd_01/attn_last_L{l}.npy')
    tot={'let':0,'par':0,'end':0,'span':0}
    for r in recs:
        a=A[r['start']:r['start']+r['n']]
        p=np.array(r['pos']); e=np.array(r['ends'])
        tot['let']=tot['let']+a[p].sum(0); tot['par']=tot['par']+a[p+1].sum(0); tot['end']=tot['end']+a[e].sum(0)
        span=sum(a[p[L]:e[L]+1].sum(0) for L in range(r['k'])); tot['span']+=span
    print('L',l,' '.join(f"{nm}:"+','.join(f'{v/len(recs):.2f}' for v in tot[nm]) for nm in tot))
