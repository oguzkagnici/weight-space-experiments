# Hypothesis H2: at each attention layer, key inputs of option-line tokens carry a consistent ordinal slot direction (slot means line up; first item stands out).
import json, numpy as np, torch, re, os, sys
from wsp.editor import *
D=sys.argv[1]
sel=json.load(open('work/p2_ids.json')); Q=questions()
meta=json.load(open(D+'/meta.json')); print({k:str(v)[:300] for k,v in meta.items()})
print(sorted(os.listdir(D)))
# token -> slot labels
labs=[];kinds=[]
for q in sel:
    text=render_think(q); enc=tok(text,return_offsets_mapping=True); off=enc['offset_mapping']
    lab=np.full(len(off),-1); kind=np.zeros(len(off),int)
    starts=[text.index('\n(%s) '%c)+1 for c in 'ABCDEFGHIJ']; end=text.index('\n\nThink step by step')
    bounds=starts+[end+1]
    for t,(a,b) in enumerate(off):
        for k in range(10):
            if bounds[k]<=a<bounds[k+1]: lab[t]=k
    for k in range(10):
        idx=np.where(lab==k)[0]; kind[idx]=2; kind[idx[0]]=1; kind[idx[1]]=1 if len(idx)>1 else 2; kind[idx[-1]]=3  # 1 = '(' / letter, 2 = text, 3 = last token of the line (newline)
    labs.append(lab);kinds.append(kind)
lab=np.concatenate(labs);kind=np.concatenate(kinds); print('tokens',len(lab),'option tokens',(lab>=0).sum(),'per slot',np.bincount(lab[lab>=0]))
qid=np.concatenate([np.full(len(l),i) for i,l in enumerate(labs)])
np.save('work/p2_lab.npy',lab);np.save('work/p2_kind.npy',kind);np.save('work/p2_qid.npy',qid)
