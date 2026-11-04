# Hypothesis: L22 MLP's direct letter write (leader suppression) costs accuracy; removing part of it (lens screen, no downstream re-computation) raises accuracy.
import numpy as np, json
from wsp.editor import questions
rows=[json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')]; qs=questions()
k=np.array([len(qs[r['id']]['options']) for r in rows]); g=np.array([r['answer'] for r in rows])
C=np.load('work/p3_comp_dl.npy'); Hd=np.load('work/p3_heads_dl.npy')
fin=C[0]+C[1]+C[2]+C[3]+C[4]
names=['h22','attn22','mlp22','conv23','mlp23']
def ev(L):
    a=[];lp=[]
    for i in range(len(g)):
        x=L[i,:k[i]]; a.append(np.argmax(x)==g[i]); x=x-x.max(); lp.append(x[g[i]]-np.log(np.exp(x).sum()))
    return np.mean(a),np.mean(lp)
print('final', ev(fin))
for j in [2,3,4]:
    print(names[j],' '.join(f'a{al}: %.3f/%.2f'%ev(fin-(1-al)*C[j]) for al in [0,0.25,0.5,0.75,1.25,1.5]))
oth=C[1]-Hd[12]
print('attn22 non-H12',' '.join(f'a{al}: %.3f/%.2f'%ev(fin-(1-al)*oth) for al in [0,0.5,1.5]))
for h in [11,13,14,15,0,1,8,10]:
    print(f'H{h} remove: %.3f/%.2f  x1.5: %.3f/%.2f'%(ev(fin-Hd[h])+ev(fin+0.5*Hd[h])), 'gold-mean dla %.2f'%np.mean([Hd[h][i,g[i]]-Hd[h][i,:k[i]].mean() for i in range(len(g))]))
