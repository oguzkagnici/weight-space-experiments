# Hypothesis: the early-letter preference is introduced after L22.H12's selection by identifiable components (L22 MLP, L23); decompose final letter logits by component on 512 DEV.
from wsp.editor import *
import numpy as np, json
D='work/fwd_03/score_3_acts/original/'
rows={json.loads(l)['id']:json.loads(l) for l in open('work/fwd_03/score_3_original.jsonl')}
ids=json.load(open(D+'index.json'))['ids']; qs=questions()
k=np.array([len(qs[i]['options']) for i in ids]); gold=np.array([qs[i]['answer_index'] for i in ids])
E=np.load('work/E_letters.npy').astype(np.float64); wn=np.load('work/wn.npy').astype(np.float64)
H={l:np.load(D+f'h{l}.npy').astype(np.float64) for l in [18,20,21,22,23,24]}
rms24=np.sqrt((H[24]**2).mean(1)+1e-5)
def lens(x,scale=None):
    s=np.sqrt((x**2).mean(1)+1e-5) if scale is None else scale
    return ((x/s[:,None])*wn)@E.T
def post(o,w): return o/np.sqrt((o**2).mean(1,keepdims=True)+1e-5)*w
Wo=load('model.layers.22.self_attn.o_proj.weight').double().numpy(); Z=np.load(D+'in_model.layers.22.self_attn.o_proj.npy').astype(np.float64)
O=Z@Wo.T; ro=np.sqrt((O**2).mean(1,keepdims=True)+1e-5); wpo=load('model.layers.22.post_operator_norm.weight').double().numpy()
att=O/ro*wpo
heads={h:(Z[:,h*128:(h+1)*128]@Wo[:,h*128:(h+1)*128].T)/ro*wpo for h in range(16)}
comp={}
for l in [22,23]:
    W2=load(f'model.layers.{l}.feed_forward.w2.weight').double().numpy(); M=np.load(D+f'mlp{l}.npy').astype(np.float64)@W2.T
    comp[f'mlp{l}']=post(M,load(f'model.layers.{l}.post_ffn_norm.weight').double().numpy())
print('recon L22 err', np.abs(H[22]+att+comp['mlp22']-H[23]).mean()/np.abs(H[23]-H[22]).mean())
comp['conv23']=H[24]-H[23]-comp['mlp23']
fin=lens(H[24]); pred=np.array([np.argmax(fin[i,:k[i]]) for i in range(len(ids))])
print('acc check', np.mean(pred==gold), np.mean([rows[i]['correct'] for i in ids]))
m10=k==10
def cen(L): L=L[m10]; return L-L.mean(1,keepdims=True)
def show(name,L): c=cen(L); print(f'{name:12s}', ' '.join(f'{v:+.2f}' for v in c.mean(0)), f'| sd {c.std(0).mean():.2f}')
print(' '*12,'    '.join('ABCDEFGHIJ'))
for l in [18,20,21,22,23]: show(f'lens h{l}',lens(H[l]))
show('final',fin)
dl=lambda x: (x/rms24[:,None]*wn)@E.T
show('attn22',dl(att)); show('H12',dl(heads[12])); show('att22-H12',dl(att-heads[12]))
show('mlp22',dl(comp['mlp22'])); show('conv23',dl(comp['conv23'])); show('mlp23',dl(comp['mlp23'])); show('pre22(h22)',dl(H[22]))
# accuracy of partial sums (10-opt)
def acc(L): return np.mean([np.argmax(L[i,:k[i]])==gold[i] for i in range(len(ids))])
parts=[('h22',dl(H[22])),('+attn22',dl(H[22]+att)),('+mlp22',dl(H[23])),('+conv23',dl(H[23]+comp['conv23'])),('+mlp23 (final)',fin),('H12 alone',dl(heads[12]))]
print(' '.join(f'{n}: {acc(L):.3f}' for n,L in parts))
# offline debias ceiling: subtract 2-fold CV constant 10-opt bias from final
c=fin.copy(); I=np.where(m10)[0]; half=I[::2],I[1::2]
for a,b in [(0,1),(1,0)]:
    bb=fin[half[a]][:, :10]; bb=(bb-bb.mean(1,keepdims=True)).mean(0)
    c[half[b],:10]-=bb
print('CV debiased final acc', acc(c), 'raw', acc(fin))
np.save('work/p3_heads_dl.npy',np.stack([dl(heads[h]) for h in range(16)])); np.save('work/p3_comp_dl.npy',np.stack([dl(H[22]),dl(att),dl(comp['mlp22']),dl(comp['conv23']),dl(comp['mlp23'])]))
