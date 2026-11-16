# READ GATE (pass 5 edited vs passes 2-3 parent, same texts): stem attention of the edited L22 heads at held-out judging positions and at background positions; next-token change there; forced conclusions on the late BASE texts; off-target |dlogp|.
import json, sys, numpy as np, importlib, collections
sys.path.insert(0,'code'); rl=importlib.import_module('regions_lib')
from wsp.editor import *
D=sys.argv[1] if len(sys.argv)>1 else 'work/fwd_05'; REG=rl.REG; L='ABCDEFGHIJ'
meta=json.load(open('work/p5_meta.json')); texts=json.load(open('work/p5_texts.json')); off=np.load(D+'/offsets_tok.npy'); A=np.load(D+'/attn_last_L22.npy',mmap_mode='r')
fitq=set(json.load(open('work/fit_questions.json'))); HEADS=json.load(open('work/edit_params.json'))['HEADS']; OTH=[h for h in range(16) if h not in HEADS]
E=np.zeros((len(texts),16,len(REG)))
for i,(t,mt) in enumerate(zip(texts,meta)):
    lab=rl.regions(t,mt); x=np.asarray(A[off[i]:off[i+1]],dtype=np.float64); assert len(lab)==len(x)
    for r in range(len(REG)): E[i,:,r]=x[lab==r].sum(0)
P3=np.load('work/p3b_attn_regions.npy')[:,10]; P2=np.load('work/p2_attn_regions.npy')[:,10]
Pm=np.array([(P3 if mt['src']=='p3' else P2)[mt['idx']] for mt in meta])
kind=np.array([m['kind'] for m in meta]); held=np.array([m['id'] not in fitq for m in meta]); J=np.isin(kind,['judge','judge2'])
st=lambda X:X[...,0]+X[...,1]
def line(tag,m):
    p=st(Pm[m]); e=st(E[m]); d=(e-p)[:,HEADS].mean(1)
    print('%-34s n %3d | stem attention, edited heads: parent %.3f -> edited %.3f (%+.3f, SE %.3f) | other L22 heads %.3f -> %.3f | options (edited heads) %.3f -> %.3f | trace %.3f -> %.3f'%(tag,m.sum(),p[:,HEADS].mean(),e[:,HEADS].mean(),d.mean(),d.std(ddof=1)/np.sqrt(m.sum()),p[:,OTH].mean(),e[:,OTH].mean(),Pm[m][:,HEADS,2].mean(),E[m][:,HEADS,2].mean(),Pm[m][:,HEADS,4].mean(),E[m][:,HEADS,4].mean()))
line('HELD-OUT judging positions',J&held); line('fit judging positions',J&~held); line('HELD-OUT background positions',(kind=='bg')&held); line('fit background positions',(kind=='bg')&~held); line('late BASE conclusion probes',kind=='ins')
# next-token distributions at the last position
def topk(Dd,idxs):
    ids=np.load(Dd+'/topk_ids.npy'); lp=np.load(Dd+'/topk_logprobs.npy'); return [dict(zip(ids[i].tolist(),lp[i].tolist())) for i in idxs],ids[idxs,0]
e_d,e_top=topk(D,list(range(len(meta))))
p3_d,p3_top=topk('work/fwd_03',[m['idx'] for m in meta if m['src']=='p3']); p2_d,p2_top=topk('work/fwd_02',[m['idx'] for m in meta if m['src']=='p2'])
pd=[];pt=[];i3=i2=0
for m in meta:
    if m['src']=='p3': pd.append(p3_d[i3]); pt.append(p3_top[i3]); i3+=1
    else: pd.append(p2_d[i2]); pt.append(p2_top[i2]); i2+=1
pt=np.array(pt)
def tv(a,b):
    ks=set(a)|set(b); return 0.5*sum(abs(np.exp(a.get(k,-30))-np.exp(b.get(k,-30))) for k in ks)
TV=np.array([tv(a,b) for a,b in zip(pd,e_d)])
for tag,m in (('HELD-OUT judging',J&held),('fit judging',J&~held),('HELD-OUT background',(kind=='bg')&held),('late BASE probes',kind=='ins')):
    print('%-22s next-token distribution: total variation parent vs edited mean %.4f, max %.3f; top-1 changed %d/%d'%(tag,TV[m].mean(),TV[m].max(),(pt[m]!=e_top[m]).sum(),m.sum()))
# forced conclusions (late BASE)
g=[];w=[];gt=[]
for i,m in enumerate(meta):
    if m['kind']!='ins': continue
    for dd,store in ((pd[i],0),(e_d[i],1)):
        fl=min(dd.values())-0.5; v=np.array([dd.get(int(t),fl) for t in LETTER_IDS[:10]]); 
        (g if store==0 else gt).append((v[L.index(m['gold'])]-v[L.index(m['wrong'])],L[int(v.argmax())]==m['gold'],m['id']))
dm=np.array([b_[0]-a_[0] for a_,b_ in zip(g,gt)]); q=np.array([a_[2] for a_ in g]); dq=np.array([dm[q==x].mean() for x in sorted(set(q))])
print('late BASE forced conclusions (30 wrong samples, 13 questions): margin log P(gold)-log P(wrong pick) parent %.2f -> edited %.2f, change %+.3f (SE %.3f by question); top = gold %d -> %d'%(np.mean([a_[0] for a_ in g]),np.mean([a_[0] for a_ in gt]),dq.mean(),dq.std(ddof=1)/np.sqrt(len(dq)),sum(a_[1] for a_ in g),sum(a_[1] for a_ in gt)))
resp={'mmlupro_2654','mmlupro_6110','mmlupro_6614','mmlupro_6800'}; print('   change on the 4 cue-responsive questions %+.3f, on the others %+.3f'%(dm[np.isin(q,list(resp))].mean(),dm[~np.isin(q,list(resp))].mean()))
# off-target: token log-probs over all teacher-forced tokens
le=np.load(D+'/token_logprobs.npy'); n=np.diff(off); se=np.concatenate([[0],np.cumsum(n-1)])
def parent_lp(src,Dd):
    o=np.load(Dd+'/offsets_tok.npy'); l=np.load(Dd+'/token_logprobs.npy'); nn=np.diff(o); s=np.concatenate([[0],np.cumsum(nn-1)]); return l,s
l3,s3=parent_lp('p3','work/fwd_03'); l2,s2=parent_lp('p2','work/fwd_02')
dl=[]
for i,m in enumerate(meta):
    l,s=(l3,s3) if m['src']=='p3' else (l2,s2); a=l[s[m['idx']]:s[m['idx']+1]]; bb=le[se[i]:se[i+1]]; assert len(a)==len(bb); dl.append(bb-a)
dl=np.concatenate(dl); print('off-target over %d teacher-forced tokens: mean dlogp %+.5f, mean |dlogp| %.5f, median %.6f, share > .01: %.4f, > .1: %.4f, max %.2f'%(len(dl),dl.mean(),np.abs(dl).mean(),np.median(np.abs(dl)),(np.abs(dl)>.01).mean(),(np.abs(dl)>.1).mean(),np.abs(dl).max()))
