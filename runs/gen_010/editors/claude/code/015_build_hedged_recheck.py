# Build the hedged-commitment re-check unit on L23 n881 (freed by folding into its near-duplicate n331).
# Gate (w1 row) = K * (w_lda - T * p): w_lda = shrinkage-LDA direction (hedged-commitment paragraph breaks vs firm breaks +
# other tokens, all 52 pass-4 texts), p = minimum-variance bias proxy (mean(x.p)=1 on generated tokens), T = 60th percentile
# of hedged-break scores (fires on ~40%). Gate projected off the prompt-sink inputs (first 4 prompt tokens).
# Up (w3 row) = ALPHA * p.  Write (w2 col) = g_final*(mean E[reopen] - mean E[wrap])/g_post, clipped to 0.98 x Q8 block absmax.
import json, sys, numpy as np, torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from wsp.editor import *
K=float(sys.argv[1]) if len(sys.argv)>1 else 1.0; ALPHA=float(sys.argv[2]) if len(sys.argv)>2 else 2.0; PCT=60
items=json.load(open('work/read4_index.json')); meta=json.load(open('work/fwd_04/meta.json'))
X=np.load('work/fwd_04/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); offs=np.cumsum([0]+meta['n_tokens'])
rng=np.random.default_rng(0); rows=[];lab=[];gen_rows=[];sink=[]
for t,(it,o) in enumerate(zip(items,offs[:-1])):
    mk={ti:h for ti,h,_ in it['marks']}
    for ti,h in mk.items(): rows.append(o+ti); lab.append(2 if h else 1)
    g=np.arange(o+it['plen'],o+it['n']); g=g[~np.isin(g,[o+k for k in mk])]
    s=rng.choice(g,size=min(400,len(g)),replace=False); rows+=list(s); lab+=[0]*len(s); gen_rows+=list(s)
    sink.append([o+i for i in range(4)])
A=np.asarray(X[np.array(rows)],np.float32); lab=np.array(lab)
m=LDA(solver='lsqr',shrinkage='auto').fit(A,(lab==2).astype(int)); w=m.coef_[0].astype(np.float64)
G=np.asarray(X[np.array(gen_rows)],np.float64); mu=G.mean(0)
C=np.cov(G.T)+1e-3*np.eye(G.shape[1])*np.trace(np.cov(G.T))/G.shape[1]
p=np.linalg.solve(C,mu); p/= (p@mu)
print('bias proxy: mean',round(float((G@p).mean()),3),'sd',round(float((G@p).std()),3))
# remove prompt-sink subspace from w (and p) so the first prompt tokens never fire
S=np.asarray(X[np.array(sink).ravel()],np.float64); Q,_=np.linalg.qr(S.T); w=w-Q@(Q.T@w)
sh=A[lab==2]@w; T=np.percentile(sh,PCT)
pre=lambda Z: K*(Z@w - T*(Z@p))
print('pre-act percentiles: hedged',np.percentile(pre(A[lab==2]),[10,50,90]).round(2),'firm',np.percentile(pre(A[lab==1]),[50,90,99]).round(2),
      'other',np.percentile(pre(A[lab==0]),[50,99.9]).round(2),'sink max',round(float(pre(S).max()),2))
print('fires(>0): hedged',round(float((pre(A[lab==2])>0).mean()),2),'firm',round(float((pre(A[lab==1])>0).mean()),3),'other',round(float((pre(A[lab==0])>0).mean()),4))
w1=load('model.layers.23.feed_forward.w1.weight'); w3=load('model.layers.23.feed_forward.w3.weight'); w2=load('model.layers.23.feed_forward.w2.weight')
r=float(w1[881]@w1[331]/(w1[331]@w1[331]))*float(w3[881]@w3[331]/(w3[331]@w3[331]))
w2[:,331]+=r*w2[:,881]
# gate row: w1 row acts on the same normalized input x; pre-activation = K*(w.x - T p.x)
w1[881]=torch.tensor(K*(w-T*p),dtype=torch.float32); w3[881]=torch.tensor(ALPHA*p,dtype=torch.float32)
E=load_base('model.embed_tokens.weight') if 'model.embed_tokens.weight' in NAMES else None
gf=load('model.embedding_norm.weight').double(); gp=load('model.layers.23.post_ffn_norm.weight').double()
RE=[3911,37967,7731,91780]; WR=[21009,43129,124902,25705,4597]
v=(E[RE].double().mean(0)-E[WR].double().mean(0))*gf/gp
col=torch.zeros(w2.shape[0],dtype=torch.float64)
# clip per Q8_0 block of 32 along each w2 row (row = output dim, blocks over the 7,680 inputs): block containing column 881
b0=(881//32)*32; blk=w2[:,b0:b0+32].clone(); blk[:,881-b0]=0; amax=blk.abs().max(1).values.double()
scale=float(torch.quantile(0.98*amax/ v.abs().clamp_min(1e-9),0.10)); col=(v*scale)   # 90% of entries within the cap, rest clipped
col=torch.maximum(torch.minimum(col,0.98*amax),-0.98*amax)
w2[:,881]=col.float()
print('fold ratio',round(r,4),'w2 col norm',round(float(col.norm()),3),'scale',round(scale,4),'old col norm',round(float(load_original('model.layers.23.feed_forward.w2.weight')[:,881].norm()),3))
for n,tn in (('model.layers.23.feed_forward.w1.weight',w1),('model.layers.23.feed_forward.w3.weight',w3),('model.layers.23.feed_forward.w2.weight',w2)): save_edit(n,tn)
json.dump({'K':K,'ALPHA':ALPHA,'PCT':PCT,'T':float(T)},open('work/edit_params.json','w'))
