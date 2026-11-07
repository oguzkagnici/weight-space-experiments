# Hypothesis: at the L23 MLP input, the current token being '"The' (id 14225) is linearly readable: a PC-cleaned mean-diff
# direction separates all '"The' positions (thinking lines '"The answer is (X)"' and the forced answer start) from every other
# prompt/thinking token. Misfires on thinking '"The' tokens are harmless because ' answer' follows there anyway.
import json, numpy as np
from wsp.editor import tok
X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); off1=np.load('work/fwd_01/offsets.npy')
cp1=json.load(open('work/p1_closepos.json')); t1=json.load(open('work/req_p1.json'))['texts']
R2=np.load('work/p2_L23_content_rows.npy').astype(np.float32); rows2=json.load(open('work/p2_L23_content_rows.json'))
meta2=json.load(open('work/p2_meta.json')); cp2={int(k):v for k,v in json.load(open('work/p2_closepos.json')).items()}; off2=np.load('work/fwd_02/offsets.npy')
# O2 p1 rows (the '"The' token right after </think>)
o2p1=[r for r in range(len(rows2)) if any(rows2[r]==off2[i]+cp2[i]+1 for i,m in enumerate(meta2) if m['open']=='O2')]
T=R2[o2p1]; print('content p1 samples',len(T))
ids=[tok(t,add_special_tokens=False)['input_ids'] for t in t1[:13]]
q=[(j,i) for j in range(13) for i in range(cp1[j]) if ids[j][i]==14225]
print('thinking/prompt "The tokens:',len(q),' next tokens:',sorted(set(tok.decode([ids[j][i+1]]) for j,i in q)))
Kall=np.concatenate([np.arange(off1[j],off1[j]+cp1[j]) for j in range(13)])
qset=set(off1[j]+i for j,i in q)
Kneg=np.array([k for k in Kall if k not in qset])
for split in ((range(0,13,2),range(1,13,2)),(range(1,13,2),range(0,13,2))):
    tr=np.concatenate([np.arange(off1[j],off1[j]+cp1[j]) for j in split[0]]); te=[j for j in split[1]]
    Xk=np.asarray(X1[tr],dtype=np.float32); mu=Xk.mean(0); ev,U=np.linalg.eigh(np.cov((Xk-mu).T)); P=U[:,::-1][:,:200]
    cl=lambda v:v-P@(P.T@v)
    pos_tr=np.asarray(X1[[off1[j]+i for j,i in q if j in split[0]]],dtype=np.float32)
    Ttr=T[[k for k in range(len(T)) if k%2==0]]; Tte=T[[k for k in range(len(T)) if k%2==1]]
    d=cl(np.concatenate([pos_tr,Ttr]).mean(0)-mu)
    tei=np.concatenate([np.arange(off1[j],off1[j]+cp1[j]) for j in te]); tei=np.array([k for k in tei if k not in qset])
    s=np.asarray(X1[tei],dtype=np.float32)@d; m,sd=s.mean(),s.std()
    pos_te=np.asarray(X1[[off1[j]+i for j,i in q if j in te]],dtype=np.float32)@d
    print(f'held-out: other thinking max z {(s.max()-m)/sd:.1f} p99.99 {(np.percentile(s,99.99)-m)/sd:.1f} | thinking "The min z {((pos_te.min()-m)/sd) if len(pos_te) else float("nan"):.1f} | content p1 (held-out) min z {((Tte@d).min()-m)/sd:.1f} median {(np.median(Tte@d)-m)/sd:.1f}')
