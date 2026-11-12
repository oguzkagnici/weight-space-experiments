# Hypothesis: removing the content-free early-letter ramp from the consistent prior-writer neurons (low sd/|mean| at decision probes) removes the primacy push at two-finalist decisions without other effects. Edit = minimal-norm ramp removal in each selected w2 column (fraction F).
import json, numpy as np, torch, sys
from wsp.editor import *
F=float(sys.argv[1]) if len(sys.argv)>1 else 0.65
res=json.load(open('work/ramp_neurons.json'))
sel=[(l,j,m,s) for (l,j,m,s,am,asd) in res if m<-0.007 and s<0.6*abs(m)]
print('selected',[(l,j,round(m,4)) for l,j,m,s in sel], 'sum slope', round(sum(m for _,_,m,_ in sel),3))
E=load('model.embed_tokens.weight')[LETTER_IDS].numpy().astype(np.float64); g=load('model.embedding_norm.weight').numpy().astype(np.float64)
Rc=(E-E.mean(0))*g
rr=np.arange(10)-4.5
u=Rc.T@np.linalg.solve(Rc@Rc.T, rr)      # residual dir whose letter readout is the pure ramp idx-4.5
print('check readout of u', np.round(Rc@u,3)[:4], 'norm u', np.linalg.norm(u))
rs=rr/(rr@rr)
bylayer={}
for l,j,m,s in sel: bylayer.setdefault(l,[]).append(j)
for l,js in sorted(bylayer.items()):
    name=f'model.layers.{l}.feed_forward.w2.weight'
    w2=load_original(name).numpy().astype(np.float64); pg=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy().astype(np.float64)
    Rp=Rc*pg[None,:]
    blk=np.abs(w2).reshape(2048,-1,32)
    for j in js:
        col=w2[:,j]; slope=rs@(Rp@col)
        bmax=np.abs(np.delete(w2.reshape(2048,-1,32)[:,j//32,:],j%32,axis=1)).max(1)
        room=np.maximum(0.98*bmax-np.abs(col),0)+1e-4   # headroom per entry before exceeding block absmax
        D=room**2
        up=D*(Rp.T@(np.linalg.pinv((Rp*D[None,:])@Rp.T)@rr))
        dcol=-F*slope*up
        new=np.clip(col+dcol,-0.98*np.maximum(bmax,np.abs(col)),0.98*np.maximum(bmax,np.abs(col)))
        nclip=int((np.abs(col+dcol)>0.98*np.maximum(bmax,np.abs(col))).sum())
        print('  clipped',nclip,'residual slope frac', (rs@(Rp@new))/slope)
        dcol=new-col
        print(f'L{l} n{j} col slope {slope:.4f} |col| {np.linalg.norm(col):.3f} |dcol| {np.linalg.norm(dcol):.4f} max|dcol|/max|col| {np.abs(dcol).max()/np.abs(col).max():.3f}')
        w2[:,j]=col+dcol
    save_edit(name, torch.tensor(w2,dtype=torch.float32))
print(edited())
