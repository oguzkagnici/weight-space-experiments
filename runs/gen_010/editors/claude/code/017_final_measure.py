# FINAL MEASUREMENT (holdout48 at 8,192; read only now, after the edit was frozen) + DEV (passk, 6 hedged questions x 4).
# Paired: overall diff (cut = wrong) and both-finish diff, SE, +/-; cut share; tokens; first differing character.
import json, glob, numpy as np, os
def load(p): return [json.loads(l) for l in open(p)]
def paired(par, ed, label):
    P={(r['id'],r.get('sample',0)):r for r in par}
    pr=[(P[(r['id'],r.get('sample',0))],r) for r in ed if (r['id'],r.get('sample',0)) in P]
    d=np.array([b['correct']-a['correct'] for a,b in pr],float); n=len(d)
    bf=[(a,b) for a,b in pr if not a['cut'] and not b['cut']]; db=np.array([b['correct']-a['correct'] for a,b in bf],float)
    se=lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else float('nan')
    fd=[next((i for i,(x,y) in enumerate(zip(a['raw'],b['raw'])) if x!=y),None) for a,b in pr]
    rel=[f/len(a['raw']) for f,(a,b) in zip(fd,pr) if f is not None]
    print(f"{label}: n {n}; acc {100*np.mean([a['correct'] for a,_ in pr]):.1f} -> {100*np.mean([b['correct'] for _,b in pr]):.1f}; "
          f"diff {100*d.mean():+.2f} (SE {100*se(d):.2f}), +{int((d>0).sum())}/-{int((d<0).sum())}; "
          f"cut {sum(a['cut'] for a,_ in pr)} -> {sum(b['cut'] for _,b in pr)}; tokens {np.mean([a['tokens'] for a,_ in pr]):.0f} -> {np.mean([b['tokens'] for _,b in pr]):.0f}")
    print(f"   both-finish n {len(db)}: diff {100*db.mean():+.2f} (SE {100*se(db):.2f}), +{int((db>0).sum())}/-{int((db<0).sum())}; "
          f"identical traces {sum(f is None for f in fd)}; median first difference at {100*np.median(rel):.0f}% of the parent trace" if len(db) else "")
    return pr
ed=[]
for k in (6,7):
    p=f'work/fwd_{k:02d}/gen_{k}_edited.jsonl'
    if os.path.exists(p): ed+=load(p)
if ed: paired(load('/home/user/weight-space-experiments/runs/shared/holdout48_bank_gen008champ_cap8192_det1.jsonl'), ed, 'HOLDOUT48 @8192')
p='work/fwd_08/gen_8_edited.jsonl'
if os.path.exists(p): paired(load('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl'), load(p), 'DEV (6 hedged passk q x4) @8192')
