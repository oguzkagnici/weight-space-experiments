# Hypothesis: the post-</think> content only restates the thinking's conclusion (same letter), so its ~250 tokens are redundant; 5 cut traces closed the think but were cut in the content.
import json, re, numpy as np, collections
from wsp.editor import tok
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
pat=re.compile(r'answer is \(?\*?\*?([A-J])\)?')
same=0;diff=[];starts=collections.Counter()
for r in B:
    if not r['finished_think']: continue
    tl=len(tok(r['think'])['input_ids']); cl=len(tok(r['content'])['input_ids'])
    m=pat.findall(r['think'])
    lastthink=m[-1] if m else None
    if not r['cut']:
        if lastthink==r['pred']: same+=1
        else: diff.append((r['id'],lastthink,r['pred'],r['gold']))
    starts[r['content'].strip()[:25]]+=1
    if r['cut'] or tl>2500: print(r['id'],'cut',r['cut'],'think_tok',tl,'content_tok',cl,'lastthink',lastthink,'gold',r['gold'],'pred',r['pred'])
print('finished: content answer == last think statement',same,'differs',diff)
print(starts.most_common(12))
# finished-trace think token distribution
tt=[len(tok(r['think'])['input_ids']) for r in B if not r['cut']]
print('finished think tokens pct',np.percentile(tt,[10,50,90,100]).round())
# last 300 chars of the think in finished traces: how does it end
for r in B[:6]:
    if not r['cut']: print(repr(r['think'][-160:]))
