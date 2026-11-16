# Do edited traces contain more re-check openers than the parent's (paired by question), and what happened on the questions whose outcome changed?
import json, sys, re, numpy as np
bank=sys.argv[1]; files=sys.argv[2:]
B={(r['id'],r.get('sample',0)):r for r in map(json.loads,open(bank))}
E={}
for f in files:
    for l in open(f): r=json.loads(l); E[(r['id'],r.get('sample',0))]=r
pat=re.compile(r'(?:\n\n|[.?!] )(Wait|Actually|Hmm|Hold on|But wait|However)\b')
keys=[k for k in E if k in B]
rb=np.array([len(pat.findall(B[k]['raw'])) for k in keys]); re_=np.array([len(pat.findall(E[k]['raw'])) for k in keys])
tb=np.array([B[k]['tokens'] for k in keys]); te=np.array([E[k]['tokens'] for k in keys])
print('n %d | re-check openers per trace parent %.1f edited %.1f | per 1,000 tokens parent %.2f edited %.2f | tokens %.0f -> %.0f (paired diff %+.0f, SE %.0f)'%(len(keys),rb.mean(),re_.mean(),1000*rb.sum()/tb.sum(),1000*re_.sum()/te.sum(),tb.mean(),te.mean(),(te-tb).mean(),(te-tb).std(ddof=1)/np.sqrt(len(keys))))
for k in keys:
    b,e=B[k],E[k]
    if b['correct']!=e['correct'] or (b['pred']!=e['pred'] and not b['cut'] and not e['cut']):
        a,c=b['raw'],e['raw']; n=next((i for i,(x,y) in enumerate(zip(a,c)) if x!=y),min(len(a),len(c)))
        print(k,b['category'],'gold',b['gold'],'parent',b['pred'],'cut' if b['cut'] else '',b['tokens'],'-> edited',e['pred'],'cut' if e['cut'] else '',e['tokens'],'| first diff at %.0f%%: parent %r | edited %r'%(100*n/len(a),a[n-30:n+40].encode('ascii','replace').decode(),c[n:n+40].encode('ascii','replace').decode()))
