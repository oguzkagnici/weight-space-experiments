# Hypothesis: in edited generations the gated mechanism is visible: fewer paragraph-initial churn openers per 1k tokens and more paragraph-initial Hence/Therefore (rare in the original's cut traces), even though traces diverge early for unrelated reasons.
import json, sys, glob, re, numpy as np
K=sys.argv[1]; D=f'work/fwd_{K}'
O={json.loads(l)['id']:json.loads(l) for l in open(glob.glob(f'{D}/gen_*_original.jsonl')[0])}
E={json.loads(l)['id']:json.loads(l) for l in open(glob.glob(f'{D}/gen_*_edited.jsonl')[0])}
CH=re.compile(r'\n\s*(Alternatively|Maybe|Perhaps|Another|Possibly|But perhaps|But maybe)\b')
HT=re.compile(r'\n\s*(Hence|Therefore)\b')
for name,S in (('original',O),('edited',E)):
    ch=sum(len(CH.findall(r['think'])) for r in S.values()); ht=sum(len(HT.findall(r['think'])) for r in S.values()); tk=sum(r['tokens'] for r in S.values())
    print(f'{name}: churn openers {ch} ({1000*ch/tk:.2f}/1k), Hence/Therefore openers {ht} ({1000*ht/tk:.2f}/1k), tokens {tk}')
for q in E:
    e=E[q]; o=O[q]
    if o['cut'] and not e['cut']:
        t=e['think']; m=[x.start() for x in HT.finditer(t)]
        print(q,'newly finished',e['pred'],'gold',e['gold'],'tokens',e['tokens'],'Hence/Therefore openers at',[round(x/len(t),2) for x in m])
        for x in m[:2]:
            print('     ', t[max(0,x-150):x+120].replace('\n','/').encode('ascii','replace').decode())
