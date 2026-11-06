# Hypothesis: some divergences are mechanism-driven (edited text writes Hence/Therefore at a paragraph break) and others are butterfly flips from tiny logit changes at unrelated positions.
import json, sys, glob
K=sys.argv[1]; D=f'work/fwd_{K}'
O={json.loads(l)['id']:json.loads(l) for l in open(glob.glob(f'{D}/gen_*_original.jsonl')[0])}
E={json.loads(l)['id']:json.loads(l) for l in open(glob.glob(f'{D}/gen_*_edited.jsonl')[0])}
for q in O:
    a,b=O[q]['raw'],E[q]['raw']
    if a==b: print(q,'IDENTICAL'); continue
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
    j=i
    while j>0 and not a[j-1].isspace() and a[j-1] not in '.,;:': j-=1
    f=lambda s: s.replace('\n','/').encode('ascii','replace').decode()
    print(q, 'prev:',repr(f(a[max(0,j-25):j])),'| orig:',repr(f(a[j:j+30])),'| edit:',repr(f(b[j:j+30])), '| cutO',O[q]['cut'],'cutE',E[q]['cut'],'okO',O[q]['correct'],'okE',E[q]['correct'])
