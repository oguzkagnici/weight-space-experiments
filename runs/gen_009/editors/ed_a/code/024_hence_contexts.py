# Hypothesis: in traces still cut, the inserted "Hence answer ..." paragraph hedges ("likely either", "might be") or is followed by a reopen; show what follows.
import json, sys, glob, re
E=[json.loads(l) for f in sys.argv[1].split(',') for l in open(glob.glob(f+'/gen_*_edited.jsonl')[0])]
for e in E:
    t=e['think']
    for m in re.finditer(r'\n\nHence answer',t):
        nxt=t[m.end():m.end()+160].replace('\n','/')
        print('%s cut%d corr%d gold %s @%.2f | %s'%(e['id'][8:],e['cut'],e['correct'],e['gold'],m.start()/len(t),nxt.encode('ascii','replace').decode()))
