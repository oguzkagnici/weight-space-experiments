# Tool: print a question and (parts of) a pass@4 trace in ASCII for manual first-wrong-step classification.
import json, sys
from wsp.editor import *
S='/home/user/weight-space-experiments/runs/shared/'
q=sys.argv[1]; s=int(sys.argv[2]); a=int(sys.argv[3]) if len(sys.argv)>3 else 0; b=int(sys.argv[4]) if len(sys.argv)>4 else 10**9
Q=questions()[q]
asc=lambda t:t.encode('ascii','replace').decode()
if a==0:
    print('Q:',asc(Q['question'])); print(' | '.join('(%s) %s'%('ABCDEFGHIJ'[i],asc(o)) for i,o in enumerate(Q['options'])),'| GOLD',Q['answer'])
for l in open(S+'passk_gen008champ_cap8192_det1.jsonl'):
    r=json.loads(l)
    if r['id']==q and r['sample']==s:
        t=asc(r['raw']); print('--- sample',s,'pred',r['pred'],'cut',r['cut'],'chars',len(t)); print(t[a:b])
