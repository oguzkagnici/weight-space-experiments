# Hypothesis: inspection helper - print question stem, options and a slice of a trace in ASCII.
import json, sys, importlib
sys.path.insert(0,'code'); m=importlib.import_module('001_overview')
from wsp.editor import *
Q=questions(); rows=m.load_rows()
qid=sys.argv[1]; s=int(sys.argv[2]); a=int(sys.argv[3]); b=int(sys.argv[4])
asc=lambda t:t.encode('ascii','replace').decode()
q=Q[qid]
if a==0:
    print('STEM:',asc(q['question'])); print('OPTS:',[ 'ABCDEFGHIJ'[i]+': '+asc(o) for i,o in enumerate(q['options'])],'gold',q['answer'])
r=[r for r in rows if r['id']==qid and r['sample']==s][0]
print('pred',r['pred'],'correct',r['correct'],'tokens',r['tokens'],'chars',len(r['raw']))
print(asc(r['raw'][a:b]))
