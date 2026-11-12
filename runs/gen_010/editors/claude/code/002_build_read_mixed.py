# Build read-pass texts: all FINISHED samples of the 15 mixed questions (right and wrong conclusions of the same question),
# to get per-token log-probs/top-k of the parent along each trace and locate low-probability decisive choices.
import json, collections
from wsp.editor import *
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
by=collections.defaultdict(list)
for r in rows: by[r['id']].append(r)
mixed=[q for q,v in by.items() if 0<sum(r['correct'] for r in v)<4]
sel=[r for q in mixed for r in by[q] if not r['cut']]
texts=[trace_text(r) for r in sel]
lens=[len(tok(t,add_special_tokens=False).input_ids) for t in texts]
print(len(texts), sum(lens), max(lens))
json.dump([{'id':r['id'],'sample':r['sample'],'correct':r['correct'],'pred':r['pred'],'gold':r['gold'],'ntok':n} for r,n in zip(sel,lens)],open('work/read1_index.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"token_logprobs":True,"topk":5}},open('work/req_read1.json','w'))
