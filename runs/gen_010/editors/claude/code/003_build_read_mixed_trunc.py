# Same as 002 but each text truncated to its first 4,090 tokens (read limit 4,096/text). Per-token log-probs + top-10 of the
# parent along right and wrong samples of mixed questions: are commitment letters/verdicts low-probability sampled choices?
import json
from wsp.editor import *
idx=json.load(open('work/read1_index.json'))
rows={(r['id'],r['sample']):r for r in map(json.loads,open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl'))}
texts=[]
for m in idx:
    ids=tok(trace_text(rows[(m['id'],m['sample'])]),add_special_tokens=False).input_ids
    m['trunc']=len(ids)>4090; ids=ids[:4090]; m['ntok_read']=len(ids)
    texts.append(tok.decode(ids))
lens=[len(tok(t,add_special_tokens=False).input_ids) for t in texts]
print(len(texts), sum(lens), max(lens), sum(m['trunc'] for m in idx))
json.dump(idx,open('work/read1_index.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"token_logprobs":True,"topk":10}},open('work/req_read1.json','w'))
