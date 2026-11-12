# Hypothesis: at a two-finalist statement (where the decision is ~coin-flip, code/004), the model's latent preference
# (forced answer "So the answer is (" right after the statement) favours gold more often than the realised pick does.
# If so, "commit now and follow your own latent preference" would be a content gain. Texts: prefix up to the end of the
# sentence holding the LAST finalist pair (in the 2nd half), + "\n\nSo the answer is (" ; prefix <= 4,080 tokens.
import json, re
from wsp.editor import *
src=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
seen={(r['id'],r['sample']) for r in src}
src+= [r for r in map(json.loads,open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl')) if (r['id'],r.get('sample',0)) not in seen]
pat=re.compile(r"(?:between|either)\s*\(?([A-J])\)?\s*(?:and|or)\s*\(?([A-J])\)?|\(?\b([A-J])\)?\s*(?:or|vs\.?|versus)\s*\(?([A-J])\b\)?")
SUF="\n\nSo the answer is ("
items=[]; texts=[]
for r in src:
    t=r['think']
    ms=[m for m in pat.finditer(t) if m.start()>len(t)*0.3]
    ms=[m for m in ms if len({g for g in m.groups() if g})==2]
    if not ms: continue
    m=ms[-1]; pair=sorted({g for g in m.groups() if g})
    end=t.find("\n",m.end()); end=len(t) if end<0 else end
    prefix_think=t[:end]
    # trace_text gives prompt + generated text; rebuild with the truncated thinking
    full=trace_text(r); gen_part=r['raw']
    cut_at=full.find(prefix_think[len("<think>"):][:200]) if prefix_think.startswith("<think>") else full.find(prefix_think[:200])
    pt=render_think(r['id'])
    body=prefix_think[len("<think>"):] if prefix_think.startswith("<think>") else prefix_think
    text=pt+body+SUF
    n=len(tok(text,add_special_tokens=False).input_ids)
    if n>4090: continue
    items.append({'id':r['id'],'sample':r.get('sample',0),'gold':r['gold'],'pred':r['pred'],'cut':r['cut'],'correct':r['correct'],
                  'pair':pair,'frac':round(m.start()/len(t),2),'ntok':n})
    texts.append(text)
print(len(texts), sum(i['ntok'] for i in items), 'gold in pair', sum(i['gold'] in i['pair'] for i in items),
      'finished', sum(not i['cut'] for i in items))
json.dump(items,open('work/probe2_index.json','w'))
json.dump({"type":"read","model":"original","texts":texts,"return":{"token_logprobs":False,"topk":50,"positions":"last"}},open('work/req_read2.json','w'))
