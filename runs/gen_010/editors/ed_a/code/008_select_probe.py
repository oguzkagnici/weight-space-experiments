# Hypothesis: (same as 007) select probe texts within the 200k-token read limit, prioritising pairs that contain gold.
import json
items=json.load(open('work/probe_items_all.json'))
ok=[i for i in items if i['ntok']<=4096]
g=[i for i in ok if i['gold'] in (i['a'],i['b'])]; o=[i for i in ok if i['gold'] not in (i['a'],i['b'])]
sel=[]; tot=0
for i in sorted(g,key=lambda i:i['ntok'])+sorted(o,key=lambda i:i['ntok']):
    if tot+i['ntok']<=195000: sel.append(i); tot+=i['ntok']
print(len(ok), len(g), 'selected', len(sel), 'gold-in-pair', sum(i['gold'] in (i['a'],i['b']) for i in sel), 'tokens', tot)
json.dump(sel, open('work/probe_sel.json','w'))
req={"type":"read","model":"original","texts":[i['text'] for i in sel],
     "return":{"topk":50,"positions":"last","hidden_states":list(range(14,25)),"mlp":list(range(14,24))}}
json.dump(req, open('work/req_p1_read.json','w'))
