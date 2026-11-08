# Pass-3/4 requests: generate with edited v1 on the 96 bank questions (two halves of 48, random split seed 8); paired vs the bank.
import json, random
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
ids=[b['id'] for b in B]; random.Random(8).shuffle(ids)
A,Bh=ids[:48],ids[48:]
json.dump({"A":A,"B":Bh},open('work/bank_split.json','w'))
for nm,s in (('p3',A),('p4',Bh)):
    json.dump({"type":"generate","model":"edited","questions":s},open(f'work/req_{nm}.json','w'))
byid={b['id']:b for b in B}
for nm,s in (('A',A),('B',Bh)): print(nm,'cut',sum(byid[i]['cut'] for i in s),'correct',sum(byid[i]['correct'] for i in s))
