# Hypothesis: (DEV set) 5 closed-but-cut bank traces are the mechanism's targets; 27 other bank questions (finished and cut) measure off-target byte-identity.
import json, numpy as np
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
tg=['mmlupro_1740','mmlupro_9274','mmlupro_8198','mmlupro_6829','mmlupro_6014']
rng=np.random.default_rng(11)
fin=[r['id'] for r in B if not r['cut']]; cut=[r['id'] for r in B if r['cut'] and r['id'] not in tg]
rng.shuffle(fin); rng.shuffle(cut)
dev=tg+fin[:14]+cut[:13]
json.dump({'dev':dev},open('work/dev_set.json','w'))
json.dump({'type':'generate','model':'both','questions':dev},open('work/req4.json','w'))
print(len(dev))
