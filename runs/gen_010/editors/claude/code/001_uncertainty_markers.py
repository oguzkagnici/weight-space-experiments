# Hypothesis: wrong finished answers concentrate in traces where the model says it lacks knowledge and guesses; right/wrong
# among mixed questions then depends on which guessing heuristic is sampled.
import json, collections, re
rows=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl')]
fin=[r for r in rows if not r['cut']]
marks={'not_familiar':r"not (intimately |very |entirely )?familiar|don't (recall|remember)|do not recall|not sure what",
       'guess':r"\bguess\b|best guess|educated guess",
       'safe':r"safe (answer|choice|general)|most general|broad(est)? (statement|answer)",
       'unsure':r"not (entirely |completely )?sure|uncertain|hard to tell|ambiguous"}
for k,p in marks.items():
    has=[r for r in fin if re.search(p,r['think'],re.I)]
    no=[r for r in fin if not re.search(p,r['think'],re.I)]
    print(f"{k:13s} with: n {len(has):3d} acc {100*sum(r['correct'] for r in has)/max(1,len(has)):5.1f} | without: n {len(no):3d} acc {100*sum(r['correct'] for r in no)/max(1,len(no)):5.1f}")
