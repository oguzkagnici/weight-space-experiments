# Hypothesis (step 1a, sharper): wrong samples never use the single decisive cue keyword during judging (after 20% of the trace); right ones do.
import json, re, collections
from wsp.editor import *
KEY = {"mmlupro_6110": r"inform|open[- ]label|aware|know (they|which)", "mmlupro_2654": r"dream", "mmlupro_6562": r"smok|copd|bronch", "mmlupro_6800": r"maxillary|transillum|purulent",
       "mmlupro_1409": r"weakest", "mmlupro_1643": r"made that day|after (the )?(trial|judgment)|did not exist|existed at", "mmlupro_1196": r"30[- ]day|another 30|extended|28 days|20 days",
       "mmlupro_1705": r"all (the )?cars|every car|all vehicles|individuali[sz]ed|stopped all", "mmlupro_6614": r"bmi|obes|weight", "mmlupro_1445": r"sale|more extensive|narrowly",
       "mmlupro_1275": r"mayor|discretion|standard", "mmlupro_1171": r"personal property|chattel", "mmlupro_6829": r"export", "mmlupro_1754": r"provided he signs|condition", "mmlupro_5510": r"odd one"}
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pkids = {r["id"] for r in pk}
allr = pk + [dict(r, sample=0) for r in bk if r["id"] not in pkids]
out = collections.defaultdict(list)
for r in allr:
    if r["id"] not in KEY or r["cut"]: continue
    t = r["raw"][len("<think>"):].split("</think>")[0].lower(); body = t[int(0.2 * len(t)):]
    n = len(re.findall(KEY[r["id"]], body)); out[r["id"]].append((r["sample"], r["correct"], r["pred"], n, round(n / len(body) * 1000, 2)))
for q, v in out.items(): print(q, sorted(v))
