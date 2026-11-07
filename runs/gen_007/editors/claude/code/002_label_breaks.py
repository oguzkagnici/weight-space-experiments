# Hypothesis (needed for the gate): paragraph breaks that END a two-finalist paragraph ("either F or G", "between C and H")
# can be located token-exactly in the parent's bank traces. Labels every paragraph-break token (any token whose text
# contains "\n\n") of the chosen traces: 1 = break right after a two-finalist statement, 0 = other break.
# Also labels churn breaks (the next paragraph opens with Alternatively/Maybe/Perhaps/Another/Possibly).
import json, re
from wsp.editor import *
X = r"\(?([A-J])\)?"
pair = re.compile(r"\b(?:either|between)\s+(?:option\s+|options\s+)?" + X + r"\s*(?:or|and|vs\.?)\s+(?:option\s+)?" + X + r"\b"
                  r"|\b(?:answer|choice|options?)\s+(?:is|are|could be|might be|seems|likely)?\s*(?:likely\s+)?" + X + r"\s+or\s+" + X + r"\b"
                  r"|\b" + X + r"\s+(?:or|vs\.?)\s+" + X + r"\s*[\?\.]", re.I)
CHURN = ("Alternatively", "Maybe", "Perhaps", "Another", "Possibly")
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl")]
def two_finalist_ends(think):
    ends = set()
    for m in pair.finditer(think):
        if len({x for x in m.groups() if x}) == 2:
            e = think.find("\n\n", m.end())
            if e >= 0: ends.add(e)
    return ends
narrow = [r for r in rows if r["cut"] and two_finalist_ends(r["raw"].replace("<think>", "", 1).split("</think>")[0])]
fin_narrow = [r for r in rows if not r["cut"] and two_finalist_ends(r["raw"].replace("<think>", "", 1).split("</think>")[0])]
others = [r for r in rows if r not in narrow and r not in fin_narrow]
print("cut narrowing", len(narrow), "finished narrowing", len(fin_narrow), "others", len(others))
json.dump({"narrow": [r["id"] for r in narrow], "fin_narrow": [r["id"] for r in fin_narrow],
           "others": [r["id"] for r in others]}, open("work/groups.json", "w"))
