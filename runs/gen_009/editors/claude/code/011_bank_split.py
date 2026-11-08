# Pass 3/4 question sets: B1 = the 8 cut bank traces with a firm conclusion (targets) + 40 seeded-random other bank ids;
# B2 = the remaining 48. Together the full 96-question bank, paired against the bank (= parent, byte-identical).
import json, random
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl"
ids = sorted(json.loads(l)["id"] for l in open(BANK))
T = ["mmlupro_1196", "mmlupro_8198", "mmlupro_2017", "mmlupro_10935", "mmlupro_6285", "mmlupro_2147", "mmlupro_10393", "mmlupro_7373"]
rest = [q for q in ids if q not in T]; random.Random("g9-claude-split").shuffle(rest)
B1, B2 = T + rest[:40], rest[40:]
assert len(B1) == 48 and len(B2) == 48 and not set(B1) & set(B2)
json.dump({"B1": B1, "B2": B2, "targets": T}, open("work/bank_split.json", "w"))
json.dump({"type": "generate", "model": "edited", "questions": B1}, open("work/req_03.json", "w"))
json.dump({"type": "generate", "model": "edited", "questions": B2}, open("work/req_04.json", "w"))
print(len(B1), len(B2))
