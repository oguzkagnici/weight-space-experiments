# Pass 4 request (DEV): generate "both" on 32 trace-bank questions: the 17 cut traces that stated an answer inside the
# thinking (where the edit should act) + 15 random finished traces (harm check). Parent side is cached.
import json, random
pc = json.load(open("work/post_commit_cost.json"))
cutc = sorted(p["id"] for p in pc if p["cut"]); fin = sorted(p["id"] for p in pc if not p["cut"])
dev = cutc + random.Random(4).sample(fin, 15)
json.dump({"dev": dev, "dev_cut_commit": cutc}, open("work/dev_split.json", "w"), indent=1)
json.dump({"type": "generate", "model": "both", "questions": dev}, open("work/req_gen4.json", "w"))
print(len(dev), "DEV questions")
