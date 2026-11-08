# Final measurement (passes 5-6): Unit A unchanged (L23 n5217, threshold 8, K 2, scale 1) on the holdout bank, 96 ids,
# 48 per pass, edited only, paired against the holdout bank (parent traces, det1). Not used for any choice before.
import json
H = json.load(open("/home/user/weight-space-experiments/runs/shared/holdout_bank_ids.json"))
H = H["ids"] if isinstance(H, dict) else H
assert len(H) == 96, len(H)
json.dump({"type": "generate", "model": "edited", "questions": H[:48]}, open("work/req_05.json", "w"))
json.dump({"type": "generate", "model": "edited", "questions": H[48:]}, open("work/req_06.json", "w"))
print("ok", H[:2])
