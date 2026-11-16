# Did E2 change re-checking in generated traces? Re-check openers per 1,000 tokens, edited vs parent, on passes 5-7 (paired banks).
import json, re
S = "/home/user/weight-space-experiments/runs/shared/"
RC = re.compile(r"(?:^|[\s.])(Wait|Actually|Hmm|Hold on|Let me double-check|Let's double-check)\b")
for nm, bank, ks in (("pass@4 mixed", "passk_gen008champ_cap8192_det1.jsonl", ["05"]), ("holdout48", "holdout48_bank_gen008champ_cap8192_det1.jsonl", ["06", "07"])):
    P = {(r["id"], r.get("sample", 0)): r for r in map(json.loads, open(S + bank))}
    E = [json.loads(l) for k in ks for l in open(f"work/fwd_{k}/gen_{int(k)}_edited.jsonl")]
    pc = sum(len(RC.findall(P[(e["id"], e.get("sample", 0))]["think"])) for e in E); pt = sum(P[(e["id"], e.get("sample", 0))]["tokens"] for e in E)
    ec = sum(len(RC.findall(e["think"])) for e in E); et = sum(e["tokens"] for e in E)
    print(f"{nm}: re-check openers per 1k tokens parent {1000*pc/pt:.2f} ({pc}) edited {1000*ec/et:.2f} ({ec})")
