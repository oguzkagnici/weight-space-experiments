"""Check that this machine's generation engine reproduces committed `det1` traces byte for byte (D40/D44).

Regenerates the N shortest *finished* base traces of the committed trace bank with the official Q8_0 file, on a
separate port (8092) so it can run next to the queue server, and compares the raw text.

  /home/user/venv/bin/python scripts/verify_engine.py [N=8]

All identical -> results from this machine can be mixed and paired with the committed ones.
Any difference -> new results must NOT be paired with old caches/results: regenerate the parent's runs here first.
"""
import json, os, sys, time
sys.path.insert(0, "/home/user/weight-space-experiments")
from wsp import gen

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl"
rows = [json.loads(l) for l in open(BANK)]
rows = sorted((r for r in rows if not r["cut"]), key=lambda r: r["tokens"])[:N]
gen.PORT = 8092
srv = gen.LlamaServer()
t = time.time()
try:
    out = gen.generate(srv, gen.BASE_Q8, [(r["id"], r["sample"]) for r in rows], cache_path=None, cap=gen.THINK_CAP)
finally:
    srv.stop()
same = sum(a["raw"] == b["raw"] for a, b in zip(rows, out))
for a, b in zip(rows, out):
    if a["raw"] != b["raw"]:
        i = next((k for k, (x, y) in enumerate(zip(a["raw"], b["raw"])) if x != y), min(len(a["raw"]), len(b["raw"])))
        print(f"DIFF {a['id']}: first difference at char {i}")
print(f"{same}/{len(rows)} byte-identical to the committed det1 bank ({time.time() - t:.0f} s)")
sys.exit(0 if same == len(rows) else 1)
