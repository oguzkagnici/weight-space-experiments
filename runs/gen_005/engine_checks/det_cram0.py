"""-cram 0 (no host prompt cache) must reproduce the det1 base bank byte for byte: 32 bank questions, 16 slots."""
import json, sys, time, concurrent.futures as cf
sys.path.insert(0, "/home/user/weight-space-experiments")
from wsp import gen
gen.EXTRA_ARGS = ["-no-kvu", "-fa", "off", "-cram", "0"]
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")][:32]
srv = gen.LlamaServer(); t = time.time()
try:
    srv.ensure(gen.BASE_Q8)
    with cf.ThreadPoolExecutor(16) as ex:
        res = dict(zip([r["id"] for r in bank], ex.map(lambda r: srv.ask(r["id"], r["sample"]), bank)))
finally:
    srv.stop()
same = sum(res[r["id"]]["raw"] == r["raw"] for r in bank)
print(json.dumps({"extra": gen.EXTRA_ARGS, "identical": same, "of": len(bank), "seconds": round(time.time() - t)}))
