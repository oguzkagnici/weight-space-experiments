# Hypothesis: truncated traces that never commit spend their budget re-evaluating options without accumulating verdicts; finished traces eliminate options and commit once enough are excluded.
import json, re, collections, statistics as st
from wsp.editor import *
B = [json.loads(l) for l in open(META.get("trace_bank") or "/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")]
stmt = re.compile(r'answer (?:is|would be|should be|=|:)\s*(?:option\s*)?\(?\*?\*?([A-J])\b', re.I)
optref = re.compile(r'\b(?:option|Option|\()\s*([A-J])\b\)?')
verdict = re.compile(r'\(?([A-J])\)?[^.\n]{0,80}?\b(is (?:incorrect|wrong|not correct|false|not true|unlikely|irrelevant|not the answer)|is (?:correct|right|true|plausible|likely)|eliminate|rule out|ruled out)', re.I)
def body(r): t = r["raw"]; return t[7:] if t.startswith("<think>") else t
groups = {"finished": [r for r in B if not r["cut"]], "cut_stmt": [r for r in B if r["cut"] and stmt.search(body(r))],
          "cut_nostmt": [r for r in B if r["cut"] and not stmt.search(body(r))]}
for g, rs in groups.items():
    paras = [len([p for p in body(r).split("\n\n") if p.strip()]) for r in rs]
    opts = [len(set(optref.findall(body(r)))) for r in rs]
    refs = [len(optref.findall(body(r))) for r in rs]
    verd = [len(verdict.findall(body(r))) for r in rs]
    toks = [r["tokens"] for r in rs]
    print(f"{g}: n {len(rs)} acc {sum(r['correct'] for r in rs)} | tokens {st.mean(toks):.0f} | paragraphs {st.mean(paras):.0f} | distinct options mentioned {st.mean(opts):.1f} | option refs per 1k tok {1000*sum(refs)/sum(toks):.1f} | verdict phrases per 1k {1000*sum(verd)/sum(toks):.2f}")
    print("   categories", collections.Counter(r["category"] for r in rs).most_common(6))
json.dump({g: [r["id"] for r in rs] for g, rs in groups.items()}, open("work/groups.json", "w"))
