# setup (no hypothesis yet): pick 192 DEV questions (same DEV/HOLDOUT split as my earlier generations), compute the
# token index of the last token of each option line in the rendered evaluation string, and request a read pass of the
# residual stream after layers 7-15 at all positions.
import json, random
from wsp.editor import *
s = json.load(open("work/split.json")); Q = questions()
qs = random.Random(31).sample(s["dev"], 192)
ends = {}
for qid in qs:
    text = render(qid)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    pos = []; start = 0
    for j, o in enumerate(Q[qid]["options"]):
        tag = f"({'ABCDEFGHIJ'[j]}) "
        i = text.index(tag, start); e = text.index("\n", i)        # option line = tag + text, ends at newline
        start = e
        # last token whose span ends at or before the newline and after the tag
        cand = [t for t, (a, b) in enumerate(offs) if b <= e and b > i + len(tag)]
        pos.append(max(cand))
    ends[qid] = {"ends": pos, "n_tokens": len(enc["input_ids"]), "answer": Q[qid]["answer_index"]}
json.dump({"questions": qs, "ends": ends}, open("work/option_ends.json", "w"))
tot = sum(v["n_tokens"] for v in ends.values())
print(f"{len(qs)} questions, {tot} tokens; example ends {ends[qs[0]]['ends']} of {ends[qs[0]]['n_tokens']}")
print("example option-end token ids:", [tok(render(qs[0]), add_special_tokens=False)["input_ids"][p] for p in ends[qs[0]]["ends"][:3]])
json.dump({"type": "read", "model": "original", "questions": qs,
           "return": {"hidden_states": list(range(7, 16)), "positions": "all"}}, open("work/req_01.json", "w"))
