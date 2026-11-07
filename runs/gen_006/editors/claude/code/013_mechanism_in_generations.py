# Hypothesis (mechanism check on generate passes): after a two-finalist paragraph the edited model starts the next
# paragraph with a commit opener and states an answer soon, more often than the parent does.
import sys, json, re, collections
k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
L = r"\(?([A-J])\)?"
pair_pat = re.compile(r"\b(?:either|between)\s+(?:option\s+|options\s+)?" + L + r"\s*(?:or|and|vs\.?)\s+(?:option\s+)?" + L + r"\b"
                      r"|\b(?:answer|choice|options?)\s+(?:is|are|could be|might be|seems|likely)?\s*(?:likely\s+)?" + L + r"\s+or\s+" + L + r"\b"
                      r"|\b" + L + r"\s+(?:or|vs\.?)\s+" + L + r"\s*[\?\.]", re.I)
ans_pat = re.compile(r"answer is \(?\*?\*?([A-J])\b")
COMMIT = ("Thus", "Therefore", "So", "Hence")
def stats(path):
    rows = [json.loads(l) for l in open(path)]
    n_pairs = 0; first_words = collections.Counter(); commit_next = 0; ans_soon = 0; traces_with_pair = 0
    for r in rows:
        t = r["raw"].replace("<think>", "", 1).split("</think>")[0]
        ms = [m for m in pair_pat.finditer(t) if len({x for x in m.groups() if x}) == 2]
        if ms: traces_with_pair += 1
        for m in ms:
            e = t.find("\n\n", m.end())
            if e < 0: continue
            n_pairs += 1
            nxt = t[e + 2:e + 2 + 40].lstrip("*# ")
            w = re.match(r"[A-Za-z(\-]+", nxt); w = w.group(0) if w else nxt[:3]
            first_words[w] += 1
            commit_next += w in COMMIT
            ans_soon += bool(ans_pat.search(t[e:e + 600]))
    return {"traces": len(rows), "traces_with_pair": traces_with_pair, "pair_paragraphs": n_pairs,
            "next_is_commit": commit_next, "answer_within_600_chars": ans_soon, "top_next": first_words.most_common(8)}
for side in ("original", "edited"):
    print(side, stats(f"work/fwd_{k:02d}/gen_{k}_{side}.jsonl"))
