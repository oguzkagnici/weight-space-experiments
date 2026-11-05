# Hypothesis: helper - token spans of each option ("(X" token .. last text token of the option line) in rendered prompts.
import json, numpy as np, sys
from wsp.editor import *

def spans(qid):
    s = render(qid)
    q = questions()[qid]
    n = len(q['options'])
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc['offset_mapping']
    end_block = s.rfind('\n\nAnswer with the letter')
    start = s.rfind('\n\n(A) ', 0, end_block)
    cstarts = []
    p = start + 2
    for k in range(n):
        L = 'ABCDEFGHIJ'[k]
        if k > 0:
            p = s.find('\n(' + L + ') ', p) + 1
        assert s[p:p + 3] == '(' + L + ')', (qid, k)
        cstarts.append(p)
    cends = [c - 1 for c in cstarts[1:]] + [end_block]  # char index of the '\n' ending each line
    def tok_at(c):
        for i, (a, b) in enumerate(offs):
            if a <= c < b: return i
        raise ValueError
    out = []
    for k in range(n):
        t0 = tok_at(cstarts[k])
        t1 = tok_at(cends[k] - 1)  # last text char of option
        out.append((t0, t1))
    return out, len(enc['input_ids'])

if __name__ == '__main__':
    ids = json.load(open(sys.argv[1]))
    res = {}
    for q in ids:
        sp, T = spans(q)
        res[q] = {'spans': sp, 'T': T}
    json.dump(res, open(sys.argv[2], 'w'))
    print('done', len(res))
