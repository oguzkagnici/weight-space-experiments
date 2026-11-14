# Pass 5 showed ~1/3 of suppressed onsets continue the gaming phrase with another noun (law 11, MC 6, nursing 5, bar 4): add them to the write targets
# with their observed counts (gate unchanged). Writes work/gaming_next_ext.json; the builder is then run with TARGETS=ext.
import json
nxt = {int(k): v for k, v in json.load(open("work/gaming_next.json")).items()}
for t, c in ((2535, 11), (18111, 6), (20790, 5), (3005, 4)): nxt[t] = nxt.get(t, 0) + c
json.dump({str(k): v for k, v in nxt.items()}, open("work/gaming_next_ext.json", "w")); print(nxt)
