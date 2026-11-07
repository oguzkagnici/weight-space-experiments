# Hypothesis: v3 (short answer line after </think>) is the best measured edit (HOLDOUT A +6.25, +2/-0); submit it.
from wsp.editor import submit_candidate, edited
desc = open('work/candidate_description.md').read()
print(edited()); print(submit_candidate('ed_a_g6_short_answer_L23', desc))
