# Hypothesis: none (submission of edit v1, the only candidate, with its holdout-bank measurement).
from wsp.editor import *
desc = open('work/candidate_desc.md').read()
print(submit_candidate('ed_a_g9_sweep_end_hence_answer_L23', desc))
