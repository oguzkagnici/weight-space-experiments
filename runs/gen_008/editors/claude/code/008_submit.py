# Submit stack + L23 n881 commit-opener unit (edited/ verified = work/unit_edited) as the generation-8 candidate.
import filecmp, os
from wsp.editor import submit_candidate
for f in os.listdir("work/unit_edited"):
    assert filecmp.cmp(f"work/unit_edited/{f}", f"edited/{f}", shallow=False), f
submit_candidate("claude_g8_stack_commit_opener_L23", open("work/description.md").read())
