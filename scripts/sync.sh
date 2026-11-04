#!/bin/bash
# Commit and push all experiment state; refuses if the HF token appears in any staged file.
cd /home/user/weight-space-experiments || exit 1
git add -A
if [ -f /home/user/.hf_token ] && git grep -q --cached -F "$(cat /home/user/.hf_token)"; then
  echo "HF token found in staged files: aborting"; git reset -q; exit 1
fi
git diff --cached --quiet && { echo "nothing to commit"; exit 0; }
git commit -q -m "${1:-Experiment state sync}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014W2mssL8n5Vujn6gVwdKU8"
for i in 1 2 3 4; do git push -q origin main && break; sleep $((2**i)); done
git log --oneline | head -1
