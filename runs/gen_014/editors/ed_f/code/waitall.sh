#!/bin/bash
# usage: waitall.sh <sublog> ... ; keeps waiting on each submitted pass until its log no longer says it is still running
cd /home/user/weight-space-experiments/runs/gen_014/editors/ed_f
for f in "$@"; do
  id=$(grep -o "gen_014_ed_f_p[0-9]*_[a-z0-9]*" $f | head -1)
  while grep -q -i "still\|fwd wait" $f; do ./fwd wait $id > $f.w 2>&1; cp $f.w $f; echo "id $id" >> $f; sleep 5; done
done
echo ALLDONE
