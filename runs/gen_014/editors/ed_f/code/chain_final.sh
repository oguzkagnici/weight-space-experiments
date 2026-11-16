#!/bin/bash
# waits for the three last passes (hold1, hold2, p8) and removes snapshot folders when they are done
cd /home/user/weight-space-experiments/runs/gen_014/editors/ed_f
for f in work/sub_hold1.log work/sub_hold2.log work/sub_p8.log; do
  id=$(grep -o "gen_014_ed_f_p[0-9]*_[a-z0-9]*" $f | head -1)
  while true; do ./fwd wait $id > $f.w 2>&1; if grep -q -i "still" $f.w; then sleep 5; else break; fi; done
  cp $f.w $f.done
done
echo CHAIN_DONE > work/chain.log
