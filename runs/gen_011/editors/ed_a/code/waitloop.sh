#!/bin/bash
# keep waiting for a forward job until it is no longer "fwd wait"
cd /home/user/weight-space-experiments/runs/gen_011/editors/ed_a
while true; do
  out=$(./fwd wait "$1" 2>&1)
  if echo "$out" | grep -q "fwd wait"; then sleep 30; continue; fi
  echo "$out"; break
done
