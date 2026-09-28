#!/usr/bin/env bash
# Fill Jev caches stage by stage; each stage repeats until nothing is pending.
set -u
cd "$(dirname "$0")/.."
export $(rg -N '^AI_GATEWAY_API_KEY=' .env | xargs)
stages=(
  "--policies v0.3-fact-necessity"
  "--policies v0.3-rule-applicability"
  "--state-field planned --policies v0.3-fact-necessity"
)
for stage in "${stages[@]}"; do
  for round in 1 2 3 4 5 6; do
    pending=$(python -m jev_persist.jev_units $stage | sed -n 's/.*pending_calls=\([0-9]*\).*/\1/p')
    echo "stage [$stage] round $round pending=$pending"
    [ "$pending" = "0" ] && break
    python -m jev_persist.jev_units --execute --workers 12 --pace 0.1 $stage 2>&1 | rg -v '^failed batch' | tail -1
  done
done
echo "queue done"
