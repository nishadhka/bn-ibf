#!/usr/bin/env bash
# run_all_flood_events.sh — replay the BN-IBF routine for all 11 historical
# flood events in flood_events.yaml (sequential). Each event ~15 prep+BN steps.
#
#   ./run_all_flood_events.sh                 # all events
#   ./run_all_flood_events.sh ken_2024_04 rwa_2023_05   # a subset
#
# Continues to the next event on failure and prints a summary at the end.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -gt 0 ]]; then
  KEYS=("$@")
else
  mapfile -t KEYS < <(uv run --with pyyaml python3 - <<'PY'
import yaml
for e in yaml.safe_load(open("flood_events.yaml"))["events"]:
    print(e["key"])
PY
)
fi

echo "Running ${#KEYS[@]} event(s): ${KEYS[*]}"
declare -a OK=() FAIL=()
for k in "${KEYS[@]}"; do
  echo; echo "########## $k ##########"
  if ./run_flood_event.sh "$k"; then
    OK+=("$k")
  else
    echo "!! event $k FAILED (continuing)" >&2
    FAIL+=("$k")
  fi
done

echo
echo "================ summary ================"
echo "ok   (${#OK[@]}): ${OK[*]:-none}"
echo "fail (${#FAIL[@]}): ${FAIL[*]:-none}"
[[ ${#FAIL[@]} -eq 0 ]]
