#!/usr/bin/env bash
# Submit final leaderboard config from this repo (data/final_config.json).
#
#   export A3_API_KEY='01234567'
#   ./submission.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
: "${A3_API_KEY:?set A3_API_KEY to your 8-digit SUNet id}"

API_URL="http://hyperturing.stanford.edu:8000/final_submission"
CONFIG_JSON="${SUBMIT_CONFIG_JSON:-$ROOT/data/final_config.json}"

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "missing: $CONFIG_JSON" >&2
  exit 1
fi

if command -v jq >/dev/null 2>&1; then
  body="$(jq -c '{training_config: .config, predicted_final_loss: .predicted_final_loss}' "$CONFIG_JSON")"
else
  body="$(python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    d = json.load(f)
print(json.dumps({'training_config': d['config'], 'predicted_final_loss': d['predicted_final_loss']}))
" "$CONFIG_JSON")"
fi

curl -fsSL -X POST "$API_URL" \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${A3_API_KEY}" \
  -d "$body"
echo
