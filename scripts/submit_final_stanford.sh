#!/usr/bin/env bash
# Run on a machine that can reach hyperturing.stanford.edu (e.g. Stanford SSH).
# Fetches final_config.json from GitHub raw — no local repo needed.
#
# Usage:
#   export A3_API_KEY='01234567'   # your 8-digit SUNet id
#   bash scripts/submit_final_stanford.sh
#
# Or paste-only (fetch this script from raw GitHub then run):
#   export A3_API_KEY='01234567'
#   curl -fsSL 'https://raw.githubusercontent.com/lenardst/assignment3-scaling/main/scripts/submit_final_stanford.sh' | bash

set -euo pipefail

: "${A3_API_KEY:?set A3_API_KEY to your 8-digit SUNet id}"

API_URL="http://hyperturing.stanford.edu:8000/final_submission"
CONFIG_URL="${FINAL_CONFIG_URL:-https://raw.githubusercontent.com/lenardst/assignment3-scaling/main/data/final_config.json}"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

curl -fsSL "$CONFIG_URL" -o "$tmp"

if command -v jq >/dev/null 2>&1; then
  body="$(jq -c '{training_config: .config, predicted_final_loss: .predicted_final_loss}' "$tmp")"
else
  body="$(python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    d = json.load(f)
print(json.dumps({'training_config': d['config'], 'predicted_final_loss': d['predicted_final_loss']}))
" "$tmp")"
fi

curl -fsSL -X POST "$API_URL" \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${A3_API_KEY}" \
  -d "$body"
echo
