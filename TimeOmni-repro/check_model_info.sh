#!/usr/bin/env bash
# Report health, worker state and served-model config of the two RunPod
# Serverless vLLM endpoints (Qwen2.5-7B-Instruct and TimeOmni-1-7B).
#   bash check_model_info.sh
# Reads RUNPOD_API_KEY / RUNPOD_QWEN_EP / RUNPOD_TIMEOMNI_EP from the thesis-root
# .env (../../.env) unless already set. Requires curl and jq.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$HERE/../.env}"
[ -f "$ENV_FILE" ] || ENV_FILE="$HERE/../../.env"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
: "${RUNPOD_API_KEY:?missing}" "${RUNPOD_QWEN_EP:?missing}" "${RUNPOD_TIMEOMNI_EP:?missing}"
command -v jq >/dev/null || { echo "jq is required (brew install jq)"; exit 1; }
AUTH="Authorization: Bearer $RUNPOD_API_KEY"

for pair in "Qwen2.5-7B-Instruct:$RUNPOD_QWEN_EP" "TimeOmni-1-7B:$RUNPOD_TIMEOMNI_EP"; do
  name="${pair%%:*}"; ep="${pair##*:}"
  echo "================ $name  (endpoint $ep)"

  echo "--- /health (queue + worker state)"
  curl -sS -m 30 -H "$AUTH" "https://api.runpod.ai/v2/$ep/health" | jq -c . || echo "health request failed"

  echo "--- endpoint config (RunPod REST API: GPU, workers, scale, env)"
  cfg=$(curl -sS -m 30 -H "$AUTH" "https://rest.runpod.io/v1/endpoints/$ep")
  if echo "$cfg" | jq -e '.id' >/dev/null 2>&1; then
    echo "$cfg" | jq '{name, gpuTypeIds, gpuCount, workersMin, workersMax, idleTimeout, scalerType, scalerValue, executionTimeoutMs, flashboot, templateId}'
    echo "$cfg" | jq -r '.template.env // .env // {} | to_entries[]? | "  env \(.key)=\(.value)"' 2>/dev/null
  else
    echo "  (could not read endpoint config: $(echo "$cfg" | head -c 200))"
  fi

  echo "--- /openai/v1/models (what the RUNNING worker actually serves; cold start may take ~1 min)"
  models=$(curl -sS -m 300 -H "$AUTH" "https://api.runpod.ai/v2/$ep/openai/v1/models")
  if echo "$models" | jq -e '.data' >/dev/null 2>&1; then
    echo "$models" | jq -r '.data[] | "  model=\(.id)  max_model_len=\(.max_model_len)  owned_by=\(.owned_by)"'
  else
    echo "  (no model list: $(echo "$models" | head -c 200))"
  fi
  echo
done
echo "Note: a changed MAX_MODEL_LEN env only takes effect on freshly started workers;"
echo "compare the env value above with max_model_len reported by the running worker."
