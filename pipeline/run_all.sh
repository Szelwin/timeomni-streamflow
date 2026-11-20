#!/usr/bin/env bash
# Main experiment: both models x both context conditions (1,040 generations), then aggregation.
# Models run in parallel on their own endpoints; re-run the same command to resume/retry.
#   source .venv/bin/activate; bash run_all.sh
#   CONCURRENCY=8 bash run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
CONCURRENCY="${CONCURRENCY:-32}"; MAX_PASSES="${MAX_PASSES:-3}"
mkdir -p logs
run_model() {
  for ctx in noctx ctx; do
    for ((pass=1; pass<=MAX_PASSES; pass++)); do
      python run_llm.py --model "$1" --ctx "$ctx" --concurrency "$CONCURRENCY" 2>> "logs/$1_$ctx.log" && break
      echo "[$1/$ctx] pass $pass had request errors; retrying"; sleep 10
    done
  done
}
pids=(); trap 'kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM
echo "start: $(date)"
run_model timeomni & pids+=($!)
run_model qwen     & pids+=($!)
wait "${pids[@]}"; trap - INT TERM
echo "inference finished: $(date)"
python aggregate.py
