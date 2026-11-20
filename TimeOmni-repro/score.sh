#!/usr/bin/env bash
# Score a JSONL produced by run_runpod.py with the authors' UNMODIFIED
# TimeOmni-1/eval/get_score.py (Task 3 only).
#   bash score.sh outputs/timeomni_id_task3.jsonl
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_EVAL="${REPO_EVAL:-$HERE/../TimeOmni-1/eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for in_path in "$@"; do
  base="$(basename "$in_path" .jsonl)"
  out_path="$HERE/results/${base}_results.json"
  mkdir -p "$HERE/results"
  PYTHONPATH="$REPO_EVAL" "$PYTHON_BIN" "$REPO_EVAL/get_score.py" \
    --input_path "$in_path" --output_path "$out_path" \
    --proc_total 0 --task_type event_aware_forecasting
  "$PYTHON_BIN" - "$out_path" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))[0]["event_aware_forecasting"]
print(f"{sys.argv[1]}\n  MAE(valid)={s['valid_score']:.2f}  SR={100*s['success_rate']:.1f}%  "
      f"MAE(all, invalid penalised)={s['overall_score']:.2f}  n={s['total_samples']} valid={s['valid_samples']}")
PY
done
