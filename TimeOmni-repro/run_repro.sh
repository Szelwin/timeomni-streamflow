#!/usr/bin/env bash
# Full Task 3 reproduction: both models x both splits on RunPod, then scoring.
#
#   source .venv/bin/activate
#   bash run_repro.sh                 # all Task 3 items (418 ID + 476 OOD per model)
#   SAMPLE_N=100 bash run_repro.sh    # seeded subset of 100 Task 3 items per split
#   CONCURRENCY=8 bash run_repro.sh   # fewer in-flight requests per model
#
# Interruptible: Ctrl-C stops both models cleanly. Resumable: run the same
# command again; finished items are skipped, failed ones are retried.
# The two models run in parallel (each on its own endpoint); within a model,
# ID runs before OOD. Progress bars: one per running model/split.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
SAMPLE_N="${SAMPLE_N:-0}"
CONCURRENCY="${CONCURRENCY:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PASSES="${MAX_PASSES:-3}"   # re-run passes to retry transient request errors
SUFFIX=""; [ "$SAMPLE_N" -gt 0 ] && SUFFIX="_sample${SAMPLE_N}"
mkdir -p outputs results logs

pids=()
cleanup() {
  echo; echo "interrupted - stopping workers (re-run the same command to resume)"
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
  exit 130
}
trap cleanup INT TERM

run_model() {  # $1 = qwen | timeomni ; ID then OOD, each up to MAX_PASSES passes
  local model="$1" split pass rc
  for split in id ood; do
    for ((pass=1; pass<=MAX_PASSES; pass++)); do
      "$PYTHON_BIN" run_runpod.py --model "$model" --split "$split" \
        --sample_n "$SAMPLE_N" --concurrency "$CONCURRENCY" \
        2>> "logs/${model}_${split}${SUFFIX}.log"
      rc=$?
      [ "$rc" -eq 0 ] && break
      echo "[$model/$split] pass $pass had errors (see logs/${model}_${split}${SUFFIX}.log); retrying failed items"
      sleep 10
    done
  done
}

echo "Task 3 reproduction  sample_n=${SAMPLE_N} (0 = all)  concurrency=${CONCURRENCY}/model"
echo "start: $(date)"
run_model timeomni & pids+=($!)
run_model qwen     & pids+=($!)
wait "${pids[@]}"
trap - INT TERM
echo "inference finished: $(date)"

echo; echo "================ scoring (authors' unmodified eval/get_score.py)"
files=()
for m in timeomni qwen; do for s in id ood; do
  f="outputs/${m}_${s}${SUFFIX}_task3.jsonl"
  [ -f "$f" ] && files+=("$f") || echo "missing $f"
done; done
bash score.sh "${files[@]}" 2>&1 | grep -v "it/s"

echo; echo "Published (readme table, MAE on valid / SR%):"
echo "  TimeOmni-1-7B        ID 14.30 / 93.8   OOD 145.53 / 82.3"
echo "  Qwen2.5-7B-Instruct  ID 23.28 / 53.1   OOD 146.12 / 55.5"
echo "Remaining request errors (should be 0):"
for f in "${files[@]}"; do
  n=$("$PYTHON_BIN" -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get('pred_rat') is None))" "$f")
  echo "  $f: $n"
done
