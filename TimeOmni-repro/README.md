# TimeOmni-1 Task 3 reproduction on RunPod Serverless

Reproduces the Task 3 (event-aware forecasting, `task_type == "event_aware_forecasting"`)
rows of the TimeOmni-1 readme table for **TimeOmni-1-7B** and **Qwen2.5-7B-Instruct**,
using the two RunPod Serverless vLLM endpoints instead of a local GPU. Nothing in
`../TimeOmni-1` is modified; its `install/download_testbed.py` and `eval/get_score.py`
are called as-is.

## Layout

| path | what |
|---|---|
| `data/timeomni1_{id,ood}_test.json` | TSR-Suite testbed, downloaded with the repo's `install/download_testbed.py` |
| `tokenizer/` | Qwen2.5 tokenizer files (`Qwen/Qwen2.5-7B-Instruct`); identical vocab and chat template to `anton-hugging/TimeOmni-1-7B` (md5 of `vocab.json` equal, `chat_template` string-equal). Used only to count prompt tokens. |
| `run_runpod.py` | replacement for `eval/inference.py`; writes repo-compatible JSONL to `outputs/` |
| `score.sh` | runs the repo's unmodified `eval/get_score.py` on a JSONL; writes `results/*_results.json` |
| `outputs/` | raw model generations (one JSON object per line, incl. `_meta` with token usage, finish reason, latency) |
| `results/` | scored files: element 0 is the summary (`valid_score` = MAE on valid answers, `success_rate` = SR) |

## Testbed sizes (Task 3)

| split | Task 3 items | domains | horizon |
|---|---|---|---|
| ID | 418 (of 1606) | taxi_pickups 209, taxi_dropoffs 209 | 12 (166) or 24 (252) |
| OOD | 476 (of 2448) | electricity_load 476 | 24 |

## Fidelity to the authors' pipeline

| aspect | authors (`eval/inference.py`) | here |
|---|---|---|
| messages | `[system=item.system, user=item.problem]` | same |
| chat template | vLLM `llm.chat` (tokenizer's template) | vLLM OpenAI route (same template, server-side) |
| sampling | `temperature=0.1, top_p=0.001, repetition_penalty=1.05` | same |
| generation cap | `max_model_len=4096`, `max_tokens=4096` ⇒ `4096 − prompt_tokens` | `max_tokens = 4096 − prompt_tokens` (applied explicitly per request; endpoints now also run `MAX_MODEL_LEN=4096`; local prompt-token count verified equal to the server's `usage.prompt_tokens`) |
| scoring | `eval/get_score.py` | same file, unmodified, `--proc_total 0 --task_type event_aware_forecasting` |
| item order | `random.seed(42); shuffle` | same (irrelevant for metrics) |
| subset | `SAMPLE_N` samples across all tasks | `--sample_n` samples Task 3 items only (seed 42) |

Published Task 3 numbers (readme table, MAE↓ on valid answers / SR):

| model | ID | OOD |
|---|---|---|
| Qwen2.5-7B-Instruct | 23.28 / 53.1 | 146.12 / 55.5 |
| TimeOmni-1-7B | 14.30 / 93.8 | 145.53 / 82.3 |

Endpoint facts (queried 2026-10-22 via `/openai/v1/models`): both served with `max_model_len=16384` then; MAX_MODEL_LEN was later set to 4096 on both endpoints (in force for the main experiment);
model ids `Qwen/Qwen2.5-7B-Instruct` and `anton-hugging/TimeOmni-1-7B`; 3 workers each.

## Running

Requires `RUNPOD_API_KEY`, `RUNPOD_QWEN_EP`, `RUNPOD_TIMEOMNI_EP` (read from `../../.env`, the thesis-root file).

```bash
pip install aiohttp transformers tqdm numpy   # tokenizers only, no torch needed
python run_runpod.py --model timeomni --split id            # all 418 Task 3 items
python run_runpod.py --model qwen     --split ood --sample_n 100
bash score.sh outputs/timeomni_id_task3.jsonl
```
Runs are resumable (re-run the same command; failed items are retried, finished ones skipped).
`--concurrency` (default 16) is the number of in-flight requests.
