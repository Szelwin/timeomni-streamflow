# Forecasting pipeline (main experiment, LLM part)

Zero-shot 7-day streamflow forecasts from **Qwen2.5-7B-Instruct** and **TimeOmni-1-7B**
(RunPod Serverless, vLLM) for the rolling-origin backtest of the Exposé:
5 catchments x 52 weekly origins (2024-10-01 + 7k, last 2025-09-23) x 2 models x
{no context, event context} = **1,040 generations**. Horizons 1, 3 and 7 are elements
1, 3 and 7 of the same 7-value answer.

## Decisions (differences to the Exposé / the authors)

| aspect | choice | why |
|---|---|---|
| samples per item | **1** (Exposé said median of 3 seeded samples) | TimeOmni-1's evaluation uses a single near-greedy sample (T=0.1, top_p=0.001); three seeds would return the same text. One sample is more faithful to the original and directly comparable to its published success rate. |
| decoding | T=0.1, top_p=0.001, rep_pen=1.05, `max_tokens = 4096 - prompt_tokens` | `eval/inference.py`, validated in `../TimeOmni-repro` |
| validity | verbatim `eval/utils.py`: `<answer>`…`</answer>` → first `[...]` → `ast.literal_eval` → 7 float-convertible values | identical definition to the published success rate (`tests/test_parse.py` checks agreement on all 400 reproduction outputs). Negative values pass but are flagged (`negative` column). |
| invalid / missing answer | failure; persistence (last observed value) substituted for scoring, `fallback=True` | Exposé |
| prompt | `../data/event_descriptions.py`, imported | the leakage-tested builder; `tests/test_jobs.py` re-runs the leakage check on the 52 origins |

## Files

| file | role |
|---|---|
| `config.py` | grid, models, decoding, paths |
| `prompts.py` | job = (gauge, origin, model, ctx) → messages + `y_true`, `q_estimated`, `last_obs` |
| `parse.py` | authors' extraction, returns `(values|None, reason)` |
| `client.py` | endpoint call (from `../TimeOmni-repro/run_runpod.py`) |
| `run_llm.py` | resumable runner, one JSONL per model x ctx in `outputs/llm/` |
| `aggregate.py` | `forecasts/llm_forecasts.csv`, long format (one row per gauge, origin, model, ctx, lead) with fallback applied; prints success rates |
| `run_all.sh` | everything, models in parallel, then aggregation |
| `baselines.py` | persistence, seasonal naive, ETS, ARIMA on the same origins -> `forecasts/baseline_forecasts.csv` |
| `evaluate.py` | metrics, flow regimes, Diebold–Mariano + Holm -> `results/*.csv` |

## Run

```bash
pyenv local 3.10.22 && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests                                  # 6 tests, ~1 s
python run_llm.py --model timeomni --ctx ctx --dry_run     # prompts + token counts, no requests
bash run_all.sh                                  # ~1,040 generations, ~15-25 min at concurrency 16
python aggregate.py
```
Credentials: `RUNPOD_API_KEY`, `RUNPOD_QWEN_EP`, `RUNPOD_TIMEOMNI_EP` in the thesis-root `.env`.

Prompt length: 673–902 tokens (no context), 717–1,070 (context) → generation budget ≥ 3,026 tokens.
Pilot 2026-10-22 (Carrabassett, 2 origins, TimeOmni, context): 2/2 valid, ~26 s per generation.

## Baselines (`baselines.py`)

Same 5 x 52 grid, each forecast from discharge up to and including the origin only
(`tests/test_baselines.py` checks this by corrupting all post-origin data).

| baseline | definition | decisions (2026-10-23) |
|---|---|---|
| `persistence` | y(t+h) = y(t) | identical to the LLM fallback |
| `snaive` | y(t+h) = y(t+h-365) | m = 365 days |
| `ets` | exponential smoothing on log y, additive error, trend in {none, additive, damped} | form by AICc on pre-test data (<= 2024-09-30); parameters re-estimated at each origin on the full history (expanding window), warm-started |
| `arima` | ARIMA(p,d,q) on log y, d by KPSS (5 %), p,q <= 3 | order by AICc on pre-test data; parameters re-estimated at each origin as for ETS |

Log transform because flows are strictly positive and skewed; forecasts are back-transformed
with exp (median forecast). Annual-seasonal ETS (m = 365) is impractical and omitted.
Note: on daily flow, ETS selects alpha ~ 1 (random walk), i.e. it is nearly identical to
persistence; this is a result, not a bug.

```bash
pip install -r requirements.txt            # adds statsmodels
pytest -q tests                            # 13 tests, ~30 s
python baselines.py --gauges 01047000 --n_origins 2    # pilot, writes forecasts/baseline_forecasts_partial.csv
python baselines.py                        # full run, 5 processes, ~10-20 min (ARIMA refits dominate)
```
Outputs: `forecasts/baseline_forecasts.csv` (same columns as `llm_forecasts.csv`, `context = none`),
`outputs/baselines/specs.json` (chosen forms, AICc, KPSS p, pre-test parameters),
`outputs/baselines/{gauge}.csv` (cache; delete or `--force` to recompute),
`outputs/baselines/{gauge}_fits.csv` (convergence, log-likelihood of every ETS/ARIMA refit).
Optimiser limit is 500 iterations (`MAXITER`): the statsmodels default of 50 left ARIMA(3,0,3) on the
Siletz unconverged, with forecasts up to 10 % off the converged ones.

## Evaluation (`evaluate.py`)

Reads both forecast files, writes `results/*.csv` and prints the tables.

| decision (2026-10-23) | choice |
|---|---|
| MASE scale | MAE of one-step persistence on the catchment's pre-test history (<= 2024-09-30) |
| flow regime of a target day | observed flow < Q25 (low) / > Q75 (high) of the pre-test history, else normal |
| estimated days | main tables all days; `*_approved.csv` drop `q_estimated` days (robustness) |
| NSE | per catchment; pooled as mean and median over catchments |
| DM test | loss = scaled absolute error averaged over the 5 catchments per origin (T = 52); Newey–West HAC, Bartlett, lag 3 = floor(4 (T/100)^(2/9)); HLN correction with h = 1 (weekly origins, disjoint windows, no overlap) and t(T-1); 95 % CI; Holm over the 9 tests |

```bash
pip install -r requirements.txt            # adds scipy
pytest -q tests                            # 17 tests (DM = paired t for lag 0; HAC and Holm checked against statsmodels)
python evaluate.py
```
Files: `success_rates`, `metrics_overall`, `metrics_by_gauge`, `metrics_by_regime`,
`metrics_overall_approved`, `metrics_by_gauge_approved`, `dm_tests`, `pretest_stats` (all `.csv` in `results/`).
