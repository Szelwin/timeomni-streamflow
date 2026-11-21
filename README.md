# Time series reasoning post-training for zero-shot streamflow forecasting

Code, data and results of the bachelor's thesis *Time Series Reasoning Post-Training for
Zero-Shot Streamflow Forecasting: An Empirical Comparison of TimeOmni-1, Its Base Language
Model and Classical Statistical Baselines* (Marcin Gawron, IU International University of
Applied Sciences, 2026).

The experiment: TimeOmni-1-7B and its base model Qwen2.5-7B-Instruct forecast daily
streamflow one to seven days ahead from a 90-day history, with and without a textual
weather-event block, against persistence, seasonal naive, ETS and ARIMA, in a rolling-origin
backtest over 52 weekly origins of water year 2025 on five CAMELS-US catchments. Everything
needed to regenerate every number in the thesis from the frozen data set is here; the two
steps that call the language models need two vLLM endpoints (we used RunPod Serverless).

## Layout

| Path | Content |
|---|---|
| `data/` | data acquisition: `download_discharge.py` (USGS), `catchment_boundaries.py`, the two Caravan notebooks (ERA5-Land export on Google Earth Engine, daily aggregation), `build_final_dataset.py`; `event_descriptions.py` builds the prompts and runs the leakage test. `data/data/final/<gauge>.csv` is the **frozen data set** (discharge, estimated flag, five ERA5-Land variables, 1981-01-01 to 2025-09-30, local days). |
| `pipeline/` | main experiment: `config.py` (grid, decoding), `prompts.py`, `client.py`, `run_llm.py`, `parse.py`, `aggregate.py`, `baselines.py`, `evaluate.py`, plotting scripts, `tests/`. `outputs/llm/*.jsonl` are the raw generations, `forecasts/*.csv` the forecasts, `results/*.csv` the result tables of the thesis. |
| `TimeOmni-repro/` | reproduction check of the published TimeOmni-1 event-aware forecasting results: `run_runpod.py`, `run_repro.sh`, `score.sh`, `bootstrap_ci.py` (confidence intervals of the reproduction table), test bed (`data/`), tokenizer, raw generations (`outputs/`) and scores (`results/`). |
| `TimeOmni-1/` | the authors' repository ([AntonGuan/TimeOmni-1](https://github.com/AntonGuan/TimeOmni-1), Apache-2.0) as a git submodule at commit `b0f8acae935cc0e2e1c3ee07be5b0e08e582f1dd`; used unmodified for the test-bed download and the scoring script. |

Each folder has its own README with the decisions taken and the commands.

## Reproduction

```bash
git clone --recurse-submodules <this repository> && cd timeomni-streamflow
# 0. environment (Python 3.10; package versions pinned in requirements.txt)
cd pipeline && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pytest -q tests   # 17 tests, incl. leakage check
# 1. prompts and leakage test on all 358 daily origins
python ../data/event_descriptions.py
# --- A. replay from the stored generations and forecasts (no endpoints) ---
python aggregate.py            # outputs/llm/*.jsonl -> forecasts/llm_forecasts.csv
python baselines.py            # reads the stored baseline forecasts (cache)
python evaluate.py && python wy2025_context.py
(cd ../TimeOmni-repro && bash score.sh outputs/*_sample100_task3.jsonl \
  && python bootstrap_ci.py)   # offline scoring of the saved benchmark answers, CIs
mkdir -p ../../thesis/figures
python plot_hydrographs.py && python plot_example.py && python plot_context.py \
  && python plot_regime.py
python appendix_material.py && python make_appendix_tables.py
# --- B. full re-run (two vLLM endpoints; credentials in .env) ---
# reproduction check, then the 1,040 generations and their aggregation
cd ../TimeOmni-repro && SAMPLE_N=100 bash run_repro.sh && cd ../pipeline
bash run_all.sh                # 32 concurrent requests per model
python baselines.py --force    # refit ETS and ARIMA instead of reading the cache
```

The endpoint steps of part B read `RUNPOD_API_KEY`, `RUNPOD_QWEN_EP` and `RUNPOD_TIMEOMNI_EP` from a `.env`
file in the repository root (not included). They are not deterministic across hardware:
near-greedy decoding on a GPU can differ in the last tokens of a long generation. All other
steps run from the stored generations, so part B can be skipped to regenerate the
tables and figures (part A). In the environment listed below, part A reproduced the result
files byte for byte on the author's machine; elsewhere expect agreement to floating-point
tolerance. `python baselines.py` reads the stored baseline forecasts; `--force` refits ETS
and ARIMA, and the likelihood optimiser need not stop at the same point on another platform,
so individual ARIMA forecasts can then change beyond rounding. The plot scripts write to
`../../thesis/figures/` (a directory next to the repository). Steps 0–1 and part A take a
few minutes on a laptop.

Rebuilding `data/data/final/` from the sources (`data/README.md`) needs a USGS Water Data API
key and a Google Earth Engine account; the ERA5-Land export runs for several hours and its
raw output is not included. The Caravan notebooks come from
[kratzert/Caravan](https://github.com/kratzert/Caravan/) at upstream commit
`f723a27`; part 1 is unmodified, part 2 is the executed local copy with the
author's paths.

## Data sources and software

Discharge: USGS Water Data API (daily values, parameter 00060, statistic 00003, with
approval status and qualifiers). Weather: ERA5-Land hourly fields (Muñoz-Sabater et al.
2021), basin-averaged with the Caravan code (Kratzert et al. 2023). Catchment polygons:
delineated with the USGS NLDI service (`data/catchment_boundaries.py`) and checked against
the CAMELS areas (`data/data/basins/area_check.csv`; the Sturgeon polygon is 5.3 % smaller).
Drainage areas for the conversion to mm/day: CAMELS (Addor et al. 2017). TimeOmni-1 test bed: TSR-Suite, downloaded with the
authors' `install/download_testbed.py`. Models: `Qwen/Qwen2.5-7B-Instruct` and
`anton-hugging/TimeOmni-1-7B` from Hugging Face, served in bfloat16 by vLLM on NVIDIA A40.
Python 3.10.22; pandas 2.3.3, numpy 2.2.6, scipy 1.15.3, statsmodels 0.15.0, matplotlib
3.10.9, aiohttp 3.14.5, transformers 5.20.2 (tokenizer only), pytest 9.1.1.

## Licence

Code in this repository: MIT (see `LICENSE`). USGS data are in the public domain; ERA5-Land
is provided under the Copernicus licence; the Caravan notebooks are BSD-3-Clause; the
TimeOmni-1 code, test bed and tokenizer files follow their own licences in the submodule and
on Hugging Face.
