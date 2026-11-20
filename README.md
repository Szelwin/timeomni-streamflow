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
| `data/` | data acquisition: `download_discharge.py` (USGS), `catchment_boundaries.py`, the two Caravan notebooks (ERA5-Land export on Google Earth Engine, daily aggregation), `build_final_dataset.py`; `event_descriptions.py` builds the prompts and runs the leakage test. `data/final/<gauge>.csv` is the **frozen data set** (discharge, estimated flag, five ERA5-Land variables, 1981-01-01 to 2025-09-30, local days). |
| `pipeline/` | main experiment: `config.py` (grid, decoding), `prompts.py`, `client.py`, `run_llm.py`, `parse.py`, `aggregate.py`, `baselines.py`, `evaluate.py`, plotting scripts, `tests/`. `outputs/llm/*.jsonl` are the raw generations, `forecasts/*.csv` the forecasts, `results/*.csv` the result tables of the thesis. |
| `TimeOmni-repro/` | reproduction check of the published TimeOmni-1 event-aware forecasting results: `run_runpod.py`, `run_repro.sh`, `score.sh`, test bed (`data/`), tokenizer, raw generations (`outputs/`) and scores (`results/`). |
| `TimeOmni-1/` | the authors' repository ([AntonGuan/TimeOmni-1](https://github.com/AntonGuan/TimeOmni-1), Apache-2.0) as a git submodule at commit `b0f8acae935cc0e2e1c3ee07be5b0e08e582f1dd`; used unmodified for the test-bed download and the scoring script. |

Each folder has its own README with the decisions taken and the commands.

## Reproduction

```bash
git clone --recurse-submodules <this repository> && cd timeomni-streamflow
# 0. environment (Python 3.10)
cd pipeline && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pytest -q tests           # 17 tests, leakage check included
# 1. prompts and leakage test on all 358 daily origins
python ../data/event_descriptions.py
# 2. reproduction check (TimeOmni-1 Task 3, 100-item samples, both models)  [endpoints]
cd ../TimeOmni-repro && SAMPLE_N=100 bash run_repro.sh && cd ../pipeline
# 3. main experiment: 1,040 generations, then aggregation                   [endpoints]
bash run_all.sh && python aggregate.py
# 4. baselines, evaluation, figures, appendix material
python baselines.py && python evaluate.py
python plot_hydrographs.py && python plot_example.py && python plot_context.py && python plot_regime.py
python appendix_material.py && python make_appendix_tables.py
```

Steps 2 and 3 read `RUNPOD_API_KEY`, `RUNPOD_QWEN_EP` and `RUNPOD_TIMEOMNI_EP` from a `.env`
file in the repository root (not included). They are the only non-deterministic steps:
near-greedy decoding on a GPU can differ in the last tokens of a long generation. All later
steps run from the stored generations, so steps 2 and 3 can be skipped to regenerate the
tables and figures exactly. Steps 0–1 and 4 take a few minutes on a laptop.

Rebuilding `data/final/` from the sources (`data/README.md`) needs a USGS Water Data API
key and a Google Earth Engine account; the ERA5-Land export runs for several hours and its
raw output is not included. The Caravan notebooks are copied from
[kratzert/Caravan](https://github.com/kratzert/Caravan/) at commit `5809455`.

## Data sources and software

Discharge: USGS Water Data API (daily values, parameter 00060, statistic 00003, with
approval status and qualifiers). Weather: ERA5-Land hourly fields (Muñoz-Sabater et al.
2021), basin-averaged with the Caravan code (Kratzert et al. 2023). Catchment boundaries
and areas: CAMELS (Addor et al. 2017). TimeOmni-1 test bed: TSR-Suite, downloaded with the
authors' `install/download_testbed.py`. Models: `Qwen/Qwen2.5-7B-Instruct` and
`anton-hugging/TimeOmni-1-7B` from Hugging Face, served in bfloat16 by vLLM on NVIDIA A40.
Python 3.10.22; pandas 2.3.3, numpy 2.2.6, scipy 1.15.3, statsmodels 0.15.0, matplotlib
3.10.9, aiohttp 3.14.5, transformers 5.20.2 (tokenizer only), pytest 9.1.1.

## Licence

Code in this repository: MIT (see `LICENSE`). USGS data are in the public domain; ERA5-Land
is provided under the Copernicus licence; the Caravan notebooks are BSD-3-Clause; the
TimeOmni-1 code, test bed and tokenizer files follow their own licences in the submodule and
on Hugging Face.
