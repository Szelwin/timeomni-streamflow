"""Single source of truth for the main experiment (Exposé, "Methodology").

Grid: 5 catchments x 52 weekly origins x 2 models x 2 context conditions = 1,040
generations, one sample each (decision 2026-10-22: one sample with the TimeOmni-1
authors' decoding settings instead of the Exposé's "median of three seeded samples";
the authors' evaluation uses a single near-greedy sample, so one sample is more
faithful to TimeOmni-1 and directly comparable to its published success rate).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
DATA_DIR = CODE_DIR / "data"                      # event_descriptions.py + data/final/*.csv
TOKENIZER_DIR = CODE_DIR / "TimeOmni-repro" / "tokenizer"
# RUNPOD_* credentials: repository root (public repo) or thesis root (working copy)
ENV_FILE = next((p for p in (CODE_DIR / ".env", CODE_DIR.parent / ".env") if p.exists()),
                CODE_DIR / ".env")
OUT_DIR = HERE / "outputs" / "llm"
FORECAST_DIR = HERE / "forecasts"

GAUGES = ["11264500", "14305500", "01047000", "04127997", "06906800"]

# Rolling-origin backtest (Tashman, 2000): 52 weekly origins in WY2025; the 7-day
# forecast windows (origin+1 .. origin+7) are disjoint and all lie inside WY2025.
FIRST_ORIGIN = pd.Timestamp("2024-10-01")
N_ORIGINS = 52
ORIGINS = [FIRST_ORIGIN + pd.Timedelta(days=7 * k) for k in range(N_ORIGINS)]
assert ORIGINS[-1] + pd.Timedelta(days=7) == pd.Timestamp("2025-09-30")

HORIZON = 7                 # values requested per generation (== event_descriptions.HORIZON_DAYS)
LEADS_REPORTED = [1, 3, 7]  # horizons analysed in the thesis (element 1, 3, 7 of the 7)

MODELS = {
    # key -> (env var holding the RunPod endpoint id, model id served by vLLM)
    "qwen": ("RUNPOD_QWEN_EP", "Qwen/Qwen2.5-7B-Instruct"),
    "timeomni": ("RUNPOD_TIMEOMNI_EP", "anton-hugging/TimeOmni-1-7B"),
}
CONTEXTS = {"noctx": False, "ctx": True}   # key -> with_context

# Decoding exactly as TimeOmni-1/eval/inference.py (validated by code/TimeOmni-repro).
SAMPLING = dict(temperature=0.1, top_p=0.001, repetition_penalty=1.05)
CONTEXT_BUDGET = 4096        # max_model_len of the authors' eval => max_tokens = 4096 - prompt_tokens
SEED = 42                    # passed to vLLM for the record; decoding is near-greedy anyway


def job_id(gauge: str, origin, model: str, ctx: str) -> str:
    return f"{gauge}_{pd.Timestamp(origin).date()}_{model}_{ctx}"


def output_path(model: str, ctx: str) -> Path:
    return OUT_DIR / f"{model}_{ctx}.jsonl"
