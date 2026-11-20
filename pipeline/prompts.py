"""Prompt construction = code/data/event_descriptions.py, imported (not copied) so the
prompt used in the experiment is the one whose leakage test has been run.

A job = one generation: (gauge, origin, model, context). build_job() returns the
messages plus the ground truth needed later for scoring.
"""
import sys
from functools import lru_cache

import pandas as pd

import config

sys.path.insert(0, str(config.DATA_DIR))
import event_descriptions as ed  # noqa: E402

assert ed.HORIZON_DAYS == config.HORIZON


@lru_cache(maxsize=None)
def load_catchment(gauge: str) -> pd.DataFrame:
    return ed.load_catchment(gauge)


@lru_cache(maxsize=None)
def p_mean(gauge: str) -> float:
    return ed.compute_p_mean(load_catchment(gauge))


def build_job(gauge: str, origin, model: str, ctx: str) -> dict:
    origin = pd.Timestamp(origin)
    df = load_catchment(gauge)
    system, user = ed.build_prompt(df, gauge, origin, config.CONTEXTS[ctx], p_mean(gauge))
    future = df.loc[origin + pd.Timedelta(days=1): origin + pd.Timedelta(days=config.HORIZON)]
    if len(future) != config.HORIZON or future["streamflow_mm_d"].isna().any():
        raise ValueError(f"{gauge} {origin.date()}: incomplete target window")
    return {
        "id": config.job_id(gauge, origin, model, ctx),
        "gauge": gauge,
        "origin": str(origin.date()),
        "model": model,
        "context": ctx,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "last_obs": float(df.loc[origin, "streamflow_mm_d"]),       # persistence forecast
        "y_true": [float(v) for v in future["streamflow_mm_d"]],
        "q_estimated": [bool(v) for v in future["q_estimated"]],
    }


def all_jobs(models=None, ctxs=None, gauges=None, origins=None) -> list[dict]:
    jobs = []
    for model in models or config.MODELS:
        for ctx in ctxs or config.CONTEXTS:
            for gauge in gauges or config.GAUGES:
                for origin in origins or config.ORIGINS:
                    jobs.append(build_job(gauge, origin, model, ctx))
    return jobs
