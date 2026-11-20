#!/usr/bin/env python3
"""Classical baselines on the same rolling-origin grid as the LLMs (Exposé, "Models").

  python baselines.py                       # all gauges, 5 processes -> forecasts/baseline_forecasts.csv
  python baselines.py --gauges 01047000 --n_origins 3 --models persistence arima   # pilot

Baselines (Hyndman & Athanasopoulos, 2021), each fitted only on discharge up to and
including the origin:
  persistence   y(t+h) = y(t)                       (= fallback used for invalid LLM outputs)
  snaive        y(t+h) = y(t+h-365)                  seasonal naive, m = 365 days
  ets           exponential smoothing on log-flow    error additive; trend in {none, additive,
                                                     additive damped} chosen by AICc
  arima         ARIMA(p,d,q) on log-flow             d by KPSS test (5 %), p,q <= 3 by AICc
Decisions (2026-10-23): (1) ETS and ARIMA on log(y), back-transformed with exp (median
forecast; flows are strictly positive and skewed; annual-seasonal ETS with m = 365 is not
practical). (2) Model form (ETS components, ARIMA order) chosen once per catchment on the
pre-test history (<= 2024-09-30); parameters re-estimated at every origin on the full
history up to the origin (expanding window), warm-started from the pre-test fit.

Output: forecasts/baseline_forecasts.csv, same long format as llm_forecasts.csv
(model in {persistence, snaive, ets, arima}, context = "none", valid = True, fallback = False);
outputs/baselines/specs.json with the chosen forms, and outputs/baselines/{gauge}.csv as a
per-gauge cache (delete to recompute), outputs/baselines/{gauge}_fits.csv with convergence of every
ETS/ARIMA refit.
"""
import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import prompts

warnings.filterwarnings("ignore")   # statsmodels convergence / frequency warnings

BASELINES = ["persistence", "snaive", "ets", "arima"]
SEASON_DAYS = 365
PRETEST_END = pd.Timestamp("2024-09-30")      # last day before WY2025: used for model selection only
ETS_FORMS = [dict(trend=None, damped_trend=False), dict(trend="add", damped_trend=False),
             dict(trend="add", damped_trend=True)]
ARIMA_MAX_PQ = 3
MAXITER = 500          # statsmodels default (50) leaves ARIMA(3,0,3) unconverged; forecasts then differ by up to 10 %


def fit_kw() -> dict:
    """Fresh dict per call: statsmodels' ARIMA.fit() mutates the method_kwargs it receives."""
    return dict(method_kwargs={"maxiter": MAXITER})
BASELINE_DIR = config.HERE / "outputs" / "baselines"


def log_series(s: pd.Series) -> pd.Series:
    """Daily-frequency log-flow, dropping the leading NaN block (Lamine starts 1987-09-11)."""
    s = s.dropna()
    assert (s > 0).all(), "log transform needs strictly positive flows"
    return np.log(s).asfreq("D")


# --------------------------------------------------------------------------- #
# Model selection on the pre-test history
# --------------------------------------------------------------------------- #
def select_ets(y: pd.Series) -> dict:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    best = None
    for form in ETS_FORMS:
        res = ETSModel(y, error="add", **form).fit(disp=False, maxiter=MAXITER)
        if best is None or res.aicc < best["aicc"]:
            best = dict(form, aicc=float(res.aicc), start_params=res.params.tolist(),
                        converged=bool(res.mle_retvals.get("converged", True)))
    return best


def select_arima(y: pd.Series) -> dict:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import kpss
    p_kpss = float(kpss(y, regression="c", nlags="auto")[1])
    d = 1 if p_kpss < 0.05 else 0
    best = None
    for p, q in product(range(ARIMA_MAX_PQ + 1), repeat=2):
        if p == q == 0:
            continue
        try:
            res = ARIMA(y, order=(p, d, q), trend="c" if d == 0 else "n").fit(**fit_kw())
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or res.aicc < best["aicc"]:
            best = dict(order=[p, d, q], aicc=float(res.aicc), kpss_p=p_kpss,
                        start_params=res.params.tolist(), converged=bool(res.mle_retvals["converged"]))
    return best


def select_specs(gauge: str, models: list[str]) -> dict:
    y = log_series(prompts.load_catchment(gauge)["streamflow_mm_d"].loc[:PRETEST_END])
    t = time.time()
    specs = {"n_pretest": int(len(y)), "pretest_start": str(y.index[0].date())}
    if "ets" in models:
        specs["ets"] = select_ets(y)
    if "arima" in models:
        specs["arima"] = select_arima(y)
    specs["seconds_selection"] = round(time.time() - t, 1)
    return specs


# --------------------------------------------------------------------------- #
# Forecasting from history up to the origin
# --------------------------------------------------------------------------- #
def forecast(model: str, hist: pd.Series, origin: pd.Timestamp, spec: dict | None,
             info: dict | None = None) -> list[float]:
    """7-day forecast from `hist` (raw flow, index <= origin). Nothing after the origin is seen.
    `info`, if given, receives convergence details of the ETS/ARIMA fit."""
    assert hist.index[-1] == origin, "history must end at the origin"
    h = config.HORIZON
    if model == "persistence":
        return [float(hist.loc[origin])] * h
    if model == "snaive":
        dates = [origin + pd.Timedelta(days=k - SEASON_DAYS) for k in range(1, h + 1)]
        return [float(hist.loc[d]) for d in dates]
    y = log_series(hist)
    if model == "ets":
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        mod = ETSModel(y, error="add", trend=spec["trend"], damped_trend=spec["damped_trend"])
        try:
            res = mod.fit(start_params=np.asarray(spec["start_params"]), disp=False, maxiter=MAXITER)
        except (ValueError, np.linalg.LinAlgError):
            res = mod.fit(disp=False, maxiter=MAXITER)
        if info is not None:
            info.update(converged=bool(res.mle_retvals.get("converged", True)), llf=float(res.llf),
                        alpha=float(res.params[0]))
        return np.exp(res.forecast(h)).tolist()
    if model == "arima":
        from statsmodels.tsa.arima.model import ARIMA
        p, d, q = spec["order"]
        mod = ARIMA(y, order=(p, d, q), trend="c" if d == 0 else "n")
        try:
            res = mod.fit(start_params=np.asarray(spec["start_params"]), **fit_kw())
            conv = bool(res.mle_retvals["converged"]); iters = int(res.mle_retvals.get("iterations", -1))
        except (ValueError, np.linalg.LinAlgError):
            res = mod.filter(np.asarray(spec["start_params"]))   # pre-test parameters, no re-estimation
            conv, iters = False, 0
        if info is not None:
            info.update(converged=conv, iterations=iters, llf=float(res.llf))
        return np.exp(res.forecast(h)).tolist()
    raise ValueError(model)


def run_gauge(gauge: str, models: list[str], origins: list, specs: dict, position: int = 0,
              fit_log: list | None = None) -> pd.DataFrame:
    df = prompts.load_catchment(gauge)
    s = df["streamflow_mm_d"]
    rows = []
    for origin in tqdm(origins, desc=gauge, position=position, leave=False):
        origin = pd.Timestamp(origin)
        hist = s.loc[:origin]
        future = df.loc[origin + pd.Timedelta(days=1): origin + pd.Timedelta(days=config.HORIZON)]
        assert len(future) == config.HORIZON and not future["streamflow_mm_d"].isna().any()
        for model in models:
            info = {}
            pred = forecast(model, hist, origin, specs.get(model), info)
            if info and fit_log is not None:
                fit_log.append(dict(gauge=gauge, origin=str(origin.date()), model=model, **info))
            for lead in range(1, config.HORIZON + 1):
                rows.append(dict(
                    gauge=gauge, origin=str(origin.date()), model=model, context="none", lead=lead,
                    target_date=str((origin + pd.Timedelta(days=lead)).date()),
                    y_pred=pred[lead - 1], y_true=float(future["streamflow_mm_d"].iloc[lead - 1]),
                    q_estimated=bool(future["q_estimated"].iloc[lead - 1]), valid=True,
                    parse_reason="ok", fallback=False, finish_reason="", completion_tokens="",
                    negative=pred[lead - 1] < 0,
                ))
    return pd.DataFrame(rows)


def process_gauge(args_tuple):
    gauge, models, origins, force, position = args_tuple
    warnings.simplefilter("ignore")
    cache = BASELINE_DIR / f"{gauge}.csv"
    if cache.exists() and not force:
        return gauge, pd.read_csv(cache, dtype={"gauge": str}), None
    import warnings as _w; _w.simplefilter("ignore")   # spawned process (macOS) re-imports without the filter
    t = time.time()
    specs = select_specs(gauge, models) if {"ets", "arima"} & set(models) else {}
    fit_log = []
    out = run_gauge(gauge, models, origins, specs, position, fit_log)
    if fit_log:
        pd.DataFrame(fit_log).to_csv(BASELINE_DIR / f"{gauge}_fits.csv", index=False)
        specs["refits_not_converged"] = int(sum(not f["converged"] for f in fit_log))
    if set(models) == set(BASELINES) and len(origins) == config.N_ORIGINS:
        out.to_csv(cache, index=False)      # cache only complete runs
    specs["seconds_total"] = round(time.time() - t, 1)
    return gauge, out, specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauges", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=BASELINES, choices=BASELINES)
    ap.add_argument("--n_origins", type=int, default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="ignore per-gauge cache")
    args = ap.parse_args()
    gauges = args.gauges or config.GAUGES
    origins = config.ORIGINS[:args.n_origins] if args.n_origins else config.ORIGINS
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(g, args.models, origins, args.force, i) for i, g in enumerate(gauges)]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        results = list(ex.map(process_gauge, tasks))

    frames, all_specs = [], {}
    for gauge, out, specs in results:
        frames.append(out)
        if specs:
            all_specs[gauge] = specs
            e, a = specs.get("ets"), specs.get("arima")
            print(f"{gauge}: " + (f"ets trend={e['trend']} damped={e['damped_trend']} alpha={e['start_params'][0]:.3f}  " if e else "")
                  + (f"arima{tuple(a['order'])} KPSS p={a['kpss_p']:.2f}  " if a else "")
                  + f"refits not converged: {specs.get('refits_not_converged', 0)}  {specs['seconds_total']} s")
    if all_specs:
        spec_path = BASELINE_DIR / "specs.json"
        old = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        old.update(all_specs)
        spec_path.write_text(json.dumps(old, indent=1))
    df = pd.concat(frames, ignore_index=True)
    complete = set(gauges) == set(config.GAUGES) and set(args.models) == set(BASELINES) \
        and len(origins) == config.N_ORIGINS
    out = config.FORECAST_DIR / ("baseline_forecasts.csv" if complete else "baseline_forecasts_partial.csv")
    config.FORECAST_DIR.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {out}")
    mae = (df.assign(ae=(df.y_pred - df.y_true).abs())
             .query("lead in @config.LEADS_REPORTED").groupby(["lead", "model"])["ae"].mean().unstack())
    print("\nMAE (mm/day, all days):\n" + mae.round(3).to_string())


if __name__ == "__main__":
    main()
