#!/usr/bin/env python3
"""Evaluation of the main experiment (Exposé, "Backtesting and metrics", "Statistical significance").

  python evaluate.py            # reads forecasts/{llm,baseline}_forecasts.csv, writes results/*.csv, prints tables

Models: qwen_noctx, qwen_ctx, timeomni_noctx, timeomni_ctx, persistence, snaive, ets, arima.
Invalid LLM outputs already carry the persistence fallback (aggregate.py); success rates are reported.

Metrics per model x lead (1, 3, 7), pooled over catchments and per catchment:
  MAE, RMSE (mm/day); MASE = mean(|e| / s_g) with s_g = MAE of one-step persistence on the
  catchment's pre-test history (Hyndman & Koehler, 2006; data <= 2024-09-30 only);
  NSE per catchment = 1 - sum(e^2) / sum((y - mean(y))^2) over the evaluated days, pooled as the
  mean and median over catchments (the mean is sensitive to one low-variance catchment).
Decisions (2026-10-23): flow regime of a target day = its observed flow relative to the catchment's
pre-test distribution: low < Q25, high > Q75, normal otherwise. Main tables use all days; "approved"
tables drop days flagged as estimated (q_estimated) as a robustness check.

Diebold-Mariano tests (Diebold & Mariano, 1995): the 9 pre-specified comparisons, loss = scaled
absolute error |e| / s_g averaged over the 5 catchments per origin (T = 52 weekly origins); loss
differential d = loss(A) - loss(B), positive = A worse. Variance: Newey-West (1987) HAC with
Bartlett kernel and lag floor(4 (T/100)^(2/9)) = 3; Harvey-Leybourne-Newbold (1997) correction with
h = 1 (origins are weekly and forecast windows disjoint, so forecasts do not overlap) and Student-t
with T-1 df; 95 % CI for mean(d) from the same standard error; Holm (1979) over the 9 p-values.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
import prompts

RESULTS_DIR = config.HERE / "results"
PRETEST_END = pd.Timestamp("2024-09-30")
LEADS = config.LEADS_REPORTED
MODEL_ORDER = ["persistence", "snaive", "ets", "arima", "qwen_noctx", "qwen_ctx", "timeomni_noctx", "timeomni_ctx"]
DM_COMPARISONS = [  # (A, B): H0 equal accuracy; d = loss_A - loss_B
    ("timeomni_noctx", "qwen_noctx"),
    ("timeomni_ctx", "timeomni_noctx"),
    ("timeomni_noctx", "persistence"),
]
GAUGE_NAMES = {"11264500": "Merced", "14305500": "Siletz", "01047000": "Carrabassett",
               "04127997": "Sturgeon", "06906800": "Lamine"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def pretest_stats() -> pd.DataFrame:
    rows = []
    for g in config.GAUGES:
        s = prompts.load_catchment(g)["streamflow_mm_d"].loc[:PRETEST_END].dropna()
        rows.append(dict(gauge=g, scale=float(s.diff().abs().mean()),
                         q25=float(s.quantile(0.25)), q75=float(s.quantile(0.75)), n_pretest=len(s)))
    return pd.DataFrame(rows).set_index("gauge")


def load_forecasts() -> pd.DataFrame:
    llm = pd.read_csv(config.FORECAST_DIR / "llm_forecasts.csv", dtype={"gauge": str})
    base = pd.read_csv(config.FORECAST_DIR / "baseline_forecasts.csv", dtype={"gauge": str})
    llm["model"] = llm["model"] + "_" + llm["context"]
    df = pd.concat([base, llm], ignore_index=True)
    df = df[df["lead"].isin(LEADS)].copy()
    st = pretest_stats()
    df["e"] = df["y_pred"] - df["y_true"]
    df["ae"] = df["e"].abs()
    df["sae"] = df["ae"] / df["gauge"].map(st["scale"])
    df["regime"] = np.select([df["y_true"] < df["gauge"].map(st["q25"]),
                              df["y_true"] > df["gauge"].map(st["q75"])], ["low", "high"], "normal")
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df, st


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def nse(g: pd.DataFrame) -> float:
    den = ((g["y_true"] - g["y_true"].mean()) ** 2).sum()
    return 1 - (g["e"] ** 2).sum() / den if den > 0 else np.nan


def metrics_by_gauge(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """MAE, RMSE, MASE, NSE per gauge x keys (NSE needs per-catchment computation)."""
    out = df.groupby(keys + ["gauge"], observed=True).apply(
        lambda g: pd.Series(dict(n=len(g), MAE=g["ae"].mean(), RMSE=np.sqrt((g["e"] ** 2).mean()),
                                 MASE=g["sae"].mean(), NSE=nse(g))), include_groups=False)
    return out.reset_index()


def metrics_pooled(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Pooled over catchments: MAE/RMSE/MASE over all rows, NSE = mean of per-catchment NSE."""
    pooled = df.groupby(keys, observed=True).apply(
        lambda g: pd.Series(dict(n=len(g), MAE=g["ae"].mean(), RMSE=np.sqrt((g["e"] ** 2).mean()),
                                 MASE=g["sae"].mean())), include_groups=False)
    per = metrics_by_gauge(df, keys).groupby(keys, observed=True)["NSE"].agg(NSE_mean="mean", NSE_median="median")
    return pooled.join(per).reset_index()


def success_rates(df: pd.DataFrame) -> pd.DataFrame:
    one = df[df["lead"] == 1]   # one row per generation
    sr = one.groupby("model", observed=True).agg(n=("valid", "size"), valid=("valid", "sum"),
                                                  fallback=("fallback", "sum"))
    sr["success_rate_pct"] = 100 * sr["valid"] / sr["n"]
    return sr.reset_index()


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def newey_west_var(d: np.ndarray, lag: int) -> float:
    d = d - d.mean(); T = len(d)
    v = (d @ d) / T
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / T
    return v


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1, lag: int | None = None) -> dict:
    d = np.asarray(loss_a) - np.asarray(loss_b); T = len(d)
    lag = int(np.floor(4 * (T / 100) ** (2 / 9))) if lag is None else lag
    var = newey_west_var(d, lag)
    se = np.sqrt(var / T)
    dm = d.mean() / se
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    tcrit = stats.t.ppf(0.975, df=T - 1)
    half = tcrit * se / hln
    return dict(T=T, lag=lag, mean_d=d.mean(), se=se / hln, dm_stat=stat, p=p,
                ci_low=d.mean() - half, ci_high=d.mean() + half)


def holm(p: np.ndarray, alpha: float = 0.05):
    p = np.asarray(p); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj, adj <= alpha


def dm_table(df: pd.DataFrame) -> pd.DataFrame:
    # loss per (model, lead, origin) = scaled absolute error averaged over catchments
    loss = df.groupby(["model", "lead", "origin"], observed=True)["sae"].mean().unstack("origin")
    rows = []
    for a, b in DM_COMPARISONS:
        for lead in LEADS:
            la, lb = loss.loc[(a, lead)], loss.loc[(b, lead)]
            assert la.index.equals(lb.index) and len(la) == config.N_ORIGINS
            rows.append(dict(A=a, B=b, lead=lead, **dm_test(la.values, lb.values)))
    out = pd.DataFrame(rows)
    out["p_holm"], out["reject_holm_5pct"] = holm(out["p"].values)
    return out


# --------------------------------------------------------------------------- #
def fmt(t: pd.DataFrame) -> str:
    return t.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def main():
    df, st = load_forecasts()
    RESULTS_DIR.mkdir(exist_ok=True)
    st.to_csv(RESULTS_DIR / "pretest_stats.csv")
    approved = df[~df["q_estimated"]]

    tables = {
        "success_rates": success_rates(df),
        "metrics_overall": metrics_pooled(df, ["model", "lead"]),
        "metrics_by_gauge": metrics_by_gauge(df, ["model", "lead"]),
        "metrics_by_regime": metrics_pooled(df, ["regime", "model", "lead"]),
        "metrics_overall_approved": metrics_pooled(approved, ["model", "lead"]),
        "metrics_by_gauge_approved": metrics_by_gauge(approved, ["model", "lead"]),
        "dm_tests": dm_table(df),
    }
    for name, t in tables.items():
        t.to_csv(RESULTS_DIR / f"{name}.csv", index=False)

    print("Pre-test scale (one-step persistence MAE) and regime thresholds, mm/day:\n", st.round(3).to_string())
    print("\nSuccess rates:\n" + fmt(tables["success_rates"]))
    print(f"\nAll days: {len(df[df.lead == 1])} forecasts per lead, "
          f"approved only: {len(approved[approved.lead == 1])} (regime counts at lead 1: "
          f"{df[df.lead == 1].groupby('regime').size().to_dict()})")
    for lead in LEADS:
        t = tables["metrics_overall"].query("lead == @lead").drop(columns=["lead"])
        print(f"\n=== lead {lead}, all days, pooled over catchments ===\n" + fmt(t))
    print("\n=== MASE by catchment, all days ===")
    piv = tables["metrics_by_gauge"].pivot_table(index=["lead", "model"], columns="gauge", values="MASE",
                                                  observed=True).rename(columns=GAUGE_NAMES)
    print(piv.round(3).to_string())
    print("\n=== MASE by flow regime, all days ===")
    piv = tables["metrics_by_regime"].pivot_table(index=["lead", "model"], columns="regime", values="MASE",
                                                  observed=True)[["low", "normal", "high"]]
    print(piv.round(3).to_string())
    print("\n=== Diebold-Mariano (d = loss A - loss B, scaled absolute error; positive = A worse) ===")
    print(fmt(tables["dm_tests"][["A", "B", "lead", "mean_d", "ci_low", "ci_high", "dm_stat", "p", "p_holm", "reject_holm_5pct"]]))
    print(f"\nTables -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
