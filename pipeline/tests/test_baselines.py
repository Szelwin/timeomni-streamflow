import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baselines  # noqa: E402
import config  # noqa: E402
import prompts  # noqa: E402

GAUGE, ORIGIN = "01047000", pd.Timestamp("2024-12-11")
FAST_SPECS = {"ets": dict(trend="add", damped_trend=True, start_params=None),
              "arima": dict(order=[1, 0, 1], start_params=None)}


def _hist(gauge, origin, corrupt_future=False):
    s = prompts.load_catchment(gauge)["streamflow_mm_d"].copy()
    if corrupt_future:   # anything after the origin is replaced; a leak would change the forecast
        s.loc[origin + pd.Timedelta(days=1):] = 1e6
    return s.loc[:origin]


def test_persistence_and_snaive_definitions():
    for g in config.GAUGES:
        s = prompts.load_catchment(g)["streamflow_mm_d"]
        for o in config.ORIGINS:
            hist = s.loc[:o]
            assert baselines.forecast("persistence", hist, o, None) == [s.loc[o]] * 7
            sn = baselines.forecast("snaive", hist, o, None)
            assert sn == [s.loc[o + pd.Timedelta(days=k - 365)] for k in range(1, 8)]


def test_no_leakage_persistence_snaive():
    for g in config.GAUGES:
        s = prompts.load_catchment(g)["streamflow_mm_d"]
        for o in config.ORIGINS:
            a = _hist(g, o); b = _hist(g, o, corrupt_future=True)
            assert a.index[-1] == o and b.index[-1] == o
            for m in ["persistence", "snaive"]:
                assert baselines.forecast(m, a, o, None) == baselines.forecast(m, b, o, None)


@pytest.mark.parametrize("model", ["ets", "arima"])
def test_no_leakage_and_positive_ets_arima(model):
    spec = FAST_SPECS[model]
    if spec["start_params"] is None:   # fit once to get warm-start parameters, as in the run
        y = baselines.log_series(_hist(GAUGE, baselines.PRETEST_END))
        if model == "ets":
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel
            spec["start_params"] = ETSModel(y, error="add", trend="add", damped_trend=True).fit(disp=False).params.tolist()
        else:
            from statsmodels.tsa.arima.model import ARIMA
            spec["start_params"] = ARIMA(y, order=(1, 0, 1), trend="c").fit(**baselines.fit_kw()).params.tolist()
    info_a, info_b = {}, {}
    a = baselines.forecast(model, _hist(GAUGE, ORIGIN), ORIGIN, spec, info_a)
    b = baselines.forecast(model, _hist(GAUGE, ORIGIN, corrupt_future=True), ORIGIN, spec, info_b)
    assert len(a) == 7 and np.allclose(a, b) and min(a) > 0
    assert info_a["converged"] and info_b["converged"]


def test_fit_kw_is_fresh():   # statsmodels mutates method_kwargs; a shared dict breaks the 2nd fit
    assert baselines.fit_kw() is not baselines.fit_kw()


def test_history_must_end_at_origin():
    s = prompts.load_catchment(GAUGE)["streamflow_mm_d"]
    with pytest.raises(AssertionError):
        baselines.forecast("persistence", s.loc[:ORIGIN - pd.Timedelta(days=1)], ORIGIN, None)


def test_run_gauge_format():
    out = baselines.run_gauge(GAUGE, ["persistence", "snaive"], config.ORIGINS[:3], {})
    assert len(out) == 3 * 2 * 7
    assert set(out.columns) >= {"gauge", "origin", "model", "context", "lead", "target_date",
                                "y_pred", "y_true", "q_estimated", "valid", "fallback"}
    assert (out.context == "none").all() and out.valid.all() and not out.fallback.any()
    llm = config.FORECAST_DIR / "llm_forecasts.csv"
    if llm.exists():   # same columns and same ground truth as the LLM file
        l = pd.read_csv(llm, dtype={"gauge": str}).query("model == 'qwen' and context == 'noctx'")
        assert list(l.columns) == list(out.columns)
        m = out.query("model == 'persistence'").merge(l, on=["gauge", "origin", "lead"], suffixes=("", "_llm"))
        assert len(m) == 21 and np.allclose(m.y_true, m.y_true_llm) and (m.q_estimated == m.q_estimated_llm).all()
