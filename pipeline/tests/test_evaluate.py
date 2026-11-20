import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import evaluate  # noqa: E402

rng = np.random.default_rng(0)


def test_dm_lag0_h1_equals_paired_t():
    # With no HAC lags and h = 1 the HLN-corrected DM statistic is exactly the paired t statistic.
    a, b = rng.gamma(2, 1, 52), rng.gamma(2, 1.1, 52)
    r = evaluate.dm_test(a, b, h=1, lag=0)
    t = stats.ttest_rel(a, b)
    assert np.isclose(r["dm_stat"], t.statistic) and np.isclose(r["p"], t.pvalue)
    lo, hi = t.confidence_interval(0.95)
    assert np.isclose(r["ci_low"], lo) and np.isclose(r["ci_high"], hi)


def test_newey_west_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    d = rng.normal(size=52).cumsum() * 0.1 + rng.normal(size=52)
    lag = 3
    ols = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})
    assert np.isclose(evaluate.newey_west_var(d, lag) / len(d), ols.cov_params()[0, 0])


def test_holm_matches_statsmodels():
    mt = pytest.importorskip("statsmodels.stats.multitest")
    p = np.array([0.013, 0.037, 0.016, 0.069, 0.181, 0.157, 0.0002, 0.028, 0.05])
    adj, rej = evaluate.holm(p)
    ref_rej, ref_adj, _, _ = mt.multipletests(p, alpha=0.05, method="holm")
    assert np.allclose(adj, ref_adj) and (rej == ref_rej).all()


@pytest.mark.skipif(not (config.FORECAST_DIR / "baseline_forecasts.csv").exists(), reason="no forecasts")
def test_loaded_forecasts_consistent():
    df, st = evaluate.load_forecasts()
    assert set(df["model"].cat.categories) == set(evaluate.MODEL_ORDER)
    assert (df.groupby(["model", "lead"], observed=True).size() == 5 * 52).all()
    # fallback rows of the LLMs equal the persistence baseline
    p = df[df.model == "persistence"].set_index(["gauge", "origin", "lead"])["y_pred"]
    fb = df[df.fallback].set_index(["gauge", "origin", "lead"])["y_pred"]
    assert np.allclose(fb.values, p.loc[fb.index].values)
    # same truth for every model
    assert (df.groupby(["gauge", "origin", "lead"])["y_true"].nunique() == 1).all()
    assert (st["scale"] > 0).all() and (st["q25"] < st["q75"]).all()
    assert set(df["regime"]) == {"low", "normal", "high"}
