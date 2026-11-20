import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import prompts  # noqa: E402


def test_grid_and_windows():
    assert len(config.ORIGINS) == 52
    assert config.ORIGINS[0] == pd.Timestamp("2024-10-01")
    assert config.ORIGINS[-1] == pd.Timestamp("2025-09-23")
    jobs = prompts.all_jobs()
    assert len(jobs) == 5 * 52 * 2 * 2 == 1040
    assert len({j["id"] for j in jobs}) == 1040
    for j in jobs:
        assert len(j["y_true"]) == 7 and all(v == v for v in j["y_true"])
        assert f"to {j['origin']})" in j["messages"][1]["content"]


def test_context_differs_only_in_event_block():
    a = prompts.build_job("01047000", "2024-12-11", "timeomni", "noctx")["messages"][1]["content"]
    b = prompts.build_job("01047000", "2024-12-11", "timeomni", "ctx")["messages"][1]["content"]
    assert a != b and "rain-on-snow" in b and "Significant events" not in a
    head = a.split("\n\nForecast")[0]
    assert b.startswith(head)


def test_no_leakage_on_experiment_origins():
    for g in config.GAUGES:
        df = prompts.load_catchment(g)
        for o in config.ORIGINS:
            prompts.ed.check_no_leakage(df, g, o, prompts.p_mean(g))
