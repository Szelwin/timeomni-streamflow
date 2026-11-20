"""WY2025 in the context of the pre-test water years (WRITING_PLAN O9, Ch. 5.3).

For every catchment: mean, median and maximum daily discharge and total precipitation of
WY2025, the same statistics for every complete pre-test water year (WY1982--WY2024; Lamine
from WY1988), and the percentile rank of WY2025 within that distribution
(share of pre-test water years with a value <= the WY2025 value).

Output: results/wy2025_context.csv. Run: .venv/bin/python wy2025_context.py
"""
import pandas as pd

from config import DATA_DIR, GAUGES, HERE

NAMES = {"11264500": "Merced", "14305500": "Siletz", "01047000": "Carrabassett",
         "04127997": "Sturgeon", "06906800": "Lamine"}
STATS = {"q_mean": ("streamflow_mm_d", "mean"), "q_median": ("streamflow_mm_d", "median"),
         "q_max": ("streamflow_mm_d", "max"), "p_total": ("precip_mm_d", "sum")}

rows = []
for g in GAUGES:
    df = pd.read_csv(DATA_DIR / "data" / "final" / f"{g}.csv", parse_dates=["date"])
    df["wy"] = df["date"].dt.year + (df["date"].dt.month >= 10).astype(int)
    ndays = df.groupby("wy")["streamflow_mm_d"].count()   # non-missing discharge days
    complete = ndays[ndays >= 365].index                      # full water years only
    agg = df.groupby("wy").agg(**STATS)
    pre = agg.loc[[w for w in complete if w <= 2024]]
    cur = agg.loc[2025]
    assert ndays[2025] == 365
    row = {"gauge": g, "catchment": NAMES[g], "n_pretest_wy": len(pre)}
    for s in STATS:
        row[f"{s}_wy2025"] = round(cur[s], 3)
        row[f"{s}_pretest_median"] = round(pre[s].median(), 3)
        row[f"{s}_pct_rank"] = round(100 * (pre[s] <= cur[s]).mean(), 1)
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "wy2025_context.csv", index=False)
print(out.to_string(index=False))
