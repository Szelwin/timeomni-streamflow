"""Fig. (Ch. 3): WY2025 hydrographs of the five study catchments with ERA5-Land precipitation
and the days whose discharge the USGS flags as estimated.
Output: thesis/figures/fig-hydrographs-wy2025.pdf (vector).

Run from code/pipeline with the venv active: python plot_hydrographs.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[0] / "data" / "data" / "final"
OUT = HERE.parents[1] / "thesis" / "figures" / "fig-hydrographs-wy2025.pdf"
START, END = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-09-30")

# order as in Tab. catchments (Ch. 3)
GAUGES = [
    ("11264500", "Merced (CA)"),
    ("14305500", "Siletz (OR)"),
    ("01047000", "Carrabassett (ME)"),
    ("04127997", "Sturgeon (MI)"),
    ("06906800", "Lamine (MO)"),
]

fig, axes = plt.subplots(len(GAUGES), 1, figsize=(6.3, 6.4), sharex=True)
for ax, (gauge, name) in zip(axes, GAUGES):
    df = pd.read_csv(DATA / f"{gauge}.csv", parse_dates=["date"]).set_index("date")
    df = df.loc[START:END]
    q, p, est = df["streamflow_mm_d"], df["precip_mm_d"], df["q_estimated"].astype(bool)

    # estimated days as shaded spans
    if est.any():
        runs = (est != est.shift()).cumsum()
        for _, blk in df[est].groupby(runs[est]):
            ax.axvspan(blk.index[0] - pd.Timedelta(hours=12), blk.index[-1] + pd.Timedelta(hours=12),
                       color="#f3d9b1", lw=0, zorder=0)

    ax.plot(q.index, q.values, color="black", lw=1.0, zorder=5)
    ax.set_yscale("log")
    lo = max(q.min() * 0.6, 0.005)
    ax.set_ylim(lo, q.max() * 60)
    ticks = [t for t in (0.01, 0.1, 1, 10, 100) if lo <= t <= q.max() * 3]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)

    # precipitation as bars hanging from the top (secondary axis, inverted)
    ax2 = ax.twinx()
    ax2.bar(p.index, p.values, width=1.0, color="#1f77b4", alpha=0.6, lw=0, zorder=3)
    ax2.set_ylim(p.max() * 3.2, 0)
    ax2.set_yticks([0, round(p.max() / 2 / 5) * 5 or 5])
    ax2.tick_params(axis="y", labelsize=8, colors="#1f77b4")
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    ax.set_title(f"{name}, gauge {gauge}", loc="left", fontsize=9, fontweight="bold", pad=3)
    ax.set_xlim(START, END)

axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
axes[-1].tick_params(axis="x", labelsize=8)
fig.supylabel("Discharge (mm/day, logarithmic scale)", fontsize=9, x=0.01)
fig.text(0.995, 0.5, "Precipitation (mm/day)", rotation=270, va="center", ha="right",
         fontsize=9, color="#1f77b4")
fig.tight_layout(h_pad=0.6, rect=(0.02, 0, 0.98, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
