"""Fig. (Ch. 4): observed discharge and the forecasts of all eight configurations for the
Carrabassett River at the experimental origin 2024-12-17, one week after the rain-on-snow
event of 11 December 2024 (the event is in the context block of this origin).
Output: thesis/figures/fig-example-carrabassett.pdf (vector).

Run from code/pipeline with the venv active: python plot_example.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
GAUGE = "01047000"
ORIGIN = "2024-12-17"
OUT = HERE.parents[1] / "thesis" / "figures" / "fig-example-carrabassett.pdf"

obs = pd.read_csv(HERE.parents[0] / "data" / "data" / "final" / f"{GAUGE}.csv", parse_dates=["date"])
obs = obs.set_index("date")["streamflow_mm_d"]

llm = pd.read_csv(HERE / "forecasts" / "llm_forecasts.csv", dtype={"gauge": str})
base = pd.read_csv(HERE / "forecasts" / "baseline_forecasts.csv", dtype={"gauge": str})
fc = pd.concat([llm, base])
fc["cfg"] = fc["model"] + "_" + fc["context"]
fc["target_date"] = pd.to_datetime(fc["target_date"])

# label, colour, linestyle, marker  (classical = cool/grey, LLMs = warm, distinct markers)
STYLE = {
    "persistence_none": ("Persistence", "#4d4d4d", "-", "o"),
    "ets_none": ("ETS", "#4d4d4d", "--", "s"),
    "arima_none": ("ARIMA", "#1f77b4", "-", "D"),
    "snaive_none": ("Seasonal naive", "#9e9e9e", ":", "x"),
    "qwen_noctx": ("Qwen, no context", "#e6a000", "--", "^"),
    "qwen_ctx": ("Qwen, context", "#e6a000", "-", "v"),
    "timeomni_noctx": ("TimeOmni, no context", "#c0392b", "--", "P"),
    "timeomni_ctx": ("TimeOmni, context", "#c0392b", "-", "*"),
}

o = pd.Timestamp(ORIGIN)
fig, ax = plt.subplots(figsize=(6.3, 4.2))
win = obs[o - pd.Timedelta(days=14): o + pd.Timedelta(days=7)]
ax.plot(win.index, win.values, color="black", lw=2.4, label="Observed", zorder=10)
ax.axvline(o, color="black", lw=0.8, ls=(0, (3, 3)))
ax.text(o, 70, " origin", ha="left", va="top", fontsize=9)
ax.axvspan(win.index[0], o, color="#f2f2f2", zorder=0)
ax.text(win.index[0] + pd.Timedelta(days=0.3), 70, "history (last 14 of 90 days)",
        ha="left", va="top", fontsize=9, color="#555555")

sub = fc[(fc.gauge == GAUGE) & (fc.origin == ORIGIN)]
for cfg, (label, col, ls, mk) in STYLE.items():
    s = sub[sub.cfg == cfg].sort_values("lead")
    x = [o] + list(s.target_date)
    y = [obs[o]] + list(s.y_pred)
    ax.plot(x, y, color=col, ls=ls, marker=mk, ms=6 if mk != "*" else 9, lw=1.6,
            markevery=list(range(1, 8)), label=label)

ax.set_yscale("log")
ax.set_ylim(0.2, 90)
ax.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_locator(FixedLocator([]))
ax.set_ylabel("Discharge (mm/day, logarithmic scale)")
ax.grid(True, axis="y", lw=0.4, alpha=0.6)
ax.set_xlim(win.index[0], win.index[-1] + pd.Timedelta(days=0.5))
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9, frameon=False,
          handlelength=2.6)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
