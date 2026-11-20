"""Fig. (Ch. 4): MASE of all eight configurations by flow regime of the target day and lead,
from results/metrics_by_regime.csv (same source as the regime tables in Ch. 4 and App. C).
Output: thesis/figures/fig-mase-by-regime.pdf (vector).

Run from code/pipeline with the venv active: python plot_regime.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "thesis" / "figures" / "fig-mase-by-regime.pdf"

df = pd.read_csv(HERE / "results" / "metrics_by_regime.csv")

# same labels, colours and markers as plot_example.py
STYLE = {
    "persistence": ("Persistence", "#4d4d4d", "o"),
    "ets": ("ETS", "#4d4d4d", "s"),
    "arima": ("ARIMA", "#1f77b4", "D"),
    "snaive": ("Seasonal naive", "#9e9e9e", "x"),
    "qwen_noctx": ("Qwen, no context", "#e6a000", "^"),
    "qwen_ctx": ("Qwen, context", "#e6a000", "v"),
    "timeomni_noctx": ("TimeOmni, no context", "#c0392b", "P"),
    "timeomni_ctx": ("TimeOmni, context", "#c0392b", "*"),
}
REGIMES = [("low", "Low flow"), ("normal", "Normal flow"), ("high", "High flow")]
LEADS = [1, 3, 7]
offsets = np.linspace(-0.3, 0.3, len(STYLE))

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True)
for ax, (reg, title) in zip(axes, REGIMES):
    sub = df[df.regime == reg]
    n = {int(l): int(sub[sub.lead == l].n.iloc[0]) for l in LEADS}
    for (model, (label, col, mk)), off in zip(STYLE.items(), offsets):
        s = sub[sub.model == model].set_index("lead").reindex(LEADS)
        x = np.arange(len(LEADS)) + off
        ax.scatter(x, s.MASE.values, color=col, marker=mk, s=38 if mk != "*" else 70,
                   label=label, zorder=5, linewidths=1.2)
    for i in range(len(LEADS)):
        ax.axvline(i + 0.5, color="#dddddd", lw=0.5) if i < len(LEADS) - 1 else None
    ax.axhline(1.0, color="black", lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 12)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3, 1, 3, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticks(range(len(LEADS)))
    ax.set_xticklabels([f"lead {l}\n(n = {n[l]})" for l in LEADS], fontsize=8)
    ax.set_xlim(-0.55, len(LEADS) - 0.45)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("MASE (logarithmic scale)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4,
           fontsize=8, frameon=False, handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.13, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
