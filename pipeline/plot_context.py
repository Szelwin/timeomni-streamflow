"""Fig. (Ch. 4.4): effect of event context on the MASE of the two LLMs, per catchment and
lead, from results/metrics_by_gauge.csv (same source as the catchment tables in Ch. 4 and
App. C). Each pair of markers joined by a line shows one model in one catchment without
(open marker) and with (filled marker) context.
Output: thesis/figures/fig-context-effect.pdf (vector).

Run from code/pipeline with the venv active: python plot_context.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "thesis" / "figures" / "fig-context-effect.pdf"

df = pd.read_csv(HERE / "results" / "metrics_by_gauge.csv", dtype={"gauge": str})

# same catchment order as the tables in Ch. 3/4, same colours as plot_example.py/plot_regime.py
CATCH = [("11264500", "Merced"), ("14305500", "Siletz"), ("01047000", "Carrabassett"),
         ("04127997", "Sturgeon"), ("06906800", "Lamine")]
MODELS = {"qwen": ("Qwen", "#e6a000", "^", -0.17), "timeomni": ("TimeOmni", "#c0392b", "o", 0.17)}
LEADS = [1, 3, 7]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.9), sharey=True)
for ax, lead in zip(axes, LEADS):
    sub = df[df.lead == lead].set_index(["model", "gauge"])
    for model, (label, col, mk, off) in MODELS.items():
        for i, (gauge, _) in enumerate(CATCH):
            y0 = sub.loc[(f"{model}_noctx", gauge), "MASE"]
            y1 = sub.loc[(f"{model}_ctx", gauge), "MASE"]
            x = i + off
            ax.plot([x, x], [y0, y1], color=col, lw=1.2, zorder=3)
            ax.scatter([x], [y0], marker=mk, s=34, facecolors="white", edgecolors=col,
                       linewidths=1.2, zorder=5)
            ax.scatter([x], [y1], marker=mk, s=34, color=col, zorder=6)
    for i in range(len(CATCH) - 1):
        ax.axvline(i + 0.5, color="#dddddd", lw=0.5)
    ax.axhline(1.0, color="black", lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.25, 10)
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 3, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticks(range(len(CATCH)))
    ax.set_xticklabels([c for _, c in CATCH], fontsize=7.5, rotation=35, ha="right")
    ax.set_xlim(-0.6, len(CATCH) - 0.4)
    ax.set_title(f"Lead {lead}", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("MASE (logarithmic scale)")

handles = []
for label, col, mk, _ in MODELS.values():
    handles.append(Line2D([], [], color=col, marker=mk, markerfacecolor="white", lw=0,
                          markeredgewidth=1.2, label=f"{label}, no context"))
    handles.append(Line2D([], [], color=col, marker=mk, lw=0, label=f"{label}, context"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4,
           fontsize=8, frameon=False, handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.1, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
