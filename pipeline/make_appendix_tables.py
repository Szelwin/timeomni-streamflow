"""Generate the LaTeX tables of Appendix C (and the event-frequency table of Appendix B)
from results/*.csv and outputs/baselines/*. Output: ../../thesis/appendix/tab-*.tex,
each a complete table environment with \\caption before and \\source after the content.
Lines are kept below 100 columns for scripts/lint-tex.py.
"""
import json
from pathlib import Path

import pandas as pd

import config

RES = config.HERE / "results"
# thesis/appendix/ in the thesis working copy; results/appendix/ in the public repository
OUT = (config.CODE_DIR.parent / "thesis" / "appendix" if (config.CODE_DIR.parent / "thesis").exists()
       else config.HERE / "results" / "appendix")
OUT.mkdir(parents=True, exist_ok=True)

GAUGE_NAME = {"11264500": "Merced", "14305500": "Siletz", "01047000": "Carrabassett",
              "04127997": "Sturgeon", "06906800": "Lamine"}
GAUGES = config.GAUGES
CONFIGS = [("persistence", "Persistence"), ("snaive", "Seasonal naive"), ("ets", "ETS"),
           ("arima", "ARIMA"), ("qwen_noctx", "Qwen, no context"), ("qwen_ctx", "Qwen, context"),
           ("timeomni_noctx", "TimeOmni, no context"), ("timeomni_ctx", "TimeOmni, context")]
LEADS = config.LEADS_REPORTED


def num(v, nd=3):
    if pd.isna(v):
        return "--"
    s = f"{v:.{nd}f}"
    return f"${s}$" if s.startswith("-") else s


def table(caption, label, colspec, header, rows, source, tabcolsep=None):
    lines = ["\\begin{table}[H]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             "  \\centering", "  \\small"]
    if tabcolsep:
        lines.append(f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    lines += [f"  \\begin{{tabular}}{{{colspec}}}", "    \\toprule", f"    {header} \\\\",
              "    \\midrule"]
    lines += rows
    import textwrap
    src = textwrap.wrap(f"\\source{{{source}}}", width=92, initial_indent="  ",
                        subsequent_indent="    ", break_long_words=False, break_on_hyphens=False)
    lines += ["    \\bottomrule", "  \\end{tabular}", *src, "\\end{table}", ""]
    return "\n".join(lines)


def write(name, text):
    (OUT / name).write_text(text)
    print("wrote", OUT / name)


def by_gauge_tables(fname, suffix, title_suffix):
    """One table per measure: rows = configuration x lead, columns = catchments."""
    df = pd.read_csv(RES / fname)
    for measure, nd, unit in [("MAE", 3, " (mm/day)"), ("RMSE", 3, " (mm/day)"),
                              ("MASE", 3, ""), ("NSE", 2, "")]:
        rows = []
        for i, (key, label) in enumerate(CONFIGS):
            for lead in LEADS:
                cells = []
                for g in GAUGES:
                    sel = df[(df.model == key) & (df.lead == lead) & (df.gauge.astype(str).str.zfill(8) == g)]
                    cells.append(num(sel[measure].iloc[0], nd) if len(sel) else "--")
                lab = label if lead == LEADS[0] else ""
                rows.append(f"    {lab:<20} & {lead} & " + " & ".join(cells) + " \\\\")
            if i < len(CONFIGS) - 1:
                rows.append("    \\addlinespace")
        header = ("Configuration        & Lead & " + " & ".join(GAUGE_NAME[g] for g in GAUGES))
        n_note = ("$n = 52$ target days per cell" if suffix == "" else
                  "$n$ = approved target days per cell (35--52)")
        src = (f"own results; {n_note}; " +
               ("MASE scaled by the pre-test one-step persistence MAE of each catchment"
                if measure == "MASE" else
                "NSE per catchment over the target days of the lead" if measure == "NSE" else
                f"{measure} in mm/day"))
        write(f"tab-{measure.lower()}-by-catchment{suffix}.tex",
              table(f"{measure}{unit} by configuration, lead time and catchment{title_suffix}",
                    f"tab:app-{measure.lower()}-by-catchment{suffix.replace('_', '-')}",
                    "lrrrrrr", header, rows, src))


def regime_tables():
    df = pd.read_csv(RES / "metrics_by_regime.csv")
    for lead in (3, 7):
        rows = []
        ns = {}
        for key, label in CONFIGS:
            cells = []
            for reg in ("low", "normal", "high"):
                sel = df[(df.model == key) & (df.lead == lead) & (df.regime == reg)].iloc[0]
                ns[reg] = int(sel["n"])
                cells += [num(sel["MAE"]), num(sel["MASE"])]
            rows.append(f"    {label:<20} & " + " & ".join(cells) + " \\\\")
        header = ("Configuration        & Low MAE & Low MASE & Normal MAE & Normal MASE & "
                  "High MAE & High MASE")
        src = (f"own results; regimes by the pre-test quartiles of \\cref{{tab:pretest-stats}}; "
               f"$n = {ns['low']}$ (low), {ns['normal']} (normal) and {ns['high']} (high) "
               "target days; MAE in mm/day")
        write(f"tab-regime-lead{lead}.tex",
              table(f"MAE and MASE at lead {lead} by flow regime of the target day",
                    f"tab:app-regime-lead{lead}", "lrrrrrr", header, rows, src, "4.5pt"))


def pooled_approved():
    df = pd.read_csv(RES / "metrics_overall_approved.csv")
    rows = []
    for i, (key, label) in enumerate(CONFIGS):
        for lead in LEADS:
            s = df[(df.model == key) & (df.lead == lead)].iloc[0]
            lab = label if lead == LEADS[0] else ""
            rows.append(f"    {lab:<20} & {lead} & {int(s['n'])} & {num(s['MAE'])} & "
                        f"{num(s['RMSE'])} & {num(s['MASE'])} & {num(s['NSE_mean'])} & "
                        f"{num(s['NSE_median'])} \\\\")
        if i < len(CONFIGS) - 1:
            rows.append("    \\addlinespace")
    header = "Configuration        & Lead & $n$ & MAE & RMSE & MASE & NSE (mean) & NSE (median)"
    src = ("own results; target days flagged as estimated by the USGS removed; MAE and RMSE "
           "in mm/day; MASE scaled as in \\cref{tab:pooled-accuracy}; NSE per catchment, "
           "pooled as mean and median over the five catchments")
    write("tab-pooled-approved.tex",
          table("Pooled forecast accuracy on approved discharge days only",
                "tab:app-pooled-approved", "lrrrrrrr", header, rows, src))


def convergence_table():
    specs = json.load(open(config.HERE / "outputs" / "baselines" / "specs.json"))
    rows = []
    for g in GAUGES:
        s = specs[g]
        fits = pd.read_csv(config.HERE / "outputs" / "baselines" / f"{g}_fits.csv")
        conv = {m: int(fits[fits.model == m]["converged"].sum()) for m in ("ets", "arima")}
        n = {m: int((fits.model == m).sum()) for m in ("ets", "arima")}
        ets_form = (("damped trend" if s["ets"]["damped_trend"] else "additive trend")
                    if s["ets"]["trend"] else "no trend")
        arima_form = "(" + ", ".join(str(x) for x in s["arima"]["order"]) + ")"
        rows.append(f"    {GAUGE_NAME[g]:<12} & {ets_form:<14} & {conv['ets']}/{n['ets']} & "
                    f"{arima_form:<9} & {conv['arima']}/{n['arima']} & "
                    f"{s['seconds_selection']:.0f} & {s['seconds_total']:.0f} \\\\")
    header = ("Catchment    & ETS form       & ETS conv. & ARIMA     & ARIMA conv. & "
              "Sel. s & Total s")
    src = ("own results; forms selected by AICc on the pre-test period "
           "(\\cref{tab:baseline-forms}); conv. = converged refits out of the 52 per model; "
           "non-converged refits are stops at the optimiser's iteration limit of 500 with the "
           "forecast retained; Sel. = seconds for form selection, Total = seconds for selection "
           "and all refits of both models, one process per catchment on an Apple M4 Pro")
    write("tab-convergence.tex",
          table("Selected baseline forms, convergence and run time per catchment",
                "tab:app-convergence", "llrlrrr", header, rows, src))


def wy2025_table():
    df = pd.read_csv(RES / "wy2025_context.csv")
    rows = []
    for g in GAUGES:
        s = df[df.gauge.astype(str).str.zfill(8) == g].iloc[0]
        rows.append(f"    {GAUGE_NAME[g]:<12} & {int(s['n_pretest_wy'])} & "
                    f"{s['q_mean_wy2025']:.2f} & {s['q_mean_pct_rank']:.0f} & "
                    f"{s['q_median_wy2025']:.2f} & {s['q_median_pct_rank']:.0f} & "
                    f"{s['q_max_wy2025']:.1f} & {s['q_max_pct_rank']:.0f} & "
                    f"{s['p_total_wy2025']:.0f} & {s['p_total_pct_rank']:.0f} \\\\")
    header = ("Catchment    & WYs & $\\bar{Q}$ & Pct & $\\tilde{Q}$ & Pct & "
              "$Q_{\\max}$ & Pct & $P$ & Pct")
    src = ("own results; WYs = complete pre-test water years; $\\bar{Q}$, $\\tilde{Q}$, "
           "$Q_{\\max}$ = mean, median and maximum daily discharge of WY 2025 in mm/day; "
           "$P$ = total precipitation of WY 2025 in mm; Pct = percentile rank of the WY 2025 "
           "value among the pre-test water years")
    write("tab-wy2025-context.tex",
          table("Water year 2025 relative to the pre-test water years of each catchment",
                "tab:app-wy2025", "lrrrrrrrrr", header, rows, src, "4.5pt"))


def event_frequency_table():
    """Appendix B: counts from `python code/data/event_descriptions.py` (358 daily origins)."""
    import subprocess, sys, io
    out = subprocess.run([sys.executable, "event_descriptions.py"], cwd=config.DATA_DIR,
                         capture_output=True, text=True, check=True).stdout
    start = out.index("   gauge")
    block = out[start:].split("\n\n")[0]
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    rows = []
    for g in GAUGES:
        s = df[df.gauge.astype(str).str.zfill(8) == g].iloc[0]
        rows.append(f"    {GAUGE_NAME[g]:<12} & {s['p_mean_mm_d']:.2f} & "
                    f"{s['heavy_threshold_mm']:.1f} & {int(s['heavy'])} & "
                    f"{int(s['rain-on-snow'])} & {int(s['dry'])} & {int(s['snowpack'])} & "
                    f"{int(s['none'])} \\\\")
    header = ("Catchment    & $\\bar{P}$ & Threshold & Heavy & Rain-on-snow & Dry & "
              "Snowpack & Empty")
    src = ("own results; $\\bar{P}$ = pre-test mean daily precipitation in mm/day; Threshold = "
           "$5\\bar{P}$ in mm (heavy precipitation); counts = number of the 358 daily origins of WY 2025 (1 October "
           "2024 to 23 September 2025) whose event block contains at least one line of the "
           "type; Empty = block reads \\enquote{None.}")
    write("tab-event-frequency.tex",
          table("Event-type frequencies over all 358 daily origins per catchment",
                "tab:app-event-frequency", "lrrrrrrr", header, rows, src, "4.5pt"))
    assert "Leakage test passed for all 358 origins x 5 catchments." in out
    (OUT / "leakage-test-output.txt").write_text(out.split("\n\n")[0] + "\n")


if __name__ == "__main__":
    by_gauge_tables("metrics_by_gauge.csv", "", "")
    by_gauge_tables("metrics_by_gauge_approved.csv", "_approved", " (approved days only)")
    regime_tables()
    pooled_approved()
    convergence_table()
    wy2025_table()
    event_frequency_table()
