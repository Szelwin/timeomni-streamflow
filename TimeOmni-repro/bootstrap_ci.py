#!/usr/bin/env python3
"""Bootstrap confidence intervals for the reproduction check (thesis Table "Reproduction check").

  python bootstrap_ci.py            # reads results/*_sample100_task3_results.json

Recipe: the resampling unit is the test item. For each of the four runs the 100 scored items
(elements 1.. of the file written by score.sh, i.e. by the authors' eval/get_score.py) are
resampled with replacement B = 100,000 times with numpy.random.default_rng(42). In every
resample the success rate is the share of items with is_valid == True and the MAE is the mean
`reward` (the item's MAE) over the valid items of that resample, as in the authors' valid_score;
invalid answers therefore enter the success rate but not the MAE. Intervals are the 2.5th and
97.5th percentiles of the B resampled values. Output: results/bootstrap_ci.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = [("timeomni", "id"), ("timeomni", "ood"), ("qwen", "id"), ("qwen", "ood")]
B, SEED = 100_000, 42


def main():
    rows = []
    for model, split in RUNS:
        scored = json.load(open(HERE / "results" / f"{model}_{split}_sample100_task3_results.json"))
        summary, items = scored[0]["event_aware_forecasting"], scored[1:]
        mae_item = np.array([float(x["reward"]) for x in items])
        valid = np.array([str(x["is_valid"]) == "True" for x in items])
        assert len(items) == summary["total_samples"] and valid.sum() == summary["valid_samples"]
        assert abs(mae_item[valid].mean() - summary["valid_score"]) < 1e-9
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(items), size=(B, len(items)))
        v = valid[idx]
        sr = 100 * v.mean(axis=1)
        mae = (mae_item[idx] * v).sum(axis=1) / v.sum(axis=1)
        lo_m, hi_m = np.percentile(mae, [2.5, 97.5])
        lo_s, hi_s = np.percentile(sr, [2.5, 97.5])
        rows.append({"model": model, "split": split, "n": len(items), "valid": int(valid.sum()),
                     "mae_valid": round(float(mae_item[valid].mean()), 2),
                     "mae_ci_low": round(float(lo_m), 1), "mae_ci_high": round(float(hi_m), 1),
                     "sr_pct": round(100 * float(valid.mean()), 1),
                     "sr_ci_low": round(float(lo_s), 1), "sr_ci_high": round(float(hi_s), 1),
                     "draws": B, "seed": SEED})
    out = HERE / "results" / "bootstrap_ci.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
