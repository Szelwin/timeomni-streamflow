#!/usr/bin/env python3
"""Collect outputs/llm/*.jsonl into forecasts/llm_forecasts.csv (long format) and print
the valid-output rate per model x context.

Point forecast = the parsed generation; an invalid or missing generation is a failure
and is replaced by persistence (last observed value, repeated) for scoring, as in the
Exposé. Columns: gauge, origin, model, context, lead (1..7), target_date, y_pred, y_true,
q_estimated, valid, parse_reason, fallback, finish_reason, completion_tokens.
"""
import json
from collections import Counter

import pandas as pd

import config


def main():
    rows, summary = [], []
    for model in config.MODELS:
        for ctx in config.CONTEXTS:
            path = config.output_path(model, ctx)
            if not path.exists():
                print(f"missing {path}"); continue
            recs = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line); recs[r["id"]] = r
            expected = [config.job_id(g, o, model, ctx) for g in config.GAUGES for o in config.ORIGINS]
            reasons = Counter()
            missing = [jid for jid in expected if jid not in recs]
            if missing:
                print(f"{model}/{ctx}: {len(missing)} of {len(expected)} jobs not run yet")
            for jid in expected:
                r = recs.get(jid)
                if r is None:
                    continue
                valid = r["parse_reason"] == "ok"
                reasons[r["parse_reason"]] += 1
                pred = r["parsed"] if valid else [r["last_obs"]] * config.HORIZON
                origin = pd.Timestamp(r["origin"])
                for lead in range(1, config.HORIZON + 1):
                    rows.append(dict(
                        gauge=r["gauge"], origin=r["origin"], model=model, context=ctx, lead=lead,
                        target_date=str((origin + pd.Timedelta(days=lead)).date()),
                        y_pred=pred[lead - 1], y_true=r["y_true"][lead - 1],
                        q_estimated=r["q_estimated"][lead - 1], valid=valid,
                        parse_reason=r["parse_reason"], fallback=not valid,
                        finish_reason=r["_meta"].get("finish_reason"),
                        completion_tokens=(r["_meta"].get("usage") or {}).get("completion_tokens"),
                        negative=valid and min(r["parsed"]) < 0,
                    ))
            n = sum(reasons.values())
            summary.append(dict(model=model, context=ctx, n=n, valid=reasons["ok"],
                                success_rate=round(100 * reasons["ok"] / n, 1) if n else None,
                                reasons=dict(reasons)))
    df = pd.DataFrame(rows)
    config.FORECAST_DIR.mkdir(exist_ok=True)
    out = config.FORECAST_DIR / "llm_forecasts.csv"
    df.to_csv(out, index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\n{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
