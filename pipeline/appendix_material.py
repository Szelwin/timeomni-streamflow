"""Material for Appendices A and E of the thesis.

Writes
  results/appendix_a_example.txt   prompts (both conditions) and the four raw answers for
                                   Carrabassett (01047000), origin 2024-12-17
  results/latency_stats.csv        per model x context: generations, median/mean latency,
                                   completion tokens, finish reasons, wall-clock sum
"""
import json
import statistics as st
from pathlib import Path

import config

GAUGE, ORIGIN = "01047000", "2024-12-17"
OUT = config.HERE / "results"


def rows(model, ctx):
    with open(config.output_path(model, ctx)) as f:
        return [json.loads(l) for l in f]


def main():
    lines, stats = [], []
    for model in config.MODELS:
        for ctx in config.CONTEXTS:
            data = rows(model, ctx)
            lat = [d["_meta"]["latency_s"] for d in data if d["_meta"].get("latency_s") is not None]
            comp = [d["_meta"]["usage"]["completion_tokens"] for d in data
                    if d["_meta"].get("usage")]
            fin = {}
            for d in data:
                fr = d["_meta"].get("finish_reason")
                fin[fr] = fin.get(fr, 0) + 1
            stats.append({"model": model, "context": ctx, "n": len(data),
                          "latency_median_s": round(st.median(lat), 1),
                          "latency_mean_s": round(st.mean(lat), 1),
                          "latency_max_s": round(max(lat), 1),
                          "latency_sum_s": round(sum(lat), 0),
                          "completion_tokens_median": int(st.median(comp)),
                          "completion_tokens_max": max(comp),
                          "finish_reasons": ";".join(f"{k}={v}" for k, v in sorted(fin.items(), key=str))})
            ex = [d for d in data if d["gauge"] == GAUGE and d["origin"] == ORIGIN][0]
            if model == "qwen":   # prompts are identical across models; print once per context
                lines.append(f"===== USER PROMPT ({ctx}) =====\n{ex['messages'][1]['content']}\n")
                if ctx == "noctx":
                    lines.insert(0, f"===== SYSTEM PROMPT =====\n{ex['messages'][0]['content']}\n")
            m = ex["_meta"]
            lines.append(f"===== RAW ANSWER {model} {ctx} (finish={m['finish_reason']}, "
                         f"completion_tokens={m['usage']['completion_tokens']}, latency={m['latency_s']} s, "
                         f"parsed={ex['parsed']}, reason={ex['parse_reason']}) =====\n{ex['pred_rat']}\n")
            lines.append(f"last_obs={ex['last_obs']:.3f}  y_true={[round(v, 3) for v in ex['y_true']]}\n")
    (OUT / "appendix_a_example.txt").write_text("\n".join(lines))
    # Verbatim files for \lstinputlisting in thesis/backmatter/appendices.tex
    app = (config.CODE_DIR.parent / "thesis" / "appendix" if (config.CODE_DIR.parent / "thesis").exists()
           else config.HERE / "results" / "appendix")
    app.mkdir(parents=True, exist_ok=True)
    for model in config.MODELS:
        for ctx in config.CONTEXTS:
            ex = [d for d in rows(model, ctx) if d["gauge"] == GAUGE and d["origin"] == ORIGIN][0]
            if model == "qwen":
                (app / f"prompt-{ctx}.txt").write_text(ex["messages"][1]["content"] + "\n")
                if ctx == "noctx":
                    (app / "prompt-system.txt").write_text(ex["messages"][0]["content"] + "\n")
            (app / f"answer-{model}-{ctx}.txt").write_text(ex["pred_rat"].strip() + "\n")
    import csv
    with open(OUT / "latency_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
