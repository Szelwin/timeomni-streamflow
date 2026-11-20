#!/usr/bin/env python3
"""Run the LLM part of the main experiment on the RunPod endpoints.

  python run_llm.py --model timeomni --ctx ctx            # 5 x 52 = 260 generations
  python run_llm.py --model qwen --ctx noctx --gauges 01047000 --n_origins 4   # pilot
  python run_llm.py --model timeomni --ctx ctx --dry_run  # build prompts, write nothing

Output: outputs/llm/{model}_{ctx}.jsonl, one record per job: prompt, raw generation
(`pred_rat`, None on request failure), parsed values, y_true, _meta. Resumable: jobs
already in the file with a non-None generation are skipped; failed ones are retried.
"""
import argparse
import asyncio
import json
import sys
import time

from tqdm import tqdm

import config
import prompts
from parse import parse_forecast


def load_done(path):
    done = set()
    if path.exists():
        kept = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("pred_rat") is not None:
                        done.add(r["id"]); kept.append(line)
        with open(path, "w", encoding="utf-8") as f:   # drop failed records -> retried
            f.writelines(kept)
    return done


async def worker(queue, ep, session, out_path, lock, bar, stats, max_retries):
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done(); return
        res = await ep.complete(session, job["messages"], max_retries)
        values, reason = parse_forecast(res["text"], config.HORIZON)
        record = {**job, "pred_rat": res["text"], "parsed": values, "parse_reason": reason,
                  "_meta": res["meta"]}
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["errors"] += res["meta"]["error"] is not None
            stats["invalid"] += reason != "ok" and res["meta"]["error"] is None
            bar.update(1); bar.set_postfix(errors=stats["errors"], invalid=stats["invalid"], refresh=False)
        queue.task_done()


async def run(args):
    gauges = args.gauges or config.GAUGES
    origins = config.ORIGINS[:args.n_origins] if args.n_origins else config.ORIGINS
    jobs = prompts.all_jobs([args.model], [args.ctx], gauges, origins)
    out_path = config.output_path(args.model, args.ctx)
    if args.dry_run:
        n_tok = None
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(config.TOKENIZER_DIR))
            n_tok = [len(tok(tok.apply_chat_template(j["messages"], add_generation_prompt=True, tokenize=False),
                             add_special_tokens=False)["input_ids"]) for j in jobs]
        except Exception as e:  # noqa: BLE001
            print("tokenizer unavailable:", e)
        print(f"dry run: {len(jobs)} jobs for {args.model}/{args.ctx}; ids unique: {len({j['id'] for j in jobs}) == len(jobs)}")
        if n_tok:
            print(f"prompt tokens: min {min(n_tok)} max {max(n_tok)} -> min generation budget {config.CONTEXT_BUDGET - max(n_tok)}")
        print("\n--- example (last job) ---\n" + jobs[-1]["messages"][1]["content"])
        return 0

    from client import Endpoint, load_env, make_session
    load_env()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{args.model}/{args.ctx}: jobs={len(jobs)} done={len(done)} todo={len(todo)} -> {out_path}")
    if not todo:
        return 0
    ep = Endpoint(args.model)
    queue = asyncio.Queue()
    for j in todo:
        queue.put_nowait(j)
    for _ in range(args.concurrency):
        queue.put_nowait(None)
    bar = tqdm(total=len(jobs), initial=len(done), desc=f"{args.model}/{args.ctx}", unit="gen", dynamic_ncols=True)
    stats = {"errors": 0, "invalid": 0}
    lock = asyncio.Lock()
    t0 = time.time()
    async with make_session(args.timeout) as session:
        await asyncio.gather(*[worker(queue, ep, session, out_path, lock, bar, stats, args.max_retries)
                               for _ in range(args.concurrency)])
    bar.close()
    print(f"finished in {time.time() - t0:.0f}s: {len(todo)} requested, {stats['errors']} request errors "
          f"(retried on re-run), {stats['invalid']} invalid answers (kept; persistence fallback at aggregation)")
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=config.MODELS, required=True)
    p.add_argument("--ctx", choices=config.CONTEXTS, required=True)
    p.add_argument("--gauges", nargs="*", default=None)
    p.add_argument("--n_origins", type=int, default=0, help="first N origins only (pilot); 0 = all 52")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--dry_run", action="store_true")
    sys.exit(asyncio.run(run(p.parse_args())))
