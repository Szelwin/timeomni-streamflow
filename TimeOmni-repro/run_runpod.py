#!/usr/bin/env python3
"""Drop-in replacement for TimeOmni-1/eval/inference.py that runs inference on a
RunPod Serverless vLLM endpoint (OpenAI-compatible route) instead of a local GPU.

Faithful to the original where it matters for the metrics:
  * same item filtering (--task_type) and the same seeded shuffle (random.seed(42));
  * same chat messages: [{system: item["system"]}, {user: item["problem"]}];
  * same sampling: temperature=0.1, top_p=0.001, repetition_penalty=1.05;
  * same generation budget: the original runs vLLM with max_model_len=4096 and
    max_tokens=4096, i.e. generation is capped at 4096 - prompt_tokens. The
    RunPod workers are configured with max_model_len=16384, so we reproduce the
    cap explicitly: max_tokens = BUDGET - prompt_tokens, with prompt_tokens
    counted locally with the Qwen2.5 tokenizer (identical vocab + chat template
    for Qwen2.5-7B-Instruct and TimeOmni-1-7B; verified by md5 of vocab.json and
    string-equality of chat_template).
  * same output schema, one JSON object per line, so the unmodified
    TimeOmni-1/eval/get_score.py can score the file with --proc_total 0.

Resumable: items whose question_id is already in the output file are skipped.

Usage (see README.md):
  python run_runpod.py --model timeomni --split id   [--sample_n 100] [--concurrency 16]
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent

MODELS = {
    # name -> (env var holding the endpoint id, model id served by the worker)
    "qwen": ("RUNPOD_QWEN_EP", "Qwen/Qwen2.5-7B-Instruct"),
    "timeomni": ("RUNPOD_TIMEOMNI_EP", "anton-hugging/TimeOmni-1-7B"),
}

SAMPLING = dict(temperature=0.1, top_p=0.001, repetition_penalty=1.05)
CONTEXT_BUDGET = 4096  # == max_model_len used by the authors' eval script


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_message(item):
    return [
        {"role": "system", "content": item["system"]},
        {"role": "user", "content": item["problem"]},
    ]


def load_test_data(test_file, task_type, sample_n, sample_seed):
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # eval/inference.py behaviour
    random.seed(42)
    random.shuffle(data)
    if task_type:
        selected = [t.strip() for t in task_type.split(",")]
        data = [x for x in data if x["task_type"] in selected]
    # Seeded subset. Unlike SAMPLE_N in run-timeomini_test.sh (which samples
    # across all tasks), we sample *after* the task filter so that sample_n is
    # the number of Task 3 items.
    if sample_n > 0:
        random.Random(sample_seed).shuffle(data)
        data = sorted(data[:sample_n], key=lambda x: x["question_id"])
    return data


async def worker(name, queue, session, url, headers, model_id, tok, out_path, lock, stats, max_retries):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        messages = get_message(item)
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_tokens = len(tok(prompt, add_special_tokens=False)["input_ids"])
        body = dict(model=model_id, messages=messages, max_tokens=CONTEXT_BUDGET - prompt_tokens, **SAMPLING)

        text, usage, finish, err = None, None, None, None
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                async with session.post(url, headers=headers, json=body) as resp:
                    if resp.status != 200:
                        err = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                        # Safety net if the worker runs max_model_len=4096 and our
                        # local prompt count is ever off: shave a few tokens and retry.
                        if resp.status == 400 and "context length" in err:
                            body["max_tokens"] -= 8
                        raise RuntimeError(err)
                    j = await resp.json()
                text = j["choices"][0]["message"]["content"]
                finish = j["choices"][0].get("finish_reason")
                usage = j.get("usage")
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = str(e)[:300]
                await asyncio.sleep(min(60, 5 * (attempt + 1)))

        record = {
            "question_id": item["question_id"],
            "problem": item["problem"],
            "pred_rat": text,  # None -> get_score.py counts it as invalid
            "response": item["response"],
            "task_type": item["task_type"],
            "domain": item.get("domain", ""),
            "system": item["system"],
            # extra bookkeeping (ignored by get_score.py)
            "_meta": {
                "model": model_id,
                "prompt_tokens_local": prompt_tokens,
                "usage": usage,
                "finish_reason": finish,
                "max_tokens": body["max_tokens"],
                "error": err,
                "latency_s": round(time.time() - t0, 1),
            },
        }
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["done"] += 1
            if err:
                stats["errors"] += 1
            stats["bar"].update(1)
            stats["bar"].set_postfix(errors=stats["errors"], refresh=False)
        queue.task_done()


async def run(args):
    env_var, model_id = MODELS[args.model]
    ep = os.environ[env_var]
    key = os.environ["RUNPOD_API_KEY"]
    url = f"https://api.runpod.ai/v2/{ep}/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    test_file = HERE / "data" / f"timeomni1_{args.split}_test.json"
    data = load_test_data(test_file, args.task_type, args.sample_n, args.sample_seed)

    tag = f"{args.model}_{args.split}" + (f"_sample{args.sample_n}" if args.sample_n else "") + "_task3"
    out_path = Path(args.output) if args.output else HERE / "outputs" / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("pred_rat") is not None or args.keep_failed:
                        done_ids.add(r["question_id"])
    todo = [x for x in data if x["question_id"] not in done_ids]
    if done_ids and not args.keep_failed:
        # rewrite file without failed records so they can be retried
        kept = []
        with open(out_path, encoding="utf-8") as f:
            kept = [l for l in f if l.strip() and json.loads(l).get("pred_rat") is not None]
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(kept)
    print(f"model={model_id} split={args.split} items={len(data)} already_done={len(done_ids)} todo={len(todo)} -> {out_path}")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    try:  # record the worker's context length for the report
        import urllib.request
        req = urllib.request.Request(url.replace("/chat/completions", "/models"), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            print("endpoint models:", [(m["id"], m.get("max_model_len")) for m in json.load(r)["data"]])
    except Exception as e:  # noqa: BLE001
        print("could not query /models:", e)
    queue = asyncio.Queue()
    for x in todo:
        queue.put_nowait(x)
    for _ in range(args.concurrency):
        queue.put_nowait(None)
    bar = tqdm(total=len(data), initial=len(done_ids), desc=tag, unit="item", dynamic_ncols=True)
    stats = {"done": 0, "errors": 0, "total": len(todo), "t0": time.time(), "bar": bar}
    lock = asyncio.Lock()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    # trust_env: honour HTTPS_PROXY / SSL_CERT_FILE if the environment sets them
    # (harmless on a normal machine).
    ssl_ctx = None
    if os.environ.get("SSL_CERT_FILE"):
        import ssl
        ssl_ctx = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True, connector=connector) as session:
        await asyncio.gather(*[
            worker(tag, queue, session, url, headers, model_id, tok, out_path, lock, stats, args.max_retries)
            for _ in range(args.concurrency)
        ])
    bar.close()
    print(f"finished {tag}: {stats['done']} requested, {stats['errors']} errors "
          f"(errors are retried on the next run)")
    sys.exit(1 if stats["errors"] else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS.keys(), required=True)
    p.add_argument("--split", choices=["id", "ood"], required=True)
    p.add_argument("--task_type", default="event_aware_forecasting")
    p.add_argument("--sample_n", type=int, default=0, help="as SAMPLE_N in run-timeomini_test.sh (0 = all)")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--keep_failed", action="store_true", help="do not retry records that failed earlier")
    p.add_argument("--tokenizer", default=str(HERE / "tokenizer"))
    p.add_argument("--output", default=None)
    p.add_argument("--env", default=str(HERE / ".env"))
    args = p.parse_args()
    load_env(Path(args.env))
    load_env(HERE.parent / ".env")         # repository root .env
    load_env(HERE.parent.parent / ".env")  # thesis-root .env (working copy)
    asyncio.run(run(args))
