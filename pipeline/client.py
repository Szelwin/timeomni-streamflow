"""One chat completion on a RunPod Serverless vLLM endpoint (OpenAI-compatible route).
Ported from code/TimeOmni-repro/run_runpod.py; same sampling and token budget."""
import asyncio
import os
import ssl
import time

import aiohttp
from transformers import AutoTokenizer

import config


def load_env(path=config.ENV_FILE):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Endpoint:
    def __init__(self, model_key: str, tokenizer=None):
        env_var, self.model_id = config.MODELS[model_key]
        self.url = f"https://api.runpod.ai/v2/{os.environ[env_var]}/openai/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                        "Content-Type": "application/json"}
        self.tok = tokenizer or AutoTokenizer.from_pretrained(str(config.TOKENIZER_DIR))

    def prompt_tokens(self, messages) -> int:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"])

    async def complete(self, session, messages, max_retries=4) -> dict:
        n_prompt = self.prompt_tokens(messages)
        body = dict(model=self.model_id, messages=messages, seed=config.SEED,
                    max_tokens=config.CONTEXT_BUDGET - n_prompt, **config.SAMPLING)
        text, usage, finish, err = None, None, None, None
        t0 = time.time()
        for attempt in range(max_retries):
            try:
                async with session.post(self.url, headers=self.headers, json=body) as resp:
                    if resp.status != 200:
                        err = f"HTTP {resp.status}: {(await resp.text())[:300]}"
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
        return {"text": text, "meta": {"model": self.model_id, "prompt_tokens_local": n_prompt,
                                       "max_tokens": body["max_tokens"], "usage": usage,
                                       "finish_reason": finish, "error": err,
                                       "latency_s": round(time.time() - t0, 1),
                                       "sampling": {**config.SAMPLING, "seed": config.SEED}}}


def make_session(timeout_s=1800):
    ssl_ctx = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"]) if os.environ.get("SSL_CERT_FILE") else None
    connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s), trust_env=True, connector=connector)
