#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SWE-bench Lite predictions via an OpenAI-compatible Chat Completions API (vLLM, Ollama, etc.)."""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import re
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


def extract_diff(response: str | None) -> str | None:
    """Extract a unified diff from model output (same logic as SWE-bench ``extract_diff``)."""
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]


def _split_system_user(text: str) -> tuple[str, str]:
    text = text.rstrip()
    if "\n" not in text:
        return "", text
    system, user = text.split("\n", 1)
    return system, user


def _completion_text(message) -> str:
    c = getattr(message, "content", None) or ""
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        c = "".join(parts)
    if isinstance(c, str) and c.strip():
        return c
    for attr in ("reasoning_content", "reasoning"):
        v = getattr(message, attr, None)
        if v:
            return str(v)
    return ""


def _call_and_record(
    client: OpenAI,
    model: str,
    instance_id: str,
    text: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> dict:
    system, user = _split_system_user(text)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    # Streaming sidesteps Ollama's 10-min HTTP-response wall: tokens arrive as SSE chunks
    # and each chunk resets the wire deadline. try/except so one failed instance writes
    # an empty-patch record and the run continues (relies on append-only JSONL dedup).
    try:
        stream = client.chat.completions.create(stream=True, **kwargs)
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        for ev in stream:
            if not ev.choices:
                continue
            delta = ev.choices[0].delta
            c = getattr(delta, "content", None)
            if c:
                content_chunks.append(c)
                continue
            r = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if r:
                reasoning_chunks.append(r)
        full = "".join(content_chunks) if content_chunks else "".join(reasoning_chunks)
        patch = extract_diff(full) if full else ""
        return {
            "instance_id": instance_id,
            "model_name_or_path": model,
            "full_output": full,
            "model_patch": patch or "",
        }
    except Exception as e:
        return {
            "instance_id": instance_id,
            "model_name_or_path": model,
            "full_output": "",
            "model_patch": "",
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--model",
        required=True,
        help='Model id sent in Chat Completions "model" field',
    )
    parser.add_argument(
        "--dataset-name",
        default=os.environ.get(
            "SWE_BENCH_LITE_DATASET",
            "princeton-nlp/SWE-bench_Lite_oracle",
        ),
        help="HF dataset with columns instance_id and text (default: Lite oracle)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Append-only JSONL (instance_id, model_name_or_path, model_patch, ...)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("SWE_BENCH_MAX_TOKENS", "16384")),
    )
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    base_url = f"http://{args.host}:{args.port}/v1"
    # timeout=1800 (30 min) gives 3x headroom over Ollama's hard 10-min per-request gen cap;
    # also override-able via OPENAI_TIMEOUT env if needed.
    client_timeout = float(os.environ.get("OPENAI_TIMEOUT", "1800"))
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=client_timeout)

    ds = load_dataset(args.dataset_name, split=args.split)
    lens = np.array([len(x["text"]) for x in ds])
    ds = ds.select(np.argsort(lens))

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing.add(json.loads(line)["instance_id"])

    rows = [ds[i] for i in range(len(ds))]
    rows = [r for r in rows if r["instance_id"] not in existing]

    if args.shard_id is not None and args.num_shards is not None:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]

    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        print("No instances to run (empty selection or already in output file).")
        return

    lock = threading.Lock()

    def work(row: dict) -> None:
        rec = _call_and_record(
            client,
            args.model,
            row["instance_id"],
            row["text"],
            args.temperature,
            args.max_tokens,
            args.top_p,
        )
        line = json.dumps(rec, ensure_ascii=False)
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    if args.max_workers <= 1:
        for row in tqdm(rows, desc="swe-bench-lite"):
            work(row)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(work, row) for row in rows]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="swe-bench-lite",
            ):
                fut.result()


if __name__ == "__main__":
    main()
