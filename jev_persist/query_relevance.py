"""Query-time Jev relevance: score every stored user turn against each indirect query.

State is the query only. Each question carries one memory text. No bridge
explanation, relation, or entity field is sent. Results are cached per
(policy, query hash, memory hash) so reruns only fill gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from jev_persist.corpus import ROOT, load_config, scored_items

POLICY = "v0.2-query-necessity"
URL = "https://ai-gateway.vercel.sh/v1/evaluate"
MODEL = "typesafe-ai/jev"
BATCH = 25
CACHE = ROOT / "results" / "query_relevance" / "cache.jsonl"
QUESTION = (
    "Earlier, the user said: \"{text}\"\n"
    "Is this fact about the user required to answer the request in the state safely, correctly, "
    "or appropriately (for example, it calls for a warning, a restriction, or a different recommendation)?"
)

_lock = threading.Lock()
_last = [0.0]


def _pace(gap: float = 0.5) -> None:
    with _lock:
        wait = _last[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def pair_key(query: str, memory_hash: str) -> str:
    return sha(f"{POLICY}\n{MODEL}\n{sha(query)}\n{memory_hash}")


def load_cache() -> dict[str, float]:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row["p"]
    return out


def post(body: dict, api_key: str) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(12):
        _pace()
        request = urllib.request.Request(
            URL,
            data=data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504, 529}:
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("Jev request failed after retries")


def pools(config: dict) -> tuple[list[dict], list[dict]]:
    targets, background = scored_items(config)
    unique_targets = {}
    for item in targets:
        unique_targets.setdefault(item["text_hash"], item)
    memories = [{"text_hash": b["text_hash"], "text": b["text"], "kind": "background"} for b in background]
    memories += [
        {"text_hash": t["text_hash"], "text": t["user_message"], "kind": "target"}
        for t in unique_targets.values()
    ]
    return targets, memories


def jobs(
    targets: list[dict],
    memories: list[dict],
    cache: dict,
    field: str = "query",
    own_pool_only: bool = False,
) -> list[tuple[str, list[dict]]]:
    work = []
    for task in targets:
        query = task[field]
        pool = memories
        if own_pool_only:
            pool = [m for m in memories if m["kind"] == "background" or m["text_hash"] == task["text_hash"]]
        todo = [m for m in pool if pair_key(query, m["text_hash"]) not in cache]
        for start in range(0, len(todo), BATCH):
            work.append((query, todo[start:start + BATCH]))
    return work


def run_batch(query: str, batch: list[dict], api_key: str) -> tuple[list[dict], float]:
    questions = {
        f"m{i}": {"type": "boolean", "instructions": QUESTION.format(text=m["text"])}
        for i, m in enumerate(batch)
    }
    payload = post({"model": MODEL, "state": "User request:\n" + query, "questions": questions}, api_key)
    cost = float(payload.get("providerMetadata", {}).get("gateway", {}).get("cost") or 0)
    rows = []
    for i, m in enumerate(batch):
        rows.append(
            {
                "key": pair_key(query, m["text_hash"]),
                "query_hash": sha(query),
                "memory_hash": m["text_hash"],
                "p": payload["answers"][f"m{i}"]["probability"],
            }
        )
    return rows, cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--field", choices=["query", "naive_query"], default="query")
    parser.add_argument("--own-pool-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    targets, memories = pools(config)
    cache = load_cache()
    work = jobs(targets, memories, cache, args.field, args.own_pool_only)
    print(f"field={args.field} policy={POLICY} tasks={len(targets)} memories={len(memories)} cached_pairs={len(cache)} pending_calls={len(work)}", flush=True)
    if not args.execute:
        return
    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("AI_GATEWAY_API_KEY is missing")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    total_cost = 0.0
    with CACHE.open("a") as handle, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_batch, query, batch, api_key) for query, batch in work]
        for future in as_completed(futures):
            try:
                rows, cost = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed batch: {error}", flush=True)
                continue
            with _lock:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
            done += 1
            total_cost += cost
            if done % 50 == 0 or done == len(work):
                print(f"calls {done}/{len(work)} gateway_cost=${total_cost:.6f}", flush=True)
    print(f"finished calls={done}/{len(work)} gateway_cost=${total_cost:.6f}", flush=True)


if __name__ == "__main__":
    main()
