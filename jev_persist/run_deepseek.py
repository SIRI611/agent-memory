"""Send the frozen pilot requests to the DeepSeek API.

The default is a no-op. Passing --execute is the only path that spends balance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from jev_persist.budget import Budget, fetch_balance
from jev_persist.corpus import ROOT

REQUESTS = ROOT / "results" / "framework" / "requests"
CACHE = ROOT / "results" / "framework" / "deepseek_cache.jsonl"
OUTPUT = ROOT / "results" / "framework" / "deepseek_responses.jsonl"


def request_id(kind: str, task_id: int, condition: str, name: str, body: dict) -> str:
    payload = json.dumps(
        {"kind": kind, "task_id": task_id, "condition": condition, "name": name, "body": body},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def iter_answer_requests():
    for path in sorted(REQUESTS.glob("*/*.json")):
        record = json.loads(path.read_text())
        for name, body in record["answers"].items():
            yield record["task_id"], record["condition"]["id"], name, body


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    found = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            found[row["request_id"]] = row
    return found


def post(api_key: str, body: dict, timeout: int = 120) -> dict:
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode()[:500]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last_error)


def execute() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    budget = Budget.from_config()
    try:
        balance = fetch_balance(api_key)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"could not read DeepSeek balance: {type(exc).__name__}: {exc}") from exc
    if not balance.get("is_available"):
        raise SystemExit(f"DeepSeek account is not available for calls: {balance}")
    cached = load_cache()
    sent = 0
    stopped = False
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as cache_handle, OUTPUT.open("a") as output_handle:
        for task_id, condition, name, body in iter_answer_requests():
            key = request_id("answer", task_id, condition, name, body)
            if key in cached:
                continue
            allowed, reservation = budget.allow(body)
            if not allowed:
                print(
                    f"stopped before task={task_id} {condition} {name}: "
                    f"reservation ${reservation} would pass the "
                    f"${budget.limit} budget (spent ${budget.spent()})",
                    flush=True,
                )
                stopped = True
                break
            payload = post(api_key, body)
            usage = payload.get("usage") or {}
            model = str(payload.get("model") or body.get("model"))
            recorded = budget.record(
                request_id=key,
                model=model,
                usage=usage,
                task_id=task_id,
                condition=condition,
                name=name,
            )
            row = {
                "request_id": key,
                "task_id": task_id,
                "condition": condition,
                "name": name,
                "response_model": model,
                "usage": usage,
                "estimated_usd": recorded["estimated_usd"],
                "spent_after_usd": recorded["spent_after_usd"],
                "content": payload["choices"][0]["message"].get("content"),
            }
            line = json.dumps(row, ensure_ascii=False)
            cache_handle.write(line + "\n")
            output_handle.write(line + "\n")
            cache_handle.flush()
            sent += 1
            print(
                f"sent {sent} task={task_id} {condition} {name} "
                f"cost=${recorded['estimated_usd']} spent=${recorded['spent_after_usd']}",
                flush=True,
            )
    print(f"new_calls={sent} stopped={stopped} spent=${budget.spent()} limit=${budget.limit}")
    return sent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the DeepSeek API. Omit this flag to leave the prepared requests on disk.",
    )
    args = parser.parse_args()
    if not args.execute:
        pending = sum(1 for _ in iter_answer_requests())
        print(f"dry-run answer requests={pending}. Pass --execute to call DeepSeek.")
        return
    execute()


if __name__ == "__main__":
    main()
