"""Query-free Jev admission scores for InMind targets vs background user turns.

The future query, the bridge explanation, and the answer model are not used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "jev_admission_cache.jsonl"
POLICY_VERSION = "v0.1-admission"
MODEL = "typesafe-ai/jev"
URL = "https://ai-gateway.vercel.sh/v1/evaluate"

QUESTIONS = {
    "memory_type": {
        "type": "choice",
        "instructions": "Classify this memory. Use only the text itself.",
        "criteria": {
            "stable_constraint": "a durable constraint, risk, allergy, medical, legal, financial, or safety limit that should change future advice",
            "preference": "a stable like or dislike",
            "identity": "who the user is",
            "long_term_goal": "an ongoing aim",
            "ongoing_commitment": "a responsibility or plan already underway",
            "resource": "something the user has, owns, or can use",
            "episodic_event": "a one-off event",
            "transient_state": "a temporary mood, plan for today, or short-lived status",
            "other": "none of the above",
        },
    },
    "stable": {
        "type": "boolean",
        "instructions": "Is this information likely to remain relevant across future sessions rather than only the current interaction?",
    },
    "decision_relevant": {
        "type": "boolean",
        "instructions": "Could this information materially change a future recommendation, decision, or action involving this user, even if a later question never names it?",
    },
    "impact": {
        "type": "score",
        "instructions": "If this fact were forgotten while giving later advice, how serious would the mistake be?",
        "criteria": ["negligible", "minor", "meaningful", "major", "critical"],
    },
    "user_specific": {
        "type": "boolean",
        "instructions": "Is this a persistent fact about the user, their environment, obligations, preferences, or constraints, rather than a general question or small talk?",
    },
}


def clean_text(text: str) -> str:
    return re.split(r"\n---TIMESTAMP:", text, maxsplit=1)[0].strip()


def cache_key(text: str) -> str:
    payload = f"{POLICY_VERSION}\n{MODEL}\n{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def load_items() -> list[dict]:
    tasks = [
        json.loads(line)
        for line in (DATA / "inmind.jsonl").read_text().splitlines()
        if line.strip()
    ]
    background = [
        json.loads(line)
        for line in (DATA / "lme_s_background.jsonl").read_text().splitlines()
        if line.strip()
    ]
    items = []
    seen = set()
    for task in tasks:
        text = clean_text(task["user_message"])
        key = cache_key(text)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "text_hash": key,
                "label": "target",
                "text": text,
                "task_id": task["task_id"],
                "domain": task["domain"],
            }
        )
    for turn in background:
        if turn["role"] != "user":
            continue
        text = clean_text(turn["content"])
        key = cache_key(text)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "text_hash": key,
                "label": "background",
                "text": text,
                "turn_id": turn["id"],
            }
        )
    return items


def load_done() -> set[str]:
    if not OUT.exists():
        return set()
    done = set()
    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            done.add(row["text_hash"])
    return done


_PACE = threading.Lock()
_NEXT_AT = 0.0


def _pace() -> None:
    global _NEXT_AT
    with _PACE:
        now = time.time()
        wait = _NEXT_AT - now
        _NEXT_AT = max(now, _NEXT_AT) + 0.4
    if wait > 0:
        time.sleep(wait)


def evaluate(api_key: str, text: str) -> dict:
    _pace()
    body = {
        "model": MODEL,
        "state": f"Memory text:\n{text}",
        "questions": QUESTIONS,
    }
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
                payload["_status"] = response.status
                return payload
        except urllib.error.HTTPError as error:
            detail = error.read().decode()[:2000]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last_error)


def main() -> None:
    api_key = os.environ["AI_GATEWAY_API_KEY"]
    items = load_items()
    done = load_done()
    pending = [item for item in items if item["text_hash"] not in done]
    print(f"items={len(items)} cached={len(done)} pending={len(pending)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures = 0

    def work(item: dict) -> dict:
        started = time.time()
        try:
            payload = evaluate(api_key, item["text"])
            cost = (
                payload.get("providerMetadata", {})
                .get("gateway", {})
                .get("cost")
            )
            return {
                "ok": True,
                "text_hash": item["text_hash"],
                "label": item["label"],
                "task_id": item.get("task_id"),
                "domain": item.get("domain"),
                "turn_id": item.get("turn_id"),
                "char_len": len(item["text"]),
                "model": payload.get("model"),
                "answers": payload.get("answers"),
                "usage": payload.get("usage"),
                "cost": cost,
                "latency_ms": int((time.time() - started) * 1000),
            }
        except Exception as error:  # noqa: BLE001
            return {
                "ok": False,
                "text_hash": item["text_hash"],
                "label": item["label"],
                "error": str(error)[:1000],
            }

    with OUT.open("a") as handle:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(work, item) for item in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                if not row["ok"]:
                    failures += 1
                    print(f"FAIL {index}/{len(pending)} {row['error'][:180]}", flush=True)
                    if "customer_verification" in row["error"] or "HTTP 401" in row["error"]:
                        raise SystemExit(row["error"])
                elif index % 25 == 0 or index == len(pending):
                    print(f"done {index}/{len(pending)} failures={failures}", flush=True)
    print(f"finished failures={failures}", flush=True)


if __name__ == "__main__":
    main()
