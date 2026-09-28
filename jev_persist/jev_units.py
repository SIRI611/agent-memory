"""Query-time Jev scores for write-time units (facts and rules from memory_rules).

State is the query only (optionally with query-side cues, see --state-field).
Each boolean question carries one fact or one rule condition. No task label
field is sent.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from jev_persist import query_relevance as qr
from jev_persist.corpus import ROOT
from jev_persist.memory_rules import load_rows
from jev_persist.routing import corpus

CACHE = ROOT / "results" / "v2" / "jev_units.jsonl"
MODEL = qr.MODEL
BATCH = 100
FACT_POLICY = "v0.3-fact-necessity"
RULE_POLICY = "v0.3-rule-applicability"
FACT_QUESTION = (
    "Stored fact about the user: \"{text}\"\n"
    "Is this fact required to answer the request in the state safely, correctly, "
    "or appropriately (for example, it calls for a warning, a restriction, or a different recommendation)?"
)
RULE_QUESTION = "Condition on the user's request: \"{text}\"\nDoes the user request in the state satisfy this condition?"
QUESTIONS = {FACT_POLICY: FACT_QUESTION, RULE_POLICY: RULE_QUESTION}
_lock = threading.Lock()


def key(policy: str, state: str, item_id: str) -> str:
    return qr.sha(f"{policy}\n{MODEL}\n{qr.sha(state)}\n{item_id}")


@lru_cache(maxsize=1)
def cache() -> dict[str, float]:
    return load_cache()


def load_cache() -> dict[str, float]:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row["p"]
    return out


def items(kinds: tuple[str, ...] = ("background", "target"), turns: set[str] | None = None) -> list[tuple[str, str, str]]:
    """(policy, item_id, text) for every fact and every rule condition of the chosen turns."""
    out = []
    for text_hash, row in load_rows().items():
        if row["kind"] not in kinds and not (turns and text_hash in turns):
            continue
        for unit in row["units"]:
            out.append((FACT_POLICY, unit["fact_id"], unit["fact"]))
            for rule in unit["rules"]:
                out.append((RULE_POLICY, rule["rule_id"], rule["when"]))
    return out


def needed_turns() -> dict[str, set[str]]:
    """Target turns each query must be scored against: its own and its negative-control partner's."""
    from jev_persist.offline_report import negative_partners

    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    need: dict[str, set[str]] = {}
    for t in targets:
        need.setdefault(t["query"], set()).add(t["text_hash"])
    for stored_id, query_id in negative_partners().items():
        need.setdefault(by_id[query_id]["query"], set()).add(by_id[stored_id]["text_hash"])
    return need


def states_for(query: str) -> str:
    from jev_persist.query_plan import planned_state

    return planned_state(query)


def post(body: dict, api_key: str) -> dict:
    """Overload errors on the gateway are instantaneous and random, so retry fast."""
    data = json.dumps(body).encode()
    for attempt in range(80):
        qr._pace()
        request = urllib.request.Request(
            qr.URL, data=data, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504, 529}:
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(4.0, 0.5 * 1.5 ** attempt))
    raise RuntimeError("Jev request failed after retries")


def work(field: str, policies: set[str], done: dict, all_targets: bool = False) -> list[tuple[str, list[tuple[str, str, str]]]]:
    targets, _, _ = corpus()
    need = needed_turns()
    jobs = []
    for query in sorted({t["query"] for t in targets}):
        state = query if field == "query" else states_for(query)
        pool = items() if all_targets else items(("background",), need[query])
        pool = [i for i in pool if i[0] in policies]
        todo = [i for i in pool if key(i[0], state, i[1]) not in done]
        for start in range(0, len(todo), BATCH):
            jobs.append((state, todo[start:start + BATCH]))
    return jobs


def run_batch(state: str, batch: list[tuple[str, str, str]], api_key: str) -> tuple[list[dict], float]:
    questions = {
        f"q{i}": {"type": "boolean", "instructions": QUESTIONS[policy].format(text=text)}
        for i, (policy, _, text) in enumerate(batch)
    }
    payload = post({"model": MODEL, "state": "User request:\n" + state, "questions": questions}, api_key)
    cost = float(payload.get("providerMetadata", {}).get("gateway", {}).get("cost") or 0)
    rows = [
        {"key": key(policy, state, item_id), "policy": policy, "item_id": item_id,
         "state_hash": qr.sha(state), "p": payload["answers"][f"q{i}"]["probability"]}
        for i, (policy, item_id, _) in enumerate(batch)
    ]
    return rows, cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pace", type=float, default=0.25)
    parser.add_argument("--state-field", choices=["query", "planned"], default="query")
    parser.add_argument("--policies", default=f"{FACT_POLICY},{RULE_POLICY}")
    parser.add_argument("--all-targets", action="store_true", help="also score every other target turn (multi-fact pool)")
    args = parser.parse_args()
    done = load_cache()
    jobs = work(args.state_field, set(args.policies.split(",")), done, args.all_targets)
    print(f"items={len(items())} cached={len(done)} pending_calls={len(jobs)}", flush=True)
    if not args.execute:
        return
    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("AI_GATEWAY_API_KEY is missing")
    original_pace = qr._pace
    qr._pace = lambda gap=args.pace: original_pace(gap)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    total, finished = 0.0, 0
    with CACHE.open("a") as handle, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_batch, s, b, api_key) for s, b in jobs]
        for future in as_completed(futures):
            try:
                rows, cost = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed batch: {error}", flush=True)
                continue
            with _lock:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))
                handle.flush()
            finished += 1
            total += cost
            if finished % 200 == 0:
                print(f"calls {finished}/{len(jobs)} gateway_cost=${total:.6f}", flush=True)
    print(f"finished {finished}/{len(jobs)} gateway_cost=${total:.6f}", flush=True)


if __name__ == "__main__":
    main()
