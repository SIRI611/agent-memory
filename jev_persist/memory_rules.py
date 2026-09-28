"""Write-time memory normalization: atomic user facts plus condition -> action rules.

The extractor sees exactly one stored user turn. It never sees a future query,
the InMind bridge explanation, the relation, or any other task field. Every
stored turn (background and target) goes through the same prompt.

Frozen before any retrieval or answer is scored with these units:
- At most 5 facts per turn and 3 rules per fact. Temperature 0.
- P1 unit score: Jev necessity question on the fact text (same wording as v0.2).
- P2 unit score: mean(necessity, max over rules of Jev applicability of `when`).
- P2 answer context shows the fact, plus the `then` of every rule whose
  applicability is >= 0.5 (TAG: only matched actions reach the executor).
- Top-5 units, same K as every other method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from functools import lru_cache

from jev_persist.budget import Budget, fetch_balance
from jev_persist.corpus import ROOT, load_config
from jev_persist.routing import corpus, unique_targets
from jev_persist.run_deepseek import post

OUT = ROOT / "results" / "v2"
FACTS = OUT / "facts.jsonl"
MAX_FACTS = 5
MAX_RULES = 3

SYSTEM = """You maintain the long-term memory of an AI assistant. You receive ONE message that the user wrote in a past conversation. You will never see what the user asks later.

Step 1. Extract the facts about the user that this message reveals: their situation, health, body, diet, possessions, pets, relationships, family, job, finances, legal or immigration status, location, schedule, upcoming events or plans, beliefs, and constraints. Temporary situations and upcoming events count. Write each fact as one short standalone sentence in the third person ("The user ..."). Keep the specifics (names, dates, products, amounts). At most 5 facts.
If the message reveals nothing about the user (for example a generic question, a request to the assistant, a formatting instruction, or a fragment), return no facts.

Step 2. For each fact, write up to 3 rules for future requests where this fact should change what a careful assistant says, even though the request itself would not mention the fact. Think about non-obvious consequences: safety, medical interactions or test preparation, legal or visa rules, financial risk, scams, device or software compatibility, scheduling conflicts, cultural or religious norms, and effects on other people.
- "when": a condition on a future user request, describing the kind of request (not one specific wording). It must be specific enough that most requests would NOT satisfy it.
- "then": what the assistant should do or mention when the condition holds.
If a fact would not change any answer, give it no rules.

Return JSON only: {"facts": [{"fact": "...", "rules": [{"when": "...", "then": "..."}]}]}"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stored_turns() -> list[dict]:
    """Every stored user turn once: 238 background turns and the unique target turns."""
    _, background, _ = corpus()
    turns = [{"text_hash": b["text_hash"], "text": b["text"], "kind": "background"} for b in background]
    turns += [{"text_hash": t["text_hash"], "text": t["user_message"], "kind": "target"} for t in unique_targets()]
    return turns


def request_body(text: str, config: dict) -> dict:
    return {
        "model": config["answer_model"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "thinking": config["thinking"],
        "max_tokens": 2048,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def parse_units(turn_hash: str, content: str) -> list[dict]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    entries = (payload.get("facts") or [])[:MAX_FACTS]
    # Accept the flattened variant {"facts": ["..."], "rules": [...]} when it names a single fact.
    shared_rules = payload.get("rules") if len(entries) == 1 else None
    units = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"fact": entry, "rules": shared_rules or []}
        fact = str(entry.get("fact") or "").strip() if isinstance(entry, dict) else ""
        if not fact:
            continue
        fact_id = sha(f"{turn_hash}\n{i}\n{fact}")
        rules = []
        for j, rule in enumerate((entry.get("rules") or [])[:MAX_RULES]):
            if not isinstance(rule, dict):
                continue
            when, then = str(rule.get("when") or "").strip(), str(rule.get("then") or "").strip()
            if when and then:
                rules.append({"rule_id": sha(f"{fact_id}\n{j}\n{when}"), "when": when, "then": then})
        units.append({"fact_id": fact_id, "fact": fact, "rules": rules})
    return units


def load_rows() -> dict[str, dict]:
    rows = {}
    if FACTS.exists():
        for line in FACTS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["text_hash"]] = row
    return rows


@lru_cache(maxsize=1)
def units_by_turn() -> dict[str, list[dict]]:
    return {h: row["units"] for h, row in load_rows().items()}


def extract(workers: int) -> None:
    config = load_config()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    if not fetch_balance(api_key).get("is_available"):
        raise SystemExit("DeepSeek account is not available")
    done = load_rows()
    todo = [t for t in stored_turns() if t["text_hash"] not in done]
    print(f"turns={len(stored_turns())} done={len(done)} todo={len(todo)}", flush=True)
    budget = Budget.from_config()
    lock = threading.Lock()
    state = {"spent": budget.spent(), "inflight": Decimal(0)}

    def one(turn: dict) -> dict | None:
        body = request_body(turn["text"], config)
        _, reservation = budget.allow(body)
        with lock:
            if state["spent"] + state["inflight"] + reservation > budget.limit:
                return None
            state["inflight"] += reservation
        try:
            payload = post(api_key, body, timeout=300)
        finally:
            with lock:
                state["inflight"] -= reservation
        usage = payload.get("usage") or {}
        content = payload["choices"][0]["message"].get("content") or ""
        with lock:
            recorded = budget.record(
                request_id=sha(f"facts\n{turn['text_hash']}\n{SYSTEM}"), model=str(payload.get("model")),
                usage=usage, task_id=-1, condition="write_facts", name=turn["kind"],
            )
            state["spent"] = Decimal(recorded["spent_after_usd"])
        return {**turn, "content": content, "units": parse_units(turn["text_hash"], content), "usage": usage}

    OUT.mkdir(parents=True, exist_ok=True)
    with FACTS.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, t) for t in todo]
        for n, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            if row is None:
                print("budget stop", flush=True)
                continue
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)} spent=${state['spent']}", flush=True)
    rows = load_rows()
    facts = sum(len(r["units"]) for r in rows.values())
    rules = sum(len(u["rules"]) for r in rows.values() for u in r["units"])
    empty = {k: sum(1 for r in rows.values() if r["kind"] == k and not r["units"]) for k in ("background", "target")}
    print(f"turns={len(rows)} facts={facts} rules={rules} empty_turns={empty} spent=${budget.spent()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not args.execute:
        print(f"dry run: {len(stored_turns())} turns, {len(load_rows())} extracted")
        return
    extract(args.workers)


if __name__ == "__main__":
    main()
