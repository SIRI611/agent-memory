"""Score cached pilot answers with the InMind judge prompts.

Uses the same $100 DeepSeek budget guard as the answer run.
"""

from __future__ import annotations

import json

from jev_persist.budget import Budget, fetch_balance
from jev_persist.corpus import ROOT
from jev_persist.run_deepseek import post, request_id

REQUESTS = ROOT / "results" / "framework" / "requests"
ANSWERS = ROOT / "results" / "framework" / "deepseek_responses.jsonl"
OUTPUT = ROOT / "results" / "framework" / "deepseek_judgements.jsonl"


def load_answers() -> dict[tuple[int, str, str], str]:
    found = {}
    for line in ANSWERS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[(row["task_id"], row["condition"], row["name"])] = row.get("content") or ""
    return found


def load_done() -> set[str]:
    if not OUTPUT.exists():
        return set()
    return {
        json.loads(line)["request_id"]
        for line in OUTPUT.read_text().splitlines()
        if line.strip()
    }


def judge_bodies(record: dict, answers: dict[tuple[int, str, str], str]) -> list[tuple[str, dict]]:
    task_id = record["task_id"]
    condition = record["condition"]["id"]
    naive = answers[(task_id, condition, "naive")]
    indirect = answers[(task_id, condition, "query")]
    filled = {"naive": naive, "application": indirect, "answer_only": indirect, "target_recall": None}
    bodies = []
    for name, body in record["judges"].items():
        payload = json.loads(json.dumps(body))
        user = payload["messages"][1]["content"]
        replacement = filled[name]
        if replacement is not None:
            user = user.replace("{answer}", replacement)
        payload["messages"][1]["content"] = user
        bodies.append((name, payload))
    return bodies


def parse_score(content: str) -> int | None:
    try:
        score = json.loads(content).get("score")
    except json.JSONDecodeError:
        return None
    if score in {0, 1, "0", "1"}:
        return int(score)
    return None


def execute() -> None:
    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    budget = Budget.from_config()
    balance = fetch_balance(api_key)
    if not balance.get("is_available"):
        raise SystemExit(f"DeepSeek account is not available for calls: {balance}")
    answers = load_answers()
    done = load_done()
    sent = 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a") as handle:
        for path in sorted(REQUESTS.glob("*/*.json")):
            record = json.loads(path.read_text())
            for name, body in judge_bodies(record, answers):
                key = request_id("judge", record["task_id"], record["condition"]["id"], name, body)
                if key in done:
                    continue
                allowed, reservation = budget.allow(body)
                if not allowed:
                    raise SystemExit(
                        f"stopped: reservation ${reservation} would pass ${budget.limit} "
                        f"(spent ${budget.spent()})"
                    )
                payload = post(api_key, body)
                usage = payload.get("usage") or {}
                model = str(payload.get("model") or body.get("model"))
                content = payload["choices"][0]["message"].get("content") or ""
                recorded = budget.record(
                    request_id=key,
                    model=model,
                    usage=usage,
                    task_id=record["task_id"],
                    condition=record["condition"]["id"],
                    name=name,
                )
                row = {
                    "request_id": key,
                    "task_id": record["task_id"],
                    "condition": record["condition"]["id"],
                    "name": name,
                    "score": parse_score(content),
                    "content": content,
                    "estimated_usd": recorded["estimated_usd"],
                    "spent_after_usd": recorded["spent_after_usd"],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done.add(key)
                sent += 1
                if sent % 20 == 0:
                    print(f"judged {sent} spent=${recorded['spent_after_usd']}", flush=True)
    print(f"new_judgements={sent} spent=${budget.spent()} limit=${budget.limit}")


if __name__ == "__main__":
    execute()
