"""Query-side risk planning (PACE conflict planner, adapted).

DeepSeek sees only the request, never the memory or any task label, and names
up to three kinds of user circumstances that would change a careful answer.
The cues are appended to the Jev state; the memory questions are unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from jev_persist.budget import Budget, fetch_balance
from jev_persist.corpus import ROOT, load_config
from jev_persist.memory_rules import sha
from jev_persist.routing import corpus
from jev_persist.run_deepseek import post

PLANS = ROOT / "results" / "v2" / "plans.jsonl"
MAX_CUES = 3
SYSTEM = """A user sends the request below to an AI assistant that keeps long-term memory about the user. You cannot see that memory.
Name up to 3 kinds of facts about the user (for example health conditions, medications, diet, possessions, devices, upcoming events, legal or immigration status, finances, relationships, dependents, religion) that, if they were true, would make a careful assistant give a different answer or add a warning. Prefer non-obvious circumstances over the ones the request already mentions. One short phrase each.
Return JSON only: {"cues": ["...", "..."]}"""


def request_body(query: str, config: dict) -> dict:
    return {
        "model": config["answer_model"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}],
        "thinking": config["thinking"],
        "max_tokens": 256,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


@lru_cache(maxsize=1)
def plans() -> dict[str, list[str]]:
    out = {}
    if PLANS.exists():
        for line in PLANS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["query"]] = row["cues"]
    return out


def planned_state(query: str) -> str:
    cues = plans()[query]
    if not cues:
        return query
    return query + "\n\nUser circumstances that could change the right answer:\n" + "\n".join(f"- {c}" for c in cues)


def parse_cues(content: str) -> list[str]:
    try:
        cues = json.loads(content).get("cues") or []
    except (json.JSONDecodeError, AttributeError):
        return []
    return [str(c).strip() for c in cues if str(c).strip()][:MAX_CUES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config()
    targets, _, _ = corpus()
    queries = sorted({t["query"] for t in targets} - set(plans()))
    print(f"queries to plan: {len(queries)}")
    if not args.execute:
        return
    api_key = os.environ["DEEPSEEK_API_KEY"]
    if not fetch_balance(api_key).get("is_available"):
        raise SystemExit("DeepSeek account is not available")
    budget = Budget.from_config()
    if budget.spent() + 1 > budget.limit:
        raise SystemExit("budget")

    def one(query: str) -> dict:
        payload = post(api_key, request_body(query, config))
        usage = payload.get("usage") or {}
        content = payload["choices"][0]["message"].get("content") or ""
        return {"query": query, "content": content, "cues": parse_cues(content), "usage": usage,
                "model": str(payload.get("model"))}

    PLANS.parent.mkdir(parents=True, exist_ok=True)
    with PLANS.open("a") as handle, ThreadPoolExecutor(12) as pool:
        for future in as_completed([pool.submit(one, q) for q in queries]):
            row = future.result()
            budget.record(request_id=sha(f"plan\n{row['query']}\n{SYSTEM}"), model=row["model"], usage=row["usage"],
                          task_id=-1, condition="query_plan", name="plan")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"planned; spent=${budget.spent()}")


if __name__ == "__main__":
    main()
